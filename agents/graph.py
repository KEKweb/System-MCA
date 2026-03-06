"""
agents/graph.py — LangGraph граф агентов.

Поток:
  router_node → context_node → generate_node → critic_node
                                      ↑               |
                                      └── retry ───── ┘

Стриминг происходит прямо во время генерации — пользователь видит
<think>...</think> рассуждения в реальном времени. Если критик
отклоняет ответ — в стрим отправляется специальный SSE-эвент,
фронтенд показывает блок с замечанием, затем стримится улучшенный ответ.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from langgraph.graph import StateGraph, END

from services.llm    import stream_generate, generate_once
from services.router import get_router
from services.rag    import get_rag
from services.search import web_search

MAX_RETRIES        = 2   # максимум retry после первой попытки
CRITIC_PASS_SCORE  = 7   # минимальная оценка критика из 10


@dataclass
class AgentState:
    messages:          list[dict]  = field(default_factory=list)
    query:             str         = ""
    web_search:        bool        = False
    mcp_servers:       list[str]   = field(default_factory=list)
    doc_uuid:          str | None  = None
    chat_uuid:         str         = ""
    client_uuid:       str         = "anonymous"
    encryption_key:    str | None  = None

    topic:             str         = "general"
    role_instruction:  str         = ""

    rag_context:       str         = ""
    web_context:       str         = ""
    enriched_messages: list[dict]  = field(default_factory=list)

async def router_node(state: AgentState) -> AgentState:
    router = await get_router()
    result = await router.route(state.query)
    state.topic            = result["topic"]
    state.role_instruction = result["role"]
    return state

async def context_node(state: AgentState) -> AgentState:
    import asyncio
    tasks = []
    if state.doc_uuid and state.chat_uuid:
        tasks.append(_fetch_rag(state))
    else:
        state.rag_context = ""
    if state.web_search and state.query:
        tasks.append(_fetch_web(state))
    else:
        state.web_context = ""
    if tasks:
        await asyncio.gather(*tasks)
    state.enriched_messages = _inject_context(
        state.messages, state.rag_context, state.web_context
    )
    return state


async def _fetch_rag(state: AgentState):
    rag    = get_rag()
    chunks = await rag.search(
        query=state.query, chat_uuid=state.chat_uuid,
        client_uuid=state.client_uuid, encryption_key=state.encryption_key,
    )
    if chunks:
        state.rag_context = "[Релевантные фрагменты документа]:\n" + "\n---\n".join(chunks)


async def _fetch_web(state: AgentState):
    state.web_context = await web_search(state.query)


def _inject_context(messages, rag_ctx, web_ctx):
    if not messages:
        return messages
    extra = ""
    if rag_ctx: extra += f"\n\n{rag_ctx}"
    if web_ctx: extra += f"\n\n{web_ctx}"
    if not extra:
        return messages
    result = [dict(m) for m in messages]
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "user":
            result[i]["content"] += extra
            break
    return result


# =============================================================================
# PUBLIC API — run_pipeline
# Граф здесь не используется как StateGraph-компилированный объект,
# потому что нам нужен сквозной стриминг через всю цепочку включая retry.
# Логика проще и прозрачнее как прямая async-функция.
# =============================================================================
async def run_pipeline(
    messages:       list[dict],
    web_search_on:  bool       = False,
    mcp_servers:    list[str]  = None,
    doc_uuid:       str | None = None,
    chat_uuid:      str        = "",
    client_uuid:    str        = "anonymous",
    encryption_key: str | None = None,
) -> AsyncIterator[str]:
    """
    Полный пайплайн с живым стримингом и агентом-критиком.

    SSE-эвенты которые отправляются фронту:
      data: {"token": "..."}           — токен ответа (включая <think> теги)
      data: {"critic": {...}}          — результат оценки критика
      data: {"retry": true}            — начинается новая попытка
      data: [DONE]                     — завершение
    """
    query = _extract_last_query(messages)

    router = await get_router()
    route_result   = await router.route(query)
    role_instruction = route_result["role"]

    state = AgentState(
        messages=messages, query=query, web_search=web_search_on,
        mcp_servers=mcp_servers or [], doc_uuid=doc_uuid,
        chat_uuid=chat_uuid, client_uuid=client_uuid,
        encryption_key=encryption_key,
    )
    state = await context_node(state)
    current_messages = state.enriched_messages

    retry_count = 0

    while True:
        full_answer = ""
        async for chunk in stream_generate(
            messages=current_messages,
            role_instruction=role_instruction,
            thinking=True,
        ):
            # chunk — SSE строка вида "data: {...}\n\n"
            # Пробрасываем токены напрямую в фронт
            if chunk == "data: [DONE]\n\n":
                break
            yield chunk

            # Параллельно собираем полный ответ для критика
            try:
                data = json.loads(chunk[5:].strip())  # срезаем "data: "
                if "token" in data:
                    full_answer += data["token"]
            except Exception:
                pass

        if retry_count >= MAX_RETRIES:
            break

        score, feedback = await _run_critic(query, full_answer)

        yield f"data: {json.dumps({'critic': {'score': score, 'feedback': feedback, 'passed': score >= CRITIC_PASS_SCORE}})}\n\n"

        if score >= CRITIC_PASS_SCORE:
            break

        retry_count += 1
        yield f"data: {json.dumps({'retry': True, 'attempt': retry_count})}\n\n"

        current_messages = list(current_messages) + [
            {"role": "assistant", "content": full_answer},
            {"role": "user", "content": (
                f"[Критик оценил твой ответ на {score}/10 и указал]: {feedback}\n\n"
                "Пожалуйста, исправь ответ с учётом этих замечаний."
            )}
        ]

    yield "data: [DONE]\n\n"


async def _run_critic(query: str, answer: str) -> tuple[int, str]:
    """Запускает агента-критика, возвращает (score, feedback)."""
    import re
    clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer).strip()

    prompt = (
        f"Вопрос пользователя:\n{query}\n\n"
        f"Ответ агента:\n{clean_answer}\n\n"
        "Оцени ответ по критериям: точность, полнота, соответствие вопросу.\n"
        "Ответь строго в формате:\n"
        "SCORE: <целое число от 0 до 10>\n"
        "FEEDBACK: <одно предложение о главном недостатке, или 'Ответ качественный'>"
    )
    raw = await generate_once(
        messages=[{"role": "user", "content": prompt}],
        role_instruction="Ты — строгий критик. Оценивай объективно и кратко.",
        thinking=False,
    )
    return _parse_critic_response(raw)


def _parse_critic_response(raw: str) -> tuple[int, str]:
    score    = CRITIC_PASS_SCORE
    feedback = ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = max(0, min(10, int(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
    return score, feedback


async def summarize_messages(messages: list[dict]) -> str:
    """Суммаризирует блок сообщений."""
    text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return await generate_once(
        messages=[{"role": "user", "content": text}],
        role_instruction=(
            "Ты — суммаризатор диалогов. Сожми переписку в краткое резюме "
            "(не более 500 слов), сохраняя ключевые факты. "
            "Верни только резюме без комментариев."
        ),
        thinking=False,
    )


def _extract_last_query(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""
