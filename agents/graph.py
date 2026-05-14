"""
agents/graph.py — Граф агентов MCA.

Поток:
  router → load_mcp_tools → generate_with_tools → critic
                                          ↑               |
                                          └── retry ───── ┘

Модель сама решает когда вызывать инструменты (web_search, rag_search,
mcp_call_tool). Инструменты передаются как tool schemas в запрос к LLM.
Результаты выполнения инструментов возвращаются модели, и она продолжает
ответ с учётом полученной информации.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from services.llm    import stream_generate, generate_once
from services.llm    import WEB_SEARCH_TOOL, RAG_SEARCH_TOOL, MCP_TOOL
from services.router import get_router
from services.mcp    import list_tools

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


async def router_node(state: AgentState) -> AgentState:
    router = await get_router()
    result = await router.route(state.query)
    state.topic            = result["topic"]
    state.role_instruction = result["role"]
    return state


async def load_mcp_tools(servers: list[str]) -> list[dict]:
    """
    Загружает список инструментов из всех MCP серверов.
    Возвращает список tool schemas в формате OpenAI API.
    """
    all_tools = []
    if not servers:
        return all_tools

    for server_url in servers:
        try:
            tools = await list_tools(server_url)
            for tool in tools:
                # Добавляем URL сервера в параметры каждого инструмента,
                # чтобы модель могла его использовать при вызове
                if "function" in tool and "parameters" in tool.get("function", {}):
                    all_tools.append(tool)
        except Exception:
            continue

    return all_tools


def _build_available_tools(state: AgentState) -> list[dict]:
    """
    Собирает список доступных инструментов для модели.

    Всегда доступны: web_search, rag_search.
    mcp_call_tool доступен только если есть MCP серверы.
    """
    tools = []

    # Web search всегда доступен (модель сама решит когда использовать)
    if state.web_search:
        tools.append(WEB_SEARCH_TOOL)

    # RAG search доступен если есть документ
    if state.doc_uuid and state.chat_uuid:
        tools.append(RAG_SEARCH_TOOL)

    # MCP tools если есть серверы
    if state.mcp_servers:
        tools.append(MCP_TOOL)

    return tools


# =============================================================================
# PUBLIC API — run_pipeline
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
    Полный пайплайн с native tool calling.

    Модель сама решает когда использовать инструменты:
    1. Router классифицирует запрос и задаёт роль
    2. Собираются доступные инструменты (web_search, rag_search, mcp)
    3. Модель генерирует ответ, вызывая инструменты через tool_calls
    4. Результаты инструментов возвращаются модели
    5. Critic оценивает финальный ответ

    SSE-эвенты:
      data: {"token": "..."}           — токен ответа
      data: {"critic": {...}}          — результат оценки критика
      data: {"retry": true}            — начинается новая попытка
      data: [DONE]                     — завершение
    """
    query = _extract_last_query(messages)

    router = await get_router()
    route_result = await router.route(query)
    role_instruction = route_result["role"]

    state = AgentState(
        messages=messages, query=query, web_search=web_search_on,
        mcp_servers=mcp_servers or [], doc_uuid=doc_uuid,
        chat_uuid=chat_uuid, client_uuid=client_uuid,
        encryption_key=encryption_key,
    )

    # Загружаем MCP инструменты если есть серверы
    if state.mcp_servers:
        mcp_tools = await load_mcp_tools(state.mcp_servers)
    else:
        mcp_tools = []

    # Собираем доступные инструменты
    available_tools = _build_available_tools(state)

    # Добавляем MCP инструменты в список
    if mcp_tools:
        available_tools.extend(mcp_tools)

    current_messages = list(messages)
    retry_count = 0

    while True:
        full_answer = ""
        async for chunk in stream_generate(
            messages=current_messages,
            role_instruction=role_instruction,
            thinking=True,
            tools=available_tools if available_tools else None,
        ):
            if chunk == "data: [DONE]\n\n":
                break
            yield chunk

            try:
                data = json.loads(chunk[5:].strip())
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
