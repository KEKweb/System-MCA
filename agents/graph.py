"""
agents/graph.py — LangGraph граф агентов.

Поток:
  router_node → context_node → generate_node

  router_node:  семантический роутинг, определяет тему и роль агента
  context_node: сбор контекста (RAG + web search + MCP)
  generate_node: стриминговая генерация через llama-server

AgentState хранит всё состояние между нодами.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncIterator

from langgraph.graph import StateGraph, END

from ..services.llm    import stream_generate, generate_once
from ..services.router import get_router
from ..services.rag    import get_rag
from ..services.search import web_search


@dataclass
class AgentState:
    messages:        list[dict]       = field(default_factory=list)
    query:           str              = ""
    web_search:      bool             = False
    mcp_servers:     list[str]        = field(default_factory=list)
    doc_uuid:        str | None       = None
    chat_uuid:       str              = ""
    client_uuid:     str              = "anonymous"
    encryption_key:  str | None       = None

    topic:           str              = "general"
    role_instruction: str             = ""

    rag_context:     str              = ""
    web_context:     str              = ""

    enriched_messages: list[dict]     = field(default_factory=list)

async def router_node(state: AgentState) -> AgentState:
    """Определяет тему и роль агента через семантический роутер."""
    router = await get_router()
    result = await router.route(state.query)
    state.topic           = result["topic"]
    state.role_instruction = result["role"]
    return state


async def context_node(state: AgentState) -> AgentState:
    """Собирает RAG и web-контекст параллельно."""
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
        state.messages,
        state.rag_context,
        state.web_context,
    )

    return state


async def _fetch_rag(state: AgentState):
    rag    = get_rag()
    chunks = await rag.search(
        query          = state.query,
        chat_uuid      = state.chat_uuid,
        client_uuid    = state.client_uuid,
        encryption_key = state.encryption_key,
    )
    if chunks:
        state.rag_context = "[Релевантные фрагменты документа]:\n" + "\n---\n".join(chunks)


async def _fetch_web(state: AgentState):
    state.web_context = await web_search(state.query)


def _inject_context(
    messages: list[dict],
    rag_ctx: str,
    web_ctx: str,
) -> list[dict]:
    """Добавляет контекст в хвост последнего user-сообщения."""
    if not messages:
        return messages

    extra = ""
    if rag_ctx:
        extra += f"\n\n{rag_ctx}"
    if web_ctx:
        extra += f"\n\n{web_ctx}"

    if not extra:
        return messages

    result = [dict(m) for m in messages]
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "user":
            result[i]["content"] += extra
            break

    return result

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("router",   router_node)
    graph.add_node("context",  context_node)
    graph.set_entry_point("router")
    graph.add_edge("router",  "context")
    graph.add_edge("context", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

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
    Запускает полный пайплайн агента.
    Yields SSE-строки с токенами.
    """
    query = _extract_last_query(messages)

    state = AgentState(
        messages        = messages,
        query           = query,
        web_search      = web_search_on,
        mcp_servers     = mcp_servers or [],
        doc_uuid        = doc_uuid,
        chat_uuid       = chat_uuid,
        client_uuid     = client_uuid,
        encryption_key  = encryption_key,
    )

    graph   = get_graph()
    state   = await graph.ainvoke(state)

    async for chunk in stream_generate(
        messages        = state.enriched_messages,
        role_instruction = state.role_instruction,
    ):
        yield chunk


async def summarize_messages(messages: list[dict]) -> str:
    """Суммаризирует блок сообщений в краткое резюме."""
    text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    result = await generate_once(
        messages=[{"role": "user", "content": text}],
        role_instruction=(
            "Ты — суммаризатор диалогов. Сожми следующую переписку в краткое "
            "резюме (не более 500 слов), сохраняя все ключевые факты и решения. "
            "Верни только резюме без комментариев."
        ),
    )
    return result


def _extract_last_query(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""