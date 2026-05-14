"""
services/llm.py — Клиент к llama-server (Node A).

Поддерживает:
  - Стриминг через OpenAI-compatible /v1/chat/completions
  - Native tool calling (function calling)
  - Non-streaming генерация для критика/суммаризации
"""

import json
import re
from typing import AsyncIterator, Any

import httpx

from config import get_settings

cfg = get_settings()

STATIC_SYSTEM_PROMPT = (
    "Ты — полезный ИИ-ассистент в составе системы MCA. "
    "Отвечай точно, лаконично и по делу. "
    "Если используешь цепочку рассуждений — оборачивай её в <think>...</think>. "
    "Если тебе нужен инструмент для выполнения задачи — вызывай его через tool_calls."
)

# =============================================================================
# Tool schemas
# =============================================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for current information. Use this when you need up-to-date information or don't know the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up",
                }
            },
            "required": ["query"],
        },
    },
}

RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Search uploaded documents for relevant information. Use this when the user asks about documents they've uploaded or when you need information from the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in documents",
                }
            },
            "required": ["query"],
        },
    },
}

MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "mcp_call_tool",
        "description": "Call a tool from a connected MCP (Model Context Protocol) server. Use this when a user explicitly asks to use an MCP tool or when an MCP tool would help complete the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "The URL of the MCP server",
                },
                "tool_name": {
                    "type": "string",
                    "description": "The name of the tool to call",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool",
                    "additionalProperties": True,
                },
            },
            "required": ["server_url", "tool_name", "arguments"],
        },
    },
}


# =============================================================================
# Tool call parsing
# =============================================================================

def _parse_tool_calls(content: str | None) -> list[dict]:
    """
    Extracts tool_calls from model response content.

    Models may emit tool calls as JSON blocks in various formats:
      - Inside <think> tags
      - As standalone JSON
      - Mixed with text

    Returns list of {"name": str, "arguments": dict, "call_id": str}
    """
    if not content:
        return []

    results = []

    # Pattern: JSON object with "name" and "arguments" keys
    # Look for tool call patterns in the content
    patterns = [
        # Standard format: {"name": "...", "arguments": {...}}
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
        # With call_id
        r'\{\s*"call_id"\s*:\s*"([^"]+)"\s*,\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content):
            if len(match.groups()) == 3:
                call_id, name, args_str = match.groups()
            else:
                call_id = f"call_{len(results)}"
                name, args_str = match.groups()

            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                continue

            results.append({
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            })

    if results:
        return results

    # Try parsing as array of tool calls
    array_pattern = r'\[\s*(\{[^}]+\})\s*\]'
    for match in re.finditer(array_pattern, content):
        try:
            tools_array = json.loads(match.group(1))
            if isinstance(tools_array, list):
                for item in tools_array:
                    if isinstance(item, dict) and "name" in item and "arguments" in item:
                        results.append({
                            "call_id": item.get("call_id", f"call_{len(results)}"),
                            "name": item["name"],
                            "arguments": item["arguments"],
                        })
        except json.JSONDecodeError:
            continue

    return results


def _build_tool_calls_message(tool_results: list[dict]) -> dict:
    """
    Builds a message containing tool results to send back to the model.

    Each tool result is formatted as a text block so the model can read
    the output and continue its response.
    """
    parts = []
    for tr in tool_results:
        parts.append(
            f"[Инструмент вызван: {tr['name']}]\n"
            f"Результат:\n{tr['content']}\n"
            f"[Конец результата]\n"
        )
    return {
        "role": "user",
        "content": "\n".join(parts),
    }


# =============================================================================
# Streaming generation with tool calling support
# =============================================================================

async def stream_generate(
    messages: list[dict],
    role_instruction: str = "",
    thinking: bool | None = None,
    tools: list[dict] | None = None,
    max_tool_rounds: int = 5,
) -> AsyncIterator[str]:
    """
    Generates a response via llama-server with streaming tokens.

    Supports native tool calling: if the model requests tools, they are
    executed and results are fed back. This loops until the model
    produces a final text response (no more tool calls).

    Args:
        messages: Conversation messages.
        role_instruction: Role instruction injected into last user message.
        thinking: Whether to enable thinking mode.
        tools: List of tool schemas to offer to the model.
        max_tool_rounds: Maximum number of tool-calling rounds before
            forcing a final response.

    Yields:
        SSE-formatted token events: `data: {"token": "..."}`\n\n
    """
    prepared = _prepare_messages(messages, role_instruction, thinking)

    payload: dict[str, Any] = {
        "model": "qwen3-8b",
        "messages": prepared,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    full_answer = ""
    round_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        while round_count < max_tool_rounds:
            round_count += 1

            async with client.stream(
                "POST",
                f"{cfg.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()

                # Collect full response for tool call parsing,
                # but stream tokens as they arrive
                response_content = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()

                    if raw == "[DONE]":
                        # End of response
                        if not _parse_tool_calls(response_content):
                            # Model gave a final text response — no more tool calls
                            yield "data: [DONE]\n\n"
                            return

                        # Model requested tool calls — execute them
                        tool_calls = _parse_tool_calls(response_content)
                        if not tool_calls:
                            yield "data: [DONE]\n\n"
                            return

                        # Execute tool calls
                        tool_results = await _execute_tool_calls(
                            tool_calls, tools, payload.get("messages", [])
                        )

                        # Build messages for next round
                        tool_msg = _build_tool_calls_message(tool_results)
                        payload["messages"] = payload.get("messages", []) + [
                            {"role": "assistant", "content": response_content},
                            tool_msg,
                        ]

                        # Reset for next generation round
                        full_answer = ""
                        response_content = ""
                        break

                    try:
                        chunk = json.loads(raw)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}) or {}

                        # Check for tool_calls in delta
                        tc_list = delta.get("tool_calls")
                        if tc_list:
                            for tc in tc_list:
                                fn = tc.get("function", {})
                                name = fn.get("name")
                                args = fn.get("arguments", "")
                                if name or args:
                                    # Accumulate tool call text into response_content
                                    if name:
                                        response_content += (
                                            f'\n{{"call_id": "{tc.get("id", "")}", '
                                            f'"name": "{name}", '
                                            f'"arguments": '
                                        )
                                    if args:
                                        response_content += args
                            continue

                        token = delta.get("content", "")
                        if token:
                            response_content += token
                            full_answer += token
                            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                    except json.JSONDecodeError:
                        continue


# =============================================================================
# Non-streaming generation
# =============================================================================

async def generate_once(
    messages: list[dict],
    role_instruction: str = "",
    thinking: bool | None = None,
    tools: list[dict] | None = None,
) -> str:
    """
    Non-streaming generation for critic and summarization.
    """
    prepared = _prepare_messages(messages, role_instruction, thinking)

    payload: dict[str, Any] = {
        "model": "qwen3-8b",
        "messages": prepared,
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()
    message = data["choices"][0].get("message", {})
    return message.get("content") or ""


# =============================================================================
# Tool execution
# =============================================================================

async def _execute_tool_calls(
    tool_calls: list[dict],
    tools: list[dict] | None,
    conversation_messages: list[dict],
) -> list[dict]:
    """
    Executes tool calls requested by the model.

    Returns list of {"name": str, "content": str} for each executed tool.
    """
    from services.search import web_search
    from services.rag import get_rag
    from services.mcp import call_tool as mcp_call_tool

    results = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if name == "web_search":
            query = args.get("query", "")
            if query:
                content = await web_search(query)
                results.append({"name": name, "content": content})

        elif name == "rag_search":
            query = args.get("query", "")
            if query:
                # Extract RAG params from conversation context
                rag_params = _extract_rag_params(conversation_messages)
                rag = get_rag()
                chunks = await rag.search(
                    query=query,
                    chat_uuid=rag_params.get("chat_uuid", ""),
                    client_uuid=rag_params.get("client_uuid", "anonymous"),
                    encryption_key=rag_params.get("encryption_key"),
                )
                if chunks:
                    content = "[Релевантные фрагменты документа]:\n" + "\n---\n".join(chunks)
                else:
                    content = "[RAG] По запросу ничего не найдено в загруженных документах."
                results.append({"name": name, "content": content})

        elif name == "mcp_call_tool":
            server_url = args.get("server_url", "")
            tool_name = args.get("tool_name", "")
            mcp_args = args.get("arguments", {})
            if server_url and tool_name:
                try:
                    content = await mcp_call_tool(server_url, tool_name, mcp_args)
                    results.append({"name": name, "content": content})
                except Exception as e:
                    results.append({
                        "name": name,
                        "content": f"[Ошибка MCP] Не удалось вызвать инструмент '{tool_name}': {e}",
                    })
            else:
                results.append({
                    "name": name,
                    "content": "[Ошибка MCP] Не указаны server_url или tool_name.",
                })

        else:
            results.append({
                "name": name,
                "content": f"[Ошибка] Неизвестный инструмент: {name}",
            })

    return results


def _extract_rag_params(messages: list[dict]) -> dict:
    """
    Extracts RAG parameters from conversation messages.
    Looks for special context markers injected by the pipeline.
    """
    chat_uuid = ""
    client_uuid = "anonymous"
    encryption_key = None

    for m in messages:
        content = m.get("content", "")
        if "[RAG_CHAT_UUID]" in content:
            chat_uuid = content.split("[RAG_CHAT_UUID]")[1].split("]")[0]
        if "[RAG_CLIENT_UUID]" in content:
            client_uuid = content.split("[RAG_CLIENT_UUID]")[1].split("]")[0]

    return {
        "chat_uuid": chat_uuid,
        "client_uuid": client_uuid,
        "encryption_key": encryption_key,
    }


# =============================================================================
# Message preparation
# =============================================================================

def _prepare_messages(
    messages: list[dict],
    role_instruction: str,
    thinking: bool | None = None,
) -> list[dict]:
    """
    Builds the final message list for llama-server:
    1. Static system prompt
    2. Conversation history
    3. Role instruction and thinking toggle appended to the last user message
    """
    result = [{"role": "system", "content": STATIC_SYSTEM_PROMPT}]

    if not messages:
        return result

    history = [dict(m) for m in messages]

    injections: list[str] = []

    if role_instruction:
        injections.append(f"[ROLE]: {role_instruction}")

    if thinking is True:
        injections.append("/think")
    elif thinking is False:
        injections.append("/no_think")

    if injections:
        suffix = "\n\n".join(injections)
        injected = False

        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                history[i] = dict(history[i])
                original = history[i].get("content", "")
                history[i]["content"] = f"{original}\n\n{suffix}" if original else suffix
                injected = True
                break

        if not injected:
            history.append({"role": "user", "content": suffix})

    result.extend(history)
    return result
