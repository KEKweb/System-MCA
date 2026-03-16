"""
Клиент к llama-server (Node A).
Поддерживает стриминг через OpenAI-compatible /v1/chat/completions.
"""

import json
from typing import AsyncIterator

import httpx

from config import get_settings

cfg = get_settings()

STATIC_SYSTEM_PROMPT = (
    "Ты — полезный ИИ-ассистент в составе системы MCA. "
    "Отвечай точно, лаконично и по делу. "
    "Если используешь цепочку рассуждений — оборачивай её в <think>...</think>."
)


async def stream_generate(
    messages: list[dict],
    role_instruction: str = "",
    thinking: bool | None = None,
) -> AsyncIterator[str]:
    """
    Генерирует ответ через llama-server, стримит токены.

    Late-Binding Role Injection:
    роль агента и thinking-переключатель инжектируются в конец
    последнего user-сообщения, чтобы не ломать статичный system prompt.
    """
    prepared = _prepare_messages(messages, role_instruction, thinking)

    payload = {
        "model": "qwen3-8b",
        "messages": prepared,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{cfg.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue

                raw = line[5:].strip()

                if raw == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return

                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}) or {}
                    token = delta.get("content", "")

                    if token:
                        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                except json.JSONDecodeError:
                    continue


async def generate_once(
    messages: list[dict],
    role_instruction: str = "",
    thinking: bool | None = None,
) -> str:
    """
    Не-стриминговая генерация — для критика и суммаризации.
    """
    prepared = _prepare_messages(messages, role_instruction, thinking)

    payload = {
        "model": "qwen3-8b",
        "messages": prepared,
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"].get("content") or ""


def _prepare_messages(
    messages: list[dict],
    role_instruction: str,
    thinking: bool | None = None,
) -> list[dict]:
    """
    Собирает финальный список сообщений для llama-server:
    1. Статичный system prompt
    2. История диалога
    3. В хвост последнего user-сообщения добавляются:
       - инструкция роли агента
       - /think или /no_think при необходимости
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
