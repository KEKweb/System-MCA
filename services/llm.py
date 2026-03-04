"""
Клиент к llama-server (Node A).
Поддерживает стриминг через OpenAI-compatible /v1/chat/completions.
"""
import json
import httpx
from typing import AsyncIterator
from ..config import get_settings

cfg = get_settings()

STATIC_SYSTEM_PROMPT = (
    "Ты — полезный ИИ-ассистент в составе системы MCA. "
    "Отвечай точно, лаконично и по делу. "
    "Если используешь цепочку рассуждений — оборачивай её в <think>...</think>."
)


async def stream_generate(
    messages: list[dict],
    role_instruction: str = "",
) -> AsyncIterator[str]:
    """
    Генерирует ответ через llama-server, стримит токены.

    Late-Binding Role Injection:
      Роль агента инжектируется в КОНЕЦ последнего user-сообщения,
      а не в system prompt — чтобы не инвалидировать KV-кэш префикса.

    Yields:
        Строки вида "data: {...}\n\n" совместимые с SSE.
    """
    prepared = _prepare_messages(messages, role_instruction)

    payload = {
        "model":       "qwen3-8b",
        "messages":    prepared,
        "stream":      True,
        "temperature": 0.7,
        "max_tokens":  4096,
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
                    token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                except json.JSONDecodeError:
                    continue


async def generate_once(messages: list[dict], role_instruction: str = "") -> str:
    """Не-стриминговая генерация — для суммаризации."""
    prepared = _prepare_messages(messages, role_instruction)
    payload  = {
        "model":       "qwen3-8b",
        "messages":    prepared,
        "stream":      False,
        "temperature": 0.3,
        "max_tokens":  2048,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _prepare_messages(messages: list[dict], role_instruction: str) -> list[dict]:
    """
    Собирает финальный список сообщений для llama-server:
      1. Статичный system prompt (первым — кэшируется)
      2. История диалога
      3. Роль агента инжектируется в хвост последнего user-сообщения
    """
    result = [{"role": "system", "content": STATIC_SYSTEM_PROMPT}]

    if not messages:
        return result

    history = [dict(m) for m in messages]

    if role_instruction:
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                history[i] = dict(history[i])
                history[i]["content"] += f"\n\n[ROLE]: {role_instruction}"
                break

    result.extend(history)
    return result