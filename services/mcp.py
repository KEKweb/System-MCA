"""services/mcp.py — MCP клиент (JSON-RPC 2.0)"""
import httpx
from typing import Any

# Общий клиент для connection pooling
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _client


async def _close_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def list_tools(server_url: str) -> list[dict]:
    """Запрашивает список инструментов у MCP сервера."""
    client = _get_client()
    try:
        resp = await client.post(server_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/list", "params": {},
        })
        resp.raise_for_status()
        return resp.json().get("result", {}).get("tools", [])
    except httpx.TimeoutException:
        return []
    except httpx.HTTPError:
        return []
    except Exception:
        return []


async def call_tool(server_url: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    """Вызывает инструмент на MCP сервере."""
    client = _get_client()
    resp = await client.post(server_url, json={
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    })
    resp.raise_for_status()
    result = resp.json().get("result", {})
    content = result.get("content", [])
    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    return "\n".join(texts) if texts else str(result)
