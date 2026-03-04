"""services/mcp.py — MCP клиент (JSON-RPC 2.0)"""
import httpx


async def list_tools(server_url: str) -> list[dict]:
    """Запрашивает список инструментов у MCP сервера."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(server_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/list", "params": {},
            })
            resp.raise_for_status()
            return resp.json().get("result", {}).get("tools", [])
    except Exception:
        return []


async def call_tool(server_url: str, tool_name: str, arguments: dict) -> str:
    """Вызывает инструмент на MCP сервере."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(server_url, json={
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        resp.raise_for_status()
        result = resp.json().get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else str(result)