"""services/search.py — SearXNG клиент"""
import httpx
from config import get_settings

cfg = get_settings()

async def web_search(query: str, num_results: int = 5) -> str:
    """Ищет в интернете через SearXNG, возвращает форматированный контекст."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{cfg.searxng_url}/search",
                params={"q": query, "format": "json", "categories": "general"},
            )
            resp.raise_for_status()
            data    = resp.json()
            results = data.get("results", [])[:num_results]

            if not results:
                return ""

            lines = ["[Результаты поиска в интернете]:"]
            for r in results:
                title   = r.get("title", "")
                url     = r.get("url", "")
                snippet = r.get("content", "")
                lines.append(f"- {title} ({url}): {snippet}")

            return "\n".join(lines)
    except Exception as e:
        return f"[Поиск недоступен: {e}]"
