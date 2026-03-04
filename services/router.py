"""
Семантический роутер.
Векторизует запрос через qwen3-embedding-0.6b (Node B),
ищет ближайший пример в Qdrant коллекции 'routes',
возвращает тему (topic) и инструкцию роли агента.
"""
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ..config import get_settings

cfg = get_settings()

ROUTES_COLLECTION = "routes"
EMBED_DIM         = 1024  # qwen3-embedding-0.6b output dim

DEFAULT_ROUTES = [
    {
        "topic": "code",
        "role":  "Ты — эксперт-программист. Помогай с кодом, отладкой, архитектурой.",
        "examples": [
            "напиши функцию на python",
            "как исправить ошибку в коде",
            "объясни этот алгоритм",
            "напиши скрипт для",
        ],
    },
    {
        "topic": "research",
        "role":  "Ты — аналитик-исследователь. Ищи информацию, анализируй, делай выводы.",
        "examples": [
            "расскажи о",
            "что такое",
            "найди информацию про",
            "объясни как работает",
        ],
    },
    {
        "topic": "document",
        "role":  "Ты — эксперт по работе с документами. Анализируй, суммаризируй, отвечай на вопросы по документу.",
        "examples": [
            "что написано в документе",
            "суммаризируй файл",
            "найди в документе",
            "о чём этот текст",
        ],
    },
    {
        "topic": "general",
        "role":  "Ты — универсальный ИИ-ассистент. Отвечай развёрнуто и точно.",
        "examples": [
            "помоги мне",
            "скажи",
            "как",
            "почему",
        ],
    },
]


class SemanticRouter:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(url=cfg.database_url)

    async def ensure_routes_collection(self):
        """Создаёт коллекцию routes и заполняет эталонами если не существует."""
        collections = await self.qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if ROUTES_COLLECTION in names:
            return

        await self.qdrant.create_collection(
            collection_name=ROUTES_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

        points = []
        idx = 0
        for route in DEFAULT_ROUTES:
            for example in route["examples"]:
                vec = await self._embed(example)
                points.append(PointStruct(
                    id=idx,
                    vector=vec,
                    payload={"topic": route["topic"], "role": route["role"]},
                ))
                idx += 1

        await self.qdrant.upsert(collection_name=ROUTES_COLLECTION, points=points)

    async def route(self, query: str) -> dict:
        """
        Возвращает {"topic": str, "role": str}.
        Берёт первые+последние 256 токенов запроса для векторизации.
        """
        truncated = _head_tail_truncate(query, max_chars=1024)
        vec = await self._embed(truncated)

        results = await self.qdrant.search(
            collection_name=ROUTES_COLLECTION,
            query_vector=vec,
            limit=1,
        )

        if not results:
            return {"topic": "general", "role": DEFAULT_ROUTES[-1]["role"]}

        payload = results[0].payload
        return {"topic": payload["topic"], "role": payload["role"]}

    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{cfg.router_url}/v1/embeddings",
                json={"model": "qwen3-embedding-0.6b", "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]


def _head_tail_truncate(text: str, max_chars: int = 1024) -> str:
    """Берёт первые + последние max_chars/2 символов запроса."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " ... " + text[-half:]


_router_instance: SemanticRouter | None = None


async def get_router() -> SemanticRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = SemanticRouter()
        await _router_instance.ensure_routes_collection()
    return _router_instance