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

DEFAULT_ROUTES =[
    {
        "topic": "code",
        "role":  "Ты — Senior Software Engineer. Пиши оптимальный, безопасный и хорошо задокументированный код. Помогай с отладкой, рефакторингом и архитектурой программного обеспечения.",
        "examples":[
            "напиши функцию на python",
            "как исправить ошибку `IndexError` в коде",
            "объясни этот алгоритм сортировки",
            "напиши bash-скрипт для бэкапа базы данных",
            "как оптимизировать SQL запрос",
            "спроектируй архитектуру микросервиса"
        ],
    },
    {
        "topic": "math",
        "role":  "Ты — эксперт-математик. Решай математические уравнения, доказывай теоремы, помогай с высшей математикой, статистикой и логическими задачами. Отвечай строго и пошагово.",
        "examples":[
            "реши дифференциальное уравнение",
            "найди интеграл функции",
            "докажи теорему Пифагора",
            "объясни теорию вероятностей",
            "как перемножить эти две матрицы",
            "посчитай дисперсию для выборки"
        ],
    },
    {
        "topic": "science",
        "role":  "Ты — ученый-естествоиспытатель. Объясняй физические, химические и биологические процессы, законы природы и сложные научные концепции простым, но академически верным языком.",
        "examples":[
            "как работает квантовая запутанность",
            "объясни общую теорию относительности",
            "какая химическая формула у серной кислоты",
            "как происходит процесс фотосинтеза",
            "расскажи про строение атома",
            "что такое черная дыра"
        ],
    },
    {
        "topic": "humanities",
        "role":  "Ты — эксперт в гуманитарных науках. Анализируй исторические события, литературные произведения, философские концепции, вопросы социологии и лингвистики.",
        "examples":[
            "каковы предпосылки Первой мировой войны",
            "сделай анализ романа 'Преступление и наказание'",
            "в чем суть категорического императива Канта",
            "расскажи историю падения Римской империи",
            "в чем разница между модернизмом и постмодернизмом",
            "проанализируй это стихотворение"
        ],
    },
    {
        "topic": "psychology",
        "role":  "Ты — эксперт-психолог и нейробиолог. Анализируй когнитивные искажения, поведенческие паттерны, психологические теории и объясняй механизмы работы человеческого мозга.",
        "examples":[
            "что такое когнитивный диссонанс",
            "каковы симптомы профессионального выгорания",
            "объясни пирамиду потребностей Маслоу",
            "какие существуют механизмы психологической защиты",
            "как работает кратковременная память",
            "что такое эффект Даннинга-Крюгера"
        ],
    },
    {
        "topic": "document",
        "role":  "Ты — эксперт по анализу данных. Внимательно читай предоставленный контекст, извлекай факты, делай суммаризацию и отвечай на вопросы строго на основе текста документа.",
        "examples":[
            "что написано в этом PDF документе",
            "сделай краткую выжимку из прикрепленного файла",
            "найди в тексте условия расторжения договора",
            "о чём этот отчет",
            "выдели главные тезисы из статьи",
            "какая итоговая сумма указана в таблице"
        ],
    },
    {
        "topic": "general",
        "role":  "Ты — универсальный ИИ-ассистент (МСА). Отвечай вежливо, развёрнуто и точно. Если не знаешь ответа — честно признайся.",
        "examples":[
            "помоги мне составить план на день",
            "расскажи интересную историю",
            "как приготовить пасту карбонара",
            "посоветуй хороший фильм на вечер",
            "давай просто поболтаем",
            "что ты умеешь делать"
        ],
    }
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
