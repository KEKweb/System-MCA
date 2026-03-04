"""
RAG сервис.
  - Загрузка документа: chunk → embed → encrypt → Qdrant
  - Поиск: embed query → Qdrant search → decrypt → вернуть plaintext
"""
import uuid
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models
from ..config import get_settings
from ..crypto import encrypt, decrypt

cfg = get_settings()

EMBED_DIM = 1024


class RAGService:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(url=cfg.database_url)
    async def store_document(
        self,
        text: str,
        chat_uuid: str,
        client_uuid: str,
        filename: str,
        encryption_key: str,
    ) -> str:
        """
        Чанкует текст, эмбеддит открытый текст, шифрует payload,
        сохраняет в Qdrant. Возвращает collection_name.
        """
        collection = f"{client_uuid}_{chat_uuid}"
        await self._ensure_collection(collection)

        chunks = _split_chunks(text, cfg.rag_chunk_size, cfg.rag_chunk_overlap)

        points = []
        for i, chunk in enumerate(chunks):
            vec = await self._embed(chunk)

            enc = encrypt(chunk, encryption_key) if cfg.client_encryption else None

            payload = {
                "chunk_index":   i,
                "filename":      filename,
                "token_count":   len(chunk) // 4,
            }
            if cfg.client_encryption and enc:
                payload["encrypted_content"] = enc["ciphertext"]
                payload["iv"]                = enc["iv"]
                payload["tag"]               = enc["tag"]
            else:
                payload["content"] = chunk

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            ))

        await self.qdrant.upsert(collection_name=collection, points=points)
        return collection

    async def search(
        self,
        query: str,
        chat_uuid: str,
        client_uuid: str,
        encryption_key: str | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        """
        Ищет релевантные чанки, расшифровывает, возвращает список строк.
        """
        collection = f"{client_uuid}_{chat_uuid}"
        k = top_k or cfg.rag_top_k_chunks

        try:
            collections = await self.qdrant.get_collections()
            names = [c.name for c in collections.collections]
            if collection not in names:
                return []
        except Exception:
            return []

        vec     = await self._embed(query)
        results = await self.qdrant.search(
            collection_name=collection,
            query_vector=vec,
            limit=k,
        )

        chunks = []
        for hit in results:
            p = hit.payload
            if "encrypted_content" in p and encryption_key:
                try:
                    text = decrypt(p["encrypted_content"], p["iv"], p["tag"], encryption_key)
                    chunks.append(text)
                except Exception:
                    pass
            elif "content" in p:
                chunks.append(p["content"])

        return chunks

    async def _ensure_collection(self, name: str):
        collections = await self.qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if name not in names:
            await self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            await self.qdrant.update_collection(
                collection_name=name,
                optimizers_config=models.OptimizersConfigDiff(),
            )

    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{cfg.embedding_url}/v1/embeddings",
                json={"model": "qwen3-embedding-8b", "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def delete_collection(self, chat_uuid: str, client_uuid: str):
        collection = f"{client_uuid}_{chat_uuid}"
        try:
            await self.qdrant.delete_collection(collection)
        except Exception:
            pass


def _split_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Делит текст на чанки по chunk_size символов с перекрытием."""
    step   = chunk_size - overlap
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


_rag_instance: RAGService | None = None


def get_rag() -> RAGService:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService()
    return _rag_instance