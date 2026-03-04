"""
Эндпоинты:
  GET  /health              — статус системы
  POST /api/generate        — генерация с SSE стримингом
  POST /api/documents       — загрузка документа (OCR → chunk → embed → Qdrant)
  DELETE /api/documents/{chat_uuid} — удаление коллекции
  POST /api/summarize       — суммаризация блока сообщений
  GET  /api/chats           — список чатов (опционально)
  POST /api/mcp/tools       — список инструментов MCP сервера
"""
import logging
import httpx

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .config import get_settings
from .crypto import generate_key
from .agents.graph import run_pipeline, summarize_messages
from .services.rag import get_rag
from .services.vision import ocr_document
from .services.mcp import list_tools

cfg = get_settings()

logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("mca")

app = FastAPI(title="MCA Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    messages:     list[dict]
    web_search:   bool       = False
    mcp_servers:  list[str]  = []
    doc_uuid:     str | None = None
    chat_uuid:    str        = ""
    stream:       bool       = True


class SummarizeRequest(BaseModel):
    messages: list[dict]


class McpToolsRequest(BaseModel):
    server_url: str

@app.get("/health")
async def health():
    """
    Проверяет доступность Node A (inference) и Node B (rag/embedding).
    """
    inference_ok = await _ping(f"{cfg.base_url}/health")
    rag_ok       = await _ping(f"{cfg.embedding_url}/health")

    status = "ok" if inference_ok and rag_ok else "degraded"
    return {
        "status":    status,
        "inference": inference_ok,
        "rag":       rag_ok,
    }


async def _ping(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(url)
            return r.status_code < 500
    except Exception:
        return False

@app.post("/api/generate")
async def generate(
    req: GenerateRequest,
    x_client_uuid:    str | None = Header(None, alias="X-Client-UUID"),
    x_encryption_key: str | None = Header(None, alias="X-Encryption-Key"),
):
    """
    Основная точка входа для пользовательских запросов.
    Возвращает SSE stream (text/event-stream).

    Headers (опционально):
      X-Client-UUID     — идентификатор клиента (для RAG изоляции)
      X-Encryption-Key  — base64 ключ AES-256-GCM для расшифровки RAG чанков
    """
    client_uuid = x_client_uuid or "anonymous"

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages не может быть пустым")

    async def event_stream():
        try:
            async for chunk in run_pipeline(
                messages        = req.messages,
                web_search_on   = req.web_search,
                mcp_servers     = req.mcp_servers,
                doc_uuid        = req.doc_uuid,
                chat_uuid       = req.chat_uuid,
                client_uuid     = client_uuid,
                encryption_key  = x_encryption_key,
            ):
                yield chunk
        except Exception as e:
            log.error("generate error: %s", e, exc_info=True)
            import json
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/documents")
async def upload_document(
    file:             UploadFile = File(...),
    chat_uuid:        str        = Form(default=""),
    client_uuid:      str        = Form(default="anonymous"),
    x_encryption_key: str | None = Header(None, alias="X-Encryption-Key"),
):
    """
    Загружает документ:
      1. Если PDF/изображение — OCR через qwen3-vl (Node B)
      2. Делит на чанки (1024 символа, 128 перекрытие)
      3. Если документ > DOCUMENT_MAX_TOKENS символов — суммаризация
      4. Эмбеддит (qwen3-embedding-8b, Node B)
      5. Шифрует AES-256-GCM
      6. Сохраняет в Qdrant с TTL 24ч

    Возвращает:
      doc_uuid       — идентификатор коллекции
      encryption_key — base64 ключ (если CLIENT_ENCRYPTION=true)
    """
    raw_bytes = await file.read()
    filename  = file.filename or "document"
    mime      = file.content_type or "application/octet-stream"

    log.info("Загрузка документа: %s (%s, %d bytes)", filename, mime, len(raw_bytes))

    if mime in ("application/pdf", "image/png", "image/jpeg", "image/webp"):
        try:
            text = await ocr_document(raw_bytes, mime)
        except Exception as e:
            log.warning("OCR недоступен, пробуем как текст: %s", e)
            text = raw_bytes.decode("utf-8", errors="replace")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Не удалось извлечь текст из документа")

    max_chars = cfg.document_max_tokens * 4
    if len(text) > max_chars:
        log.info("Документ %d симв > лимит %d, суммаризируем", len(text), max_chars)
        try:
            summary = await summarize_messages([
                {"role": "user", "content": f"Суммаризируй следующий документ:\n\n{text[:max_chars * 3]}"}
            ])
            text = summary
        except Exception as e:
            log.warning("Суммаризация документа не удалась: %s", e)
            text = text[:max_chars]

    enc_key = x_encryption_key
    if cfg.client_encryption and not enc_key:
        enc_key = generate_key()

    doc_uuid = chat_uuid or f"doc_{hash(filename) & 0xFFFFFF}"
    rag      = get_rag()

    await rag.store_document(
        text           = text,
        chat_uuid      = doc_uuid,
        client_uuid    = client_uuid,
        filename       = filename,
        encryption_key = enc_key or "",
    )

    log.info("Документ сохранён: collection=%s_%s", client_uuid, doc_uuid)

    response = {"doc_uuid": doc_uuid, "filename": filename, "chars": len(text)}
    if cfg.client_encryption and enc_key:
        response["encryption_key"] = enc_key

    return JSONResponse(response)


@app.delete("/api/documents/{chat_uuid}")
async def delete_document(
    chat_uuid:   str,
    client_uuid: str = "anonymous",
):
    rag = get_rag()
    await rag.delete_collection(chat_uuid, client_uuid)
    return {"status": "deleted"}

@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages пустой")

    try:
        summary = await summarize_messages(req.messages)
        return {"summary": summary}
    except Exception as e:
        log.error("summarize error: %s", e)
        raise HTTPException(status_code=503, detail=f"Суммаризация недоступна: {e}")

@app.post("/api/mcp/tools")
async def mcp_tools(req: McpToolsRequest):
    """Возвращает список инструментов MCP сервера."""
    tools = await list_tools(req.server_url)
    return {"tools": tools}

@app.on_event("startup")
async def startup():
    log.info("MCA Backend запущен")
    log.info("Node A (inference): %s", cfg.base_url)
    log.info("Node B (embedding): %s", cfg.embedding_url)
    log.info("Node B (vision):    %s", cfg.vision_url)
    log.info("Node B (router):    %s", cfg.router_url)
    log.info("Qdrant:             %s", cfg.database_url)
    log.info("SearXNG:            %s", cfg.searxng_url)

    # Инициализируем semantic router (создаёт коллекцию routes если нет)
    try:
        from .services.router import get_router
        await get_router()
        log.info("Semantic router инициализирован")
    except Exception as e:
        log.warning("Semantic router недоступен при старте: %s", e)