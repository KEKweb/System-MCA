"""
services/vision.py — OCR/Vision через qwen3-vl (Node B)

Поддерживаемые типы:
  Изображения:  image/png, image/jpeg, image/webp, image/gif
                → один вызов, base64 в image_url

  PDF:          application/pdf
                → разбивается на страницы через pypdf,
                  каждая страница — отдельный вызов к модели,
                  результаты конкатенируются

  Текстовые:    text/plain, text/markdown, text/csv,
                application/json, и прочие text/*
                → декодируем как UTF-8, чанкуем по символам,
                  каждый чанк скармливается как текстовое сообщение

Лимиты:
  MAX_IMAGE_BYTES   — максимальный размер одного изображения
  MAX_PAGES         — максимум страниц PDF (больше — отклоняем)
  MAX_TEXT_BYTES    — максимальный размер текстового файла
  TEXT_CHUNK_CHARS  — размер текстового чанка для модели
"""

import base64
import asyncio
import logging
from io import BytesIO

import httpx
from fastapi import HTTPException

from config import get_settings

log = logging.getLogger("mca.vision")
cfg = get_settings()

# ---------------------------------------------------------------------------
# Лимиты
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES  = 20 * 1024 * 1024   # 20 МБ — один кадр
MAX_PAGES        = 50                  # страниц PDF максимум
MAX_TEXT_BYTES   = 512 * 1024          # 512 КБ текста
TEXT_CHUNK_CHARS = 8000                # символов на один вызов модели
PAGE_CONCURRENCY = 3                   # сколько страниц PDF обрабатывать параллельно

IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/jpg",
    "image/webp", "image/gif", "image/bmp",
}

TEXT_MIMES = {
    "text/plain", "text/markdown", "text/csv",
    "text/html", "application/json",
    "application/x-yaml", "text/yaml",
}

async def ocr_document(file_bytes: bytes, mime_type: str = "application/pdf") -> str:
    """
    Главная точка входа.
    Определяет тип файла и вызывает нужный обработчик.
    Возвращает извлечённый текст.
    Бросает HTTPException при превышении лимитов.
    """
    mime_type = mime_type.lower().split(";")[0].strip()

    if mime_type in IMAGE_MIMES:
        return await _process_image(file_bytes, mime_type)

    if mime_type == "application/pdf":
        return await _process_pdf(file_bytes)

    if mime_type in TEXT_MIMES or mime_type.startswith("text/"):
        return await _process_text(file_bytes)

    # Неизвестный тип — пробуем как текст
    log.warning("Неизвестный mime-тип '%s', пробуем как текст", mime_type)
    try:
        return await _process_text(file_bytes)
    except Exception:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип файла: {mime_type}",
        )

async def _process_image(file_bytes: bytes, mime_type: str) -> str:
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Изображение слишком большое: {len(file_bytes) // 1024} КБ. "
                f"Максимум: {MAX_IMAGE_BYTES // 1024 // 1024} МБ."
            ),
        )

    log.info("OCR изображения: %s, %d bytes", mime_type, len(file_bytes))
    return await _vlm_image_call(file_bytes, mime_type)

async def _process_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pypdf не установлен. Добавь pypdf в requirements.txt.",
        )

    reader = pypdf.PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)

    if total_pages > MAX_PAGES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"PDF слишком большой: {total_pages} страниц. "
                f"Максимум: {MAX_PAGES} страниц."
            ),
        )

    log.info("OCR PDF: %d страниц", total_pages)

    try:
        pages_texts = await _pdf_via_render(file_bytes, total_pages)
    except Exception as e:
        log.warning("Рендер PDF не удался (%s), извлекаем текст напрямую", e)
        pages_texts = _pdf_extract_text(reader)

    result = "\n\n".join(
        f"[Страница {i+1}]:\n{t}"
        for i, t in enumerate(pages_texts)
        if t.strip()
    )
    return result


async def _pdf_via_render(file_bytes: bytes, total_pages: int) -> list[str]:
    """
    Рендерит каждую страницу PDF в PNG (через pdf2image + poppler),
    отправляет в qwen3-vl. Страницы обрабатываются батчами PAGE_CONCURRENCY.
    """
    try:
        from pdf2image import convert_from_bytes  # type: ignore
    except ImportError:
        raise RuntimeError("pdf2image не установлен")

    loop = asyncio.get_event_loop()
    images = await loop.run_in_executor(
        None,
        lambda: convert_from_bytes(file_bytes, dpi=150, fmt="PNG"),
    )

    semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    results   = [None] * len(images)

    async def process_page(idx: int, img):
        async with semaphore:
            buf = BytesIO()
            img.save(buf, format="PNG")
            page_bytes = buf.getvalue()

            if len(page_bytes) > MAX_IMAGE_BYTES:
                log.warning(
                    "Страница %d слишком большая (%d bytes), пропускаем",
                    idx + 1, len(page_bytes),
                )
                results[idx] = f"[Страница {idx+1}: слишком большая для обработки]"
                return

            log.debug("OCR страница %d/%d", idx + 1, len(images))
            results[idx] = await _vlm_image_call(page_bytes, "image/png")

    await asyncio.gather(*[process_page(i, img) for i, img in enumerate(images)])
    return results


def _pdf_extract_text(reader) -> list[str]:
    """Фолбэк: извлечение текста из PDF без рендеринга."""
    return [page.extract_text() or "" for page in reader.pages]

async def _process_text(file_bytes: bytes) -> str:
    if len(file_bytes) > MAX_TEXT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Текстовый файл слишком большой: {len(file_bytes) // 1024} КБ. "
                f"Максимум: {MAX_TEXT_BYTES // 1024} КБ."
            ),
        )

    text = file_bytes.decode("utf-8", errors="replace")
    log.info("Обработка текстового файла: %d символов", len(text))

    if len(text) <= TEXT_CHUNK_CHARS:
        return await _vlm_text_call(text)

    chunks = _split_text(text, TEXT_CHUNK_CHARS)
    log.info("Текст разбит на %d чанков", len(chunks))

    results = []
    for i, chunk in enumerate(chunks):
        log.debug("Обработка текстового чанка %d/%d", i + 1, len(chunks))
        result = await _vlm_text_call(chunk, chunk_index=i, total_chunks=len(chunks))
        results.append(result)

    return "\n\n".join(results)

async def _vlm_image_call(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": "qwen3-vl-8b",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Извлеки весь текст из этого изображения/документа. "
                        "Сохрани оригинальную структуру: заголовки, списки, таблицы. "
                        "Верни только извлечённый текст без комментариев."
                    ),
                },
            ],
        }],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.vision_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _vlm_text_call(
    text: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    context_note = f" (часть {chunk_index + 1} из {total_chunks})" if total_chunks > 1 else ""

    payload = {
        "model": "qwen3-vl-8b",
        "messages": [{
            "role": "user",
            "content": (
                f"Обработай следующий текстовый фрагмент документа{context_note}. "
                "Верни его очищенную и структурированную версию, "
                "сохранив всё содержимое:\n\n"
                + text
            ),
        }],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.vision_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

def _split_text(text: str, chunk_size: int) -> list[str]:
    """Делит текст на чанки по chunk_size символов, стараясь резать по абзацам."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        cut = text.rfind("\n\n", start, end)
        if cut <= start:
            cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end

        chunks.append(text[start:cut])
        start = cut

    return [c.strip() for c in chunks if c.strip()]
