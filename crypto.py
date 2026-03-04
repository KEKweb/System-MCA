"""
Схема:
  - Ключ генерируется на сервере при первой загрузке документа
    и возвращается клиенту, который хранит его в localStorage.
  - Сервер ключ не сохраняет — при каждом запросе с RAG клиент
    передаёт ключ обратно в заголовке X-Encryption-Key.
  - В Qdrant хранятся только зашифрованные чанки + вектор открытого текста.
"""
import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_key() -> str:
    """Генерирует 256-bit ключ, возвращает base64-строку."""
    return base64.b64encode(os.urandom(32)).decode()


def encrypt(plaintext: str, key_b64: str) -> dict:
    """
    Шифрует строку.
    Возвращает dict с полями: ciphertext, iv, tag (все base64).
    """
    key = base64.b64decode(key_b64)
    iv  = os.urandom(12)
    aesgcm = AESGCM(key)
    ct_with_tag = aesgcm.encrypt(iv, plaintext.encode(), None)
    ct  = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]
    return {
        "ciphertext": base64.b64encode(ct).decode(),
        "iv":         base64.b64encode(iv).decode(),
        "tag":        base64.b64encode(tag).decode(),
    }


def decrypt(ciphertext_b64: str, iv_b64: str, tag_b64: str, key_b64: str) -> str:
    """Расшифровывает чанк. Бросает InvalidTag при подделке."""
    key        = base64.b64decode(key_b64)
    iv         = base64.b64decode(iv_b64)
    ct         = base64.b64decode(ciphertext_b64)
    tag        = base64.b64decode(tag_b64)
    aesgcm     = AESGCM(key)
    plaintext  = aesgcm.decrypt(iv, ct + tag, None)
    return plaintext.decode()