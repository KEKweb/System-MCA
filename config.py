from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    base_url: str = "http://localhost:11434"

    embedding_url: str  = "http://localhost:11435"
    vision_url: str     = "http://localhost:11436"
    router_url: str     = "http://localhost:11437"

    database_url: str   = "http://localhost:6333"
    searxng_url: str    = "http://localhost:8080"

    context_length: int         = 30720
    kv_cache_quantization: str  = "q8_0"
    parallel_slots: int         = 9

    summarization_threshold: int = 20480
    document_max_tokens: int     = 10240
    rag_ttl_seconds: int         = 86400

    router_top_k: int            = 1
    rag_chunk_size: int          = 1024
    rag_chunk_overlap: int       = 128
    rag_top_k_chunks: int        = 5

    client_encryption: bool      = True

    log_level: str   = "INFO"
    sod_docker: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()