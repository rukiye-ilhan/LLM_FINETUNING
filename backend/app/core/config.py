import os
from pathlib import Path


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

BASE_DIR = Path(__file__).resolve().parents[3]

def _get_path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value.strip()) if value and value.strip() else default

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


BASE_MODEL_PATH = _get_path_env("BASE_MODEL_PATH", BASE_DIR / "models" / "base_llm")
ADAPTER_PATH = _get_path_env("LORA_ADAPTER_PATH", BASE_DIR / "outputs" / "lora_adapter")
COUNSEL_FULL_PATH = _get_path_env("COUNSEL_FULL_PATH", BASE_DIR / "data" / "gold" / "counsel_full.parquet")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = _get_int_env("QDRANT_PORT", 6333)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "counsel_rag")

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
HF_LOCAL_FILES_ONLY = _get_bool_env("HF_LOCAL_FILES_ONLY", True)

TOP_K = _get_int_env("TOP_K", 4)
MAX_NEW_TOKENS = _get_int_env("MAX_NEW_TOKENS", 120)

CHAT_HISTORY_MESSAGE_LIMIT = _get_int_env("CHAT_HISTORY_MESSAGE_LIMIT", 24)
PROMPT_HISTORY_MESSAGE_LIMIT = _get_int_env("PROMPT_HISTORY_MESSAGE_LIMIT", 10)

RAG_RETRIEVAL_POLICIES = {
    "always",
    "first_turn_only",
    "first_turn_or_topic_shift",
    "never",
}
RAG_RETRIEVAL_POLICY = os.getenv(
    "RAG_RETRIEVAL_POLICY",
    "first_turn_only",
).strip().lower()

if RAG_RETRIEVAL_POLICY not in RAG_RETRIEVAL_POLICIES:
    RAG_RETRIEVAL_POLICY = "first_turn_only"

RAG_ALLOW_EXPLICIT_TOPIC_SHIFT = _get_bool_env(
    "RAG_ALLOW_EXPLICIT_TOPIC_SHIFT",
    True,
)
