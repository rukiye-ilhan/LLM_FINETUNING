from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

BASE_MODEL_PATH = BASE_DIR / "models" / "base_llm"
ADAPTER_PATH = BASE_DIR / "outputs" / "lora_adapter"
COUNSEL_FULL_PATH = BASE_DIR / "data" / "gold" / "counsel_full.parquet"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "counsel_rag"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4
MAX_NEW_TOKENS = 120