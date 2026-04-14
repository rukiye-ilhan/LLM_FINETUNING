from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_DIR = BASE_DIR / "models" / "base_llm"

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
)

print("Model indirildi:", LOCAL_DIR)