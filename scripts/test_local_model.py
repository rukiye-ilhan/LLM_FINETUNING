from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "base_llm"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.float16,
    device_map="auto"
)

print("Model localden açıldı.")
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Path:", MODEL_PATH)