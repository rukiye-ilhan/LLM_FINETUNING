from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = BASE_DIR / "models" / "base_llm"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


ADAPTER_PATH = resolve_repo_path(
    os.getenv("LORA_ADAPTER_PATH", str(BASE_DIR / "outputs" / "lora_adapter"))
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))


TEST_USER_QUERY = (
    "I feel overwhelmed at work and I keep thinking that I'm not good enough. "
    "I want to handle this in a healthier way."
)

TEST_CONTEXT = """
Title: How can I cope with stress and self-doubt at work?
Topic: workplace-relationships
Question: I often feel anxious and not good enough in my job. What can I do?
Answer: It can help to break the problem into smaller steps, notice negative self-talk,
and focus on practical routines that support calm and confidence. Small, realistic actions
often work better than trying to solve everything at once.
""".strip()

TEST_EMOTION = "anxiety"
TEST_TONE = "calm, reassuring, grounded"


def build_test_prompt(
    user_query: str,
    context_text: str,
    predicted_emotion: str,
    tone: str,
) -> str:
    return (
        "Use only the context below to answer the user.\n"
        "Be empathetic, grounded, concise, and practical.\n"
        "Do not continue with another example.\n"
        "Do not write titles, labels, or extra sections.\n\n"
        "Answer as a direct supportive assistant, not as a forum commenter.\n"
        "Do not say you are not a professional, do not say you have heard things, and do not use casual openings like 'yeah' or 'look'.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User: {user_query}\n"
        f"Emotion: {predicted_emotion}\n"
        f"Tone: {tone}\n\n"
        "Answer:"
    )


STOP_PATTERNS = [
    r"\nHuman:",
    r"\nUser:",
    r"\nContext:",
    r"\nTitle:",
    r"\nTopic:",
    r"\nQuestion:",
    r"\nAnswer:",
    r"\n###",
    r"Human:",
    r"User:",
    r"Context:",
    r"Title:",
    r"Topic:",
    r"Question:",
    r"Answer:",
    r"###",
    r"\[Document\]",
    r"\n\[Document\]",
]


def clean_model_output(decoded_text: str) -> str:
    text = decoded_text.strip()

    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()

    cut_positions = []
    for pattern in STOP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            cut_positions.append(match.start())

    if cut_positions:
        text = text[:min(cut_positions)].strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Base model path: %s", BASE_MODEL_PATH)
    logger.info("LoRA adapter path: %s", ADAPTER_PATH)

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {ADAPTER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )
    model.eval()

    prompt = build_test_prompt(
        user_query=TEST_USER_QUERY,
        context_text=TEST_CONTEXT,
        predicted_emotion=TEST_EMOTION,
        tone=TEST_TONE,
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = clean_model_output(decoded)

    print("\n" + "=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)
    print(decoded)

    print("\n" + "=" * 80)
    print("FINAL ANSWER ONLY")
    print("=" * 80)
    print(final_answer)


if __name__ == "__main__":
    main()
