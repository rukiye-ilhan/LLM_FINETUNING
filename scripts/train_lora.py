from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "base_llm"
TRAIN_PATH = BASE_DIR / "data" / "llm" / "sft_train.jsonl"
VAL_PATH = BASE_DIR / "data" / "llm" / "sft_val.jsonl"

OUTPUT_DIR = BASE_DIR / "outputs" / "lora_adapter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================
SEED = 42
MAX_LENGTH = 1024

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOGGING_STEPS = 20
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Qwen tipi modeller için güvenli hedef modüller
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16


# =========================
# HELPERS
# =========================
def validate_paths() -> None:
    required_paths = [MODEL_PATH, TRAIN_PATH, VAL_PATH]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Gerekli dosya/klasör bulunamadı: {path}")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_record(record: Dict, idx: int) -> None:
    required_keys = {"instruction", "input", "output"}
    missing = required_keys - set(record.keys())
    if missing:
        raise ValueError(f"Kayıt {idx} içinde eksik alan var: {missing}")


def validate_dataset(records: List[Dict], name: str) -> None:
    if not records:
        raise ValueError(f"{name} dataset boş.")

    for i, record in enumerate(records[:10]):
        validate_record(record, i)


def format_example(example: Dict) -> Dict[str, str]:
    """
    Instruction tuning format
    """
    text = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )
    return {"text": text}


def tokenize_function(examples: Dict[str, List[str]], tokenizer) -> Dict[str, List[List[int]]]:
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def print_dataset_preview(train_records: List[Dict]) -> None:
    sample = train_records[0]
    logger.info("Örnek kayıt preview:")
    logger.info("instruction: %s", sample["instruction"][:120])
    logger.info("input: %s", sample["input"][:200])
    logger.info("output: %s", sample["output"][:200])


def get_trainable_parameters(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def compute_metrics_from_eval_loss(eval_loss: float) -> Dict[str, float]:
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return {"eval_perplexity": perplexity}


# =========================
# MAIN
# =========================
def main():
    logger.info("LoRA training başladı...")
    set_seed(SEED)

    validate_paths()

    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("BF16 supported: %s", torch.cuda.is_bf16_supported())

    logger.info("Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Base model yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32),
)

    logger.info("LoRA config hazırlanıyor...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )

    logger.info("PEFT/LoRA uygulanıyor...")
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    trainable, total = get_trainable_parameters(model)
    logger.info("Trainable params: %s", f"{trainable:,}")
    logger.info("Total params: %s", f"{total:,}")
    logger.info("Trainable ratio: %.4f%%", 100 * trainable / total)

    logger.info("Dataset yükleniyor...")
    train_records = load_jsonl(TRAIN_PATH)
    val_records = load_jsonl(VAL_PATH)

    validate_dataset(train_records, "train")
    validate_dataset(val_records, "val")
    print_dataset_preview(train_records)

    logger.info("Train kayıt sayısı: %s", len(train_records))
    logger.info("Val kayıt sayısı: %s", len(val_records))

    logger.info("Format dönüşümü yapılıyor...")
    train_records = [format_example(x) for x in train_records]
    val_records = [format_example(x) for x in val_records]

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    logger.info("Tokenization başlıyor...")
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )

    val_ds = val_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val dataset",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("TrainingArguments hazırlanıyor...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=USE_BF16,
        fp16=USE_FP16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        load_best_model_at_end=False,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    logger.info("Training başlıyor...")
    train_result = trainer.train()

    logger.info("Final evaluation çalışıyor...")
    eval_metrics = trainer.evaluate()

    if "eval_loss" in eval_metrics:
        eval_metrics.update(compute_metrics_from_eval_loss(eval_metrics["eval_loss"]))

    logger.info("Train metrics: %s", train_result.metrics)
    logger.info("Eval metrics: %s", eval_metrics)

    logger.info("Model/adapters kaydediliyor...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "config": {
                    "max_length": MAX_LENGTH,
                    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                    "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                    "target_modules": TARGET_MODULES,
                    "bf16": USE_BF16,
                    "fp16": USE_FP16,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Tamamlandı.")
    logger.info("Adapter output path: %s", OUTPUT_DIR)
    logger.info("Metrics path: %s", metrics_path)


if __name__ == "__main__":
    main()