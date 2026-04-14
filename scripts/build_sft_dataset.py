from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd


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

COUNSEL_TRAIN_PATH = BASE_DIR / "data" / "gold" / "counsel_train.parquet"
COUNSEL_VAL_PATH = BASE_DIR / "data" / "gold" / "counsel_val.parquet"

EMOTION_TRAIN_PATH = BASE_DIR / "data" / "llm" / "emotion_tone_train.jsonl"
EMOTION_VAL_PATH = BASE_DIR / "data" / "llm" / "emotion_tone_val.jsonl"

OUT_DIR = BASE_DIR / "data" / "llm"
RAG_TONE_TRAIN_PATH = OUT_DIR / "rag_tone_train.jsonl"
RAG_TONE_VAL_PATH = OUT_DIR / "rag_tone_val.jsonl"
SFT_TRAIN_PATH = OUT_DIR / "sft_train.jsonl"
SFT_VAL_PATH = OUT_DIR / "sft_val.jsonl"
STATS_PATH = OUT_DIR / "sft_stats.json"


# =========================
# CONFIG
# =========================
TOPIC_TO_TONE = {
    "anxiety": "calm, reassuring, grounded",
    "depression": "warm, validating, gentle",
    "self-esteem": "affirming, non-judgmental, supportive",
    "workplace-relationships": "respectful, balanced, supportive",
    "stress": "steady, practical, supportive",
    "behavioral-change": "encouraging, practical, supportive",
}

EMOTION_MIX_RATIO = 0.35   # RAG train boyutunun %35'i kadar emotion örneği ekle
MIN_EMOTION_VAL = 100      # val için minimum emotion örneği


# =========================
# HELPERS
# =========================
def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_counsel_columns(df: pd.DataFrame) -> None:
    required = {
        "question_text",
        "answer_text",
        "topic",
        "rag_document",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Counsel parquet içinde eksik kolonlar var: {missing}")


def get_tone_from_topic(topic: str) -> str:
    return TOPIC_TO_TONE.get(topic, "warm, validating, supportive")


# =========================
# BUILD RAG TONE DATASET
# =========================
def build_rag_record(row: pd.Series) -> Dict:
    tone = get_tone_from_topic(row["topic"])

    return {
        "instruction": "Answer the user using the provided context. Be empathetic, grounded, and do not go beyond the context.",
        "input": (
            f"Context:\n{row['rag_document']}\n\n"
            f"User Question: {row['question_text']}\n"
            f"Topic: {row['topic']}\n"
            f"Tone: {tone}"
        ),
        "output": row["answer_text"],
        "source": "rag_tone",
    }


def build_rag_records(df: pd.DataFrame) -> List[Dict]:
    return [build_rag_record(row) for _, row in df.iterrows()]


# =========================
# MIX DATASETS
# =========================
def mix_datasets(
    rag_records: List[Dict],
    emotion_records: List[Dict],
    ratio: float,
) -> List[Dict]:
    rag_count = len(rag_records)
    emotion_take = int(rag_count * ratio)

    if emotion_take <= 0:
        return rag_records.copy()

    emotion_take = min(emotion_take, len(emotion_records))
    merged = rag_records + emotion_records[:emotion_take]
    return merged


def mix_val_datasets(
    rag_records: List[Dict],
    emotion_records: List[Dict],
    ratio: float,
    min_emotion_val: int,
) -> List[Dict]:
    rag_count = len(rag_records)
    emotion_take = max(min_emotion_val, int(rag_count * ratio))
    emotion_take = min(emotion_take, len(emotion_records))

    merged = rag_records + emotion_records[:emotion_take]
    return merged


# =========================
# STATS
# =========================
def count_sources(records: List[Dict]) -> Dict[str, int]:
    counts = {}
    for r in records:
        src = r.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def build_stats(
    rag_train: List[Dict],
    rag_val: List[Dict],
    emotion_train: List[Dict],
    emotion_val: List[Dict],
    sft_train: List[Dict],
    sft_val: List[Dict],
) -> Dict:
    return {
        "rag_train_count": len(rag_train),
        "rag_val_count": len(rag_val),
        "emotion_train_count": len(emotion_train),
        "emotion_val_count": len(emotion_val),
        "sft_train_count": len(sft_train),
        "sft_val_count": len(sft_val),
        "sft_train_source_distribution": count_sources(sft_train),
        "sft_val_source_distribution": count_sources(sft_val),
        "emotion_mix_ratio": EMOTION_MIX_RATIO,
        "min_emotion_val": MIN_EMOTION_VAL,
    }


# =========================
# MAIN
# =========================
def main():
    logger.info("SFT dataset build başladı...")

    ensure_output_dir(OUT_DIR)

    validate_file(COUNSEL_TRAIN_PATH)
    validate_file(COUNSEL_VAL_PATH)
    validate_file(EMOTION_TRAIN_PATH)
    validate_file(EMOTION_VAL_PATH)

    logger.info("Counsel parquet dosyaları okunuyor...")
    counsel_train_df = pd.read_parquet(COUNSEL_TRAIN_PATH)
    counsel_val_df = pd.read_parquet(COUNSEL_VAL_PATH)

    validate_counsel_columns(counsel_train_df)
    validate_counsel_columns(counsel_val_df)

    logger.info("RAG tone kayıtları oluşturuluyor...")
    rag_train_records = build_rag_records(counsel_train_df)
    rag_val_records = build_rag_records(counsel_val_df)

    logger.info("Emotion tone kayıtları okunuyor...")
    emotion_train_records = load_jsonl(EMOTION_TRAIN_PATH)
    emotion_val_records = load_jsonl(EMOTION_VAL_PATH)

    logger.info("RAG tone datasetleri kaydediliyor...")
    write_jsonl(RAG_TONE_TRAIN_PATH, rag_train_records)
    write_jsonl(RAG_TONE_VAL_PATH, rag_val_records)

    logger.info("Final SFT train/val datasetleri birleştiriliyor...")
    sft_train_records = mix_datasets(
        rag_records=rag_train_records,
        emotion_records=emotion_train_records,
        ratio=EMOTION_MIX_RATIO,
    )

    sft_val_records = mix_val_datasets(
        rag_records=rag_val_records,
        emotion_records=emotion_val_records,
        ratio=EMOTION_MIX_RATIO,
        min_emotion_val=MIN_EMOTION_VAL,
    )

    logger.info("Final SFT datasetleri kaydediliyor...")
    write_jsonl(SFT_TRAIN_PATH, sft_train_records)
    write_jsonl(SFT_VAL_PATH, sft_val_records)

    stats = build_stats(
        rag_train=rag_train_records,
        rag_val=rag_val_records,
        emotion_train=emotion_train_records,
        emotion_val=emotion_val_records,
        sft_train=sft_train_records,
        sft_val=sft_val_records,
    )

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Tamamlandı.")
    logger.info("rag_train : %s", len(rag_train_records))
    logger.info("rag_val   : %s", len(rag_val_records))
    logger.info("emotion_train : %s", len(emotion_train_records))
    logger.info("emotion_val   : %s", len(emotion_val_records))
    logger.info("sft_train : %s", len(sft_train_records))
    logger.info("sft_val   : %s", len(sft_val_records))
    logger.info("SFT train source dist: %s", count_sources(sft_train_records))
    logger.info("SFT val source dist  : %s", count_sources(sft_val_records))


if __name__ == "__main__":
    main()