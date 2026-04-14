from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


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
RAW_PATH = BASE_DIR / "data" / "raw" / "emotion-emotion_69k.csv"
OUT_DIR = BASE_DIR / "data" / "llm"

TRAIN_OUTPUT_PATH = OUT_DIR / "emotion_tone_train.jsonl"
VAL_OUTPUT_PATH = OUT_DIR / "emotion_tone_val.jsonl"
STATS_OUTPUT_PATH = OUT_DIR / "emotion_tone_stats.json"


# =========================
# CONFIG
# =========================
REQUIRED_COLUMNS = ["Situation", "emotion", "empathetic_dialogues", "labels"]
MIN_USER_TEXT_LENGTH = 15
MIN_OUTPUT_LENGTH = 3
VAL_RATIO = 0.10
RANDOM_STATE = 42

VALID_EMOTIONS = {
    "surprised",
    "excited",
    "angry",
    "proud",
    "sad",
    "annoyed",
    "lonely",
    "grateful",
    "afraid",
    "terrified",
    "disgusted",
    "furious",
    "guilty",
    "anxious",
    "anticipating",
    "confident",
    "hopeful",
    "impressed",
    "nostalgic",
    "disappointed",
    "jealous",
    "joyful",
    "prepared",
    "content",
    "devastated",
    "embarrassed",
    "sentimental",
    "caring",
    "trusting",
    "ashamed",
    "apprehensive",
    "faithful",
}


# =========================
# REGEX
# =========================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")
WHITESPACE_PATTERN = re.compile(r"\s+")


# =========================
# EMOTION -> TONE
# =========================
EMOTION_TO_TONE = {
    "anxious": "calm, reassuring, grounded",
    "afraid": "safe, calm, reassuring",
    "terrified": "very calm, stabilizing, gentle",
    "apprehensive": "calm, reassuring, careful",
    "sad": "warm, validating, gentle",
    "lonely": "warm, validating, connecting",
    "guilty": "non-judgmental, compassionate, reflective",
    "ashamed": "non-judgmental, validating, gentle",
    "disappointed": "gentle, validating, hopeful",
    "devastated": "very gentle, validating, supportive",
    "angry": "calm, respectful, de-escalating",
    "annoyed": "calm, respectful, validating",
    "furious": "calm, contained, de-escalating",
    "disgusted": "respectful, calm, validating",
    "excited": "positive, warm, engaged",
    "hopeful": "encouraging, warm, supportive",
    "proud": "affirming, positive, warm",
    "grateful": "warm, positive, reflective",
    "content": "warm, positive, steady",
    "joyful": "warm, positive, engaged",
    "confident": "supportive, affirming, clear",
    "embarrassed": "gentle, non-judgmental, supportive",
    "jealous": "non-judgmental, reflective, calm",
    "sentimental": "warm, reflective, gentle",
    "surprised": "calm, curious, attentive",
    "impressed": "warm, engaged, reflective",
    "nostalgic": "warm, reflective, gentle",
    "caring": "warm, supportive, attentive",
    "faithful": "steady, warm, supportive",
    "trusting": "warm, steady, reassuring",
    "prepared": "clear, calm, supportive",
    "anticipating": "calm, engaged, attentive",
}


# =========================
# HELPERS
# =========================
def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_input_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file bulunamadı: {file_path}")


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def has_min_length(text: str, min_len: int = 15) -> bool:
    return isinstance(text, str) and len(text.strip()) >= min_len


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik kolonlar bulundu: {missing_columns}")


def clean_dialogue_prefix(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\bCustomer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAgent\s*:\s*", "", text, flags=re.IGNORECASE)
    return normalize_text(text)


def normalize_emotion_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    label = str(label).strip().lower()
    if label == "nan":
        return ""
    return label


def get_tone(emotion: str) -> str:
    return EMOTION_TO_TONE.get(emotion, "warm, validating, supportive")


def build_user_text(situation: str, dialogue: str) -> str:
    if has_min_length(dialogue, 10):
        return dialogue
    return situation


def build_output(label_text: str, emotion: str) -> str:
    label_text = normalize_text(label_text)
    if has_min_length(label_text, MIN_OUTPUT_LENGTH):
        return label_text
    return f"It sounds like you're feeling {emotion}. I'm here with you."


def build_instruction_record(user_text: str, emotion: str, tone: str, output: str) -> dict:
    return {
        "instruction": "Respond to the user with an empathetic and emotionally appropriate tone.",
        "input": f"User: {user_text}\nEmotion: {emotion}\nTone: {tone}",
        "output": output,
        "source": "emotion_tone",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_emotion_from_input(text: str) -> str:
    match = re.search(r"Emotion:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return "unknown"


# =========================
# PIPELINE
# =========================
def load_and_clean_raw_dataset(file_path: Path) -> pd.DataFrame:
    validate_input_file(file_path)

    logger.info("Emotion CSV okunuyor...")
    df = pd.read_csv(file_path)

    logger.info("Gereksiz kolonlar temizleniyor...")
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    logger.info("Kolon kontrolü yapılıyor...")
    validate_required_columns(df)

    logger.info("Emotion label temizleniyor...")
    df["emotion"] = df["emotion"].apply(normalize_emotion_label)

    before = len(df)
    df = df[df["emotion"] != ""].copy()
    logger.info("Empty emotion filter: %s -> %s", before, len(df))

    before = len(df)
    df = df[df["emotion"].isin(VALID_EMOTIONS)].copy()
    logger.info("Valid emotion whitelist filter: %s -> %s", before, len(df))

    logger.info("Metin alanları temizleniyor...")
    df["situation_clean"] = df["Situation"].apply(normalize_text)
    df["dialogue_clean"] = df["empathetic_dialogues"].apply(clean_dialogue_prefix)
    df["label_clean"] = df["labels"].apply(normalize_text)

    logger.info("User text oluşturuluyor...")
    df["user_text"] = df.apply(
        lambda row: build_user_text(
            situation=row["situation_clean"],
            dialogue=row["dialogue_clean"],
        ),
        axis=1,
    )

    logger.info("Length filter uygulanıyor...")
    before = len(df)
    df = df[
        df["user_text"].apply(lambda x: has_min_length(x, MIN_USER_TEXT_LENGTH))
        & df["label_clean"].apply(lambda x: has_min_length(x, MIN_OUTPUT_LENGTH))
    ].copy()
    logger.info("Length filter: %s -> %s", before, len(df))

    logger.info("Tone kolonları ekleniyor...")
    df["tone"] = df["emotion"].apply(get_tone)

    logger.info("Duplicate temizleniyor...")
    before = len(df)
    df = df.drop_duplicates(subset=["user_text", "emotion", "label_clean"]).reset_index(drop=True)
    logger.info("Dedup: %s -> %s", before, len(df))

    if df.empty:
        raise ValueError("Temizleme sonrası emotion dataset boş kaldı.")

    return df


def build_instruction_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Instruction dataset oluşturuluyor...")
    records = []

    for _, row in df.iterrows():
        output = build_output(row["label_clean"], row["emotion"])
        records.append(
            build_instruction_record(
                user_text=row["user_text"],
                emotion=row["emotion"],
                tone=row["tone"],
                output=output,
            )
        )

    out_df = pd.DataFrame(records)

    if out_df.empty:
        raise ValueError("Instruction dataset boş üretildi.")

    return out_df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Train/val split yapılıyor...")

    emotion_labels = df["input"].apply(extract_emotion_from_input)
    emotion_counts = emotion_labels.value_counts()
    can_stratify = len(emotion_counts) > 1 and (emotion_counts >= 2).all()

    if can_stratify:
        train_df, val_df = train_test_split(
            df,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            stratify=emotion_labels,
        )
    else:
        logger.warning("Emotion stratify mümkün değil, random split uygulanıyor.")
        train_df, val_df = train_test_split(
            df,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_stats(raw_df: pd.DataFrame, instruction_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    return {
        "raw_clean_shape": list(raw_df.shape),
        "instruction_shape": list(instruction_df.shape),
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "emotion_distribution": raw_df["emotion"].value_counts().to_dict(),
        "tone_distribution": raw_df["tone"].value_counts().to_dict(),
        "user_text_length_summary": {
            "min": int(raw_df["user_text"].str.len().min()),
            "max": int(raw_df["user_text"].str.len().max()),
            "mean": float(raw_df["user_text"].str.len().mean()),
            "median": float(raw_df["user_text"].str.len().median()),
        },
        "output_length_summary": {
            "min": int(raw_df["label_clean"].str.len().min()),
            "max": int(raw_df["label_clean"].str.len().max()),
            "mean": float(raw_df["label_clean"].str.len().mean()),
            "median": float(raw_df["label_clean"].str.len().median()),
        },
    }


def save_outputs(raw_df: pd.DataFrame, instruction_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    ensure_output_dir(OUT_DIR)

    logger.info("JSONL dosyaları kaydediliyor...")
    write_jsonl(TRAIN_OUTPUT_PATH, train_df.to_dict(orient="records"))
    write_jsonl(VAL_OUTPUT_PATH, val_df.to_dict(orient="records"))

    logger.info("Stats json kaydediliyor...")
    stats = build_stats(raw_df, instruction_df, train_df, val_df)
    with open(STATS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    logger.info("Emotion tone dataset hazırlama başladı...")

    raw_clean_df = load_and_clean_raw_dataset(RAW_PATH)
    instruction_df = build_instruction_dataset(raw_clean_df)
    train_df, val_df = split_dataset(instruction_df)
    save_outputs(raw_clean_df, instruction_df, train_df, val_df)

    logger.info("Tamamlandı.")
    logger.info("RAW CLEAN : %s", raw_clean_df.shape)
    logger.info("INSTRUCT  : %s", instruction_df.shape)
    logger.info("TRAIN     : %s", train_df.shape)
    logger.info("VAL       : %s", val_df.shape)
    logger.info("Emotion distribution: %s", raw_clean_df["emotion"].value_counts().to_dict())


if __name__ == "__main__":
    main()