from __future__ import annotations

import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
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
RAW_PATH = Path(os.getenv("COUNSEL_RAW_PATH", "data/raw/counsel_revised_zero_data_loss.xlsx"))
RAW_SHEET_NAME = os.getenv("COUNSEL_RAW_SHEET", "rag_ready")
OUT_DIR = Path("data/gold")

FULL_OUTPUT_PATH = OUT_DIR / "counsel_full.parquet"
TRAIN_OUTPUT_PATH = OUT_DIR / "counsel_train.parquet"
VAL_OUTPUT_PATH = OUT_DIR / "counsel_val.parquet"
STATS_OUTPUT_PATH = OUT_DIR / "counsel_stats.json"


# =========================
# CONFIG
# =========================
TARGET_TOPICS = None

PRIMARY_TOPICS = {"anxiety", "depression"}
SECONDARY_TOPICS: set[str] = set()

TOPIC_NORMALIZATION_MAP = {
    "anxiety": "anxiety",
    "depression": "depression",
    "self-esteem": "self-esteem",
    "workplace-relationships": "workplace-relationships",
    "stress": "stress",
    "behavioral-change": "behavioral-change",
}

REQUIRED_COLUMNS = [
    "topic",
    "upvotes",
]

COLUMN_ALIASES = {
    "question_id": ["question_id", "questionID"],
    "question_title": ["question_title", "questionTitle"],
    "question_text": ["question_text", "questionText"],
    "question_link": ["question_link", "questionLink"],
    "topic": ["topic"],
    "therapist_info": ["therapist_info", "therapistInfo"],
    "therapist_url": ["therapist_url", "therapistURL"],
    "answer_text": ["answer_text", "answer", "answerText"],
    "upvotes": ["upvotes"],
    "views": ["views"],
    "original_question_text": ["original_question_text"],
    "detected_approach": ["detected_approach"],
    "revision_status": ["revision_status"],
    "context_cluster": ["context_cluster"],
    "revision_note": ["revision_note"],
}

REQUIRED_CANONICAL_COLUMNS = [
    "question_id",
    "question_title",
    "question_text",
    "topic",
    "answer_text",
    "upvotes",
]

OPTIONAL_COLUMN_DEFAULTS = {
    "question_link": "",
    "therapist_info": "",
    "therapist_url": "",
    "views": 0,
    "original_question_text": "",
    "detected_approach": "",
    "revision_status": "",
    "context_cluster": "",
    "revision_note": "",
}

MIN_TEXT_LENGTH = 15
VAL_RATIO = 0.10
RANDOM_STATE = 42
REMOVE_URLS = True
DROP_LOW_QUALITY_ARTIFACTS = os.getenv("COUNSEL_DROP_ARTIFACTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}


# =========================
# REGEX
# =========================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")
WHITESPACE_PATTERN = re.compile(r"\s+")

FORUM_STYLE_REPLACEMENTS = [
    (
        re.compile(r"^\s*(yeah|yes)[,. ]+i get the same problem\.{0,3}\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\blook[,]?\s+i'?m not a professional but i'?ve heard (?:a few )?things\.?\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bi'?m not a professional\b[,. ]*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bi am not a professional\b[,. ]*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bi'?ve heard (?:a few )?things\.?\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(
            r"\ba powernap can help\.?\s*just a half hour of sleep can clear your mind and let you refocus\.?",
            re.IGNORECASE,
        ),
        "A brief reset can help you clear your mind and refocus.",
    ),
    (
        re.compile(
            r"\balso, brain activity increases with physical exertion\.?\s*just walk around for a minute and get your brain working and that'll help you reach the task at hand\.?",
            re.IGNORECASE,
        ),
        "A short walk can also help you reset before returning to the task at hand.",
    ),
    (
        re.compile(r"\btaking breaks it totally okay\b", re.IGNORECASE),
        "Taking breaks is okay",
    ),
]

BAD_ANSWER_PATTERNS = [
    r"\bthe other .* post answers?\b",
    r"\bthe other .* answers?\b",
    r"\bas mentioned above\b",
    r"\bsee (the )?illustration below\b",
    r"\bsee below\b",
    r"\bworking with me you will learn\b",
    r"\ball about thinking errors\b",
    r"\bother post\b",
    r"\bother answers\b",
]

CLICKBAIT_TITLE_PATTERNS = [
    r"^\d+\s+ways\b",
    r"^\d+\s+signs\b",
    r"^\d+\s+reasons\b",
    r"^\d+\s+tips\b",
    r"^\d+\s+benefits\b",
]

ELLIPSIS_HEAVY_PATTERN = re.compile(r"\.\.\.")
NUMBERED_LIST_HEAVY_PATTERN = re.compile(r"(?:^|\s)\d+\.")
BAD_ANSWER_REGEXES = [re.compile(p, re.IGNORECASE) for p in BAD_ANSWER_PATTERNS]
CLICKBAIT_REGEXES = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_TITLE_PATTERNS]


# =========================
# HELPERS
# =========================
def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_input_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file bulunamadı: {file_path}")


def read_source_dataframe(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(file_path)
        selected_sheet = (
            sheet_name
            if sheet_name in excel_file.sheet_names
            else excel_file.sheet_names[0]
        )
        return pd.read_excel(file_path, sheet_name=selected_sheet)

    if suffix == ".csv":
        return pd.read_csv(file_path)

    raise ValueError(f"Desteklenmeyen raw veri formati: {file_path.suffix}")


def find_source_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    for alias in aliases:
        if alias in df.columns:
            return alias

    return None


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        canonical_name
        for canonical_name in REQUIRED_CANONICAL_COLUMNS
        if find_source_column(df, canonical_name) is None
    ]
    if missing_columns:
        raise ValueError(f"Eksik kolonlar bulundu: {missing_columns}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df)

    standardized = pd.DataFrame(index=df.index)

    for canonical_name in REQUIRED_CANONICAL_COLUMNS:
        source_column = find_source_column(df, canonical_name)
        standardized[canonical_name] = df[source_column]

    for canonical_name, default_value in OPTIONAL_COLUMN_DEFAULTS.items():
        source_column = find_source_column(df, canonical_name)
        if source_column is None:
            standardized[canonical_name] = default_value
        else:
            standardized[canonical_name] = df[source_column]

    for numeric_column in ["upvotes", "views"]:
        standardized[numeric_column] = pd.to_numeric(
            standardized[numeric_column],
            errors="coerce",
        ).fillna(0)

    return standardized.copy()


def normalize_text(text: Optional[str], remove_urls: bool = True) -> str:
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")

    if remove_urls:
        text = URL_PATTERN.sub(" ", text)

    text = EMAIL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def sanitize_answer_for_generation(text: Optional[str]) -> str:
    sanitized = normalize_text(text, remove_urls=True)

    for pattern, replacement in FORUM_STYLE_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)

    sanitized = WHITESPACE_PATTERN.sub(" ", sanitized).strip()
    sanitized = re.sub(r"^[.\s]+", "", sanitized).strip()
    if not sanitized:
        return ""

    return sanitized[:1].upper() + sanitized[1:]


def normalize_topic(topic: Optional[str]) -> str:
    if topic is None or pd.isna(topic):
        return "other"

    topic = str(topic).strip().lower()
    topic = TOPIC_NORMALIZATION_MAP.get(topic, topic)
    topic = WHITESPACE_PATTERN.sub("-", topic)
    return topic or "other"


def has_min_length(text: str, min_len: int = 15) -> bool:
    return isinstance(text, str) and len(text.strip()) >= min_len


def get_topic_tier(topic: str) -> str:
    if topic in PRIMARY_TOPICS:
        return "primary"
    if topic in SECONDARY_TOPICS or topic != "other":
        return "secondary"
    return "other"


def build_rag_document(
    question_title: str,
    topic: str,
    question_text: str,
    answer_text: str,
    detected_approach: str = "",
    context_cluster: str = "",
) -> str:
    parts = [
        f"Title: {question_title}",
        f"Topic: {topic}",
    ]

    if detected_approach:
        parts.append(f"Approach: {detected_approach}")

    if context_cluster:
        parts.append(f"Context Cluster: {context_cluster}")

    parts.extend(
        [
            f"Question: {question_text}",
            f"Answer: {answer_text}",
        ]
    )
    return "\n".join(parts)


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    upvotes = df["upvotes"].fillna(0).clip(lower=0)
    views = df["views"].fillna(0).clip(lower=0)
    answer_length = df["answer_text"].str.len().fillna(0)

    views_log = np.log1p(views)

    def minmax(series: pd.Series) -> pd.Series:
        min_v = series.min()
        max_v = series.max()

        if max_v == min_v:
            return pd.Series([0.5] * len(series), index=series.index)

        return (series - min_v) / (max_v - min_v)

    upvotes_norm = minmax(upvotes)
    views_norm = minmax(views_log)
    answer_len_norm = minmax(answer_length)

    df["quality_score"] = (
        0.45 * upvotes_norm
        + 0.20 * views_norm
        + 0.35 * answer_len_norm
    ).round(4)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "questionID": "question_id",
            "questionTitle": "question_title",
            "questionText": "question_text",
            "questionLink": "question_link",
            "topic": "topic",
            "therapistInfo": "therapist_info",
            "therapistURL": "therapist_url",
            "answerText": "answer_text",
            "upvotes": "upvotes",
            "views": "views",
        }
    ).copy()


def clean_text_columns(df: pd.DataFrame, remove_urls: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["question_title"] = df["question_title"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["question_text"] = df["question_text"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["answer_text"] = df["answer_text"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["answer_for_generation"] = df["answer_text"].apply(sanitize_answer_for_generation)
    df["original_question_text"] = df["original_question_text"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["detected_approach"] = df["detected_approach"].apply(
        lambda x: normalize_text(x, remove_urls=False)
    )
    df["context_cluster"] = df["context_cluster"].apply(
        lambda x: normalize_text(x, remove_urls=False)
    )
    return df


def clean_topic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["topic"] = df["topic"].apply(normalize_topic)
    df["topic_tier"] = df["topic"].apply(get_topic_tier)
    return df


def apply_answer_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["answer_text"].str.strip() != ""].copy()
    df.loc[df["answer_for_generation"].str.strip() == "", "answer_for_generation"] = df["answer_text"]
    return df


def apply_question_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question_text"] = df.apply(
        lambda row: row["question_text"]
        if row["question_text"].strip()
        else row["question_title"],
        axis=1,
    )
    return df


def apply_min_length_filter(df: pd.DataFrame, min_text_length: int) -> pd.DataFrame:
    df = df.copy()
    return df[
        df["question_text"].apply(lambda x: has_min_length(x, min_text_length))
        & df["answer_text"].apply(lambda x: has_min_length(x, min_text_length))
    ].copy()


def apply_target_topic_filter(df: pd.DataFrame, target_topics: set[str]) -> pd.DataFrame:
    df = df.copy()
    if not target_topics:
        return df

    return df[df["topic"].isin(target_topics)].copy()


def contains_bad_answer_pattern(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern in BAD_ANSWER_REGEXES)


def looks_like_clickbait_title(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    return any(pattern.search(text) for pattern in CLICKBAIT_REGEXES)


def is_low_quality_artifact_answer(text: str) -> bool:
    if not isinstance(text, str):
        return True

    t = text.strip()
    if not t:
        return True

    if contains_bad_answer_pattern(t):
        return True

    if looks_like_clickbait_title(t):
        return True

    if ELLIPSIS_HEAVY_PATTERN.search(t) and len(t) < 500:
        return True

    if NUMBERED_LIST_HEAVY_PATTERN.search(t) and len(t.split()) < 50:
        return True

    return False


def apply_bad_answer_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = ~df["answer_text"].apply(is_low_quality_artifact_answer)
    return df[mask].copy()


def apply_title_artifact_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = ~df["question_title"].apply(looks_like_clickbait_title)
    return df[mask].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["answer_length"] = df["answer_text"].str.len()
    df["doc_id"] = (
        df["question_id"].astype(str)
        + "_"
        + df.groupby("question_id").cumcount().astype(str)
    )

    df["rag_document"] = df.apply(
        lambda row: build_rag_document(
            question_title=row["question_title"],
            topic=row["topic"],
            question_text=row["question_text"],
            answer_text=row["answer_for_generation"],
            detected_approach=row.get("detected_approach", ""),
            context_cluster=row.get("context_cluster", ""),
        ),
        axis=1,
    )

    df = compute_quality_score(df)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "doc_id",
        "question_id",
        "question_title",
        "question_text",
        "answer_text",
        "answer_for_generation",
        "topic",
        "topic_tier",
        "upvotes",
        "views",
        "answer_length",
        "quality_score",
        "original_question_text",
        "detected_approach",
        "revision_status",
        "context_cluster",
        "rag_document",
    ]
    return df[keep_columns].reset_index(drop=True)


def preprocess_counsel_dataset(
    file_path: Path,
    target_topics: Optional[set[str]] = None,
    min_text_length: int = 15,
    remove_urls: bool = True,
) -> pd.DataFrame:
    if target_topics is None:
        target_topics = TARGET_TOPICS

    validate_input_file(file_path)

    logger.info("Raw veri okunuyor...")
    df = read_source_dataframe(file_path, sheet_name=RAW_SHEET_NAME)

    logger.info("Kolon kontrolü yapılıyor...")
    validate_required_columns(df)

    logger.info("Kolon isimleri standart hale getiriliyor...")
    df = standardize_columns(df)

    logger.info("Metin sütunları temizleniyor...")
    df = clean_text_columns(df, remove_urls=remove_urls)

    logger.info("Topic normalize ediliyor...")
    df = clean_topic_columns(df)

    logger.info("Boş answer kayıtları atılıyor...")
    before = len(df)
    df = apply_answer_filter(df)
    logger.info("Answer filter: %s -> %s", before, len(df))

    logger.info("Question fallback uygulanıyor...")
    df = apply_question_fallback(df)

    logger.info("Minimum uzunluk filtresi uygulanıyor...")
    before = len(df)
    df = apply_min_length_filter(df, min_text_length=min_text_length)
    logger.info("Length filter: %s -> %s", before, len(df))

    logger.info("Target topic filtresi uygulanıyor...")
    before = len(df)
    df = apply_target_topic_filter(df, target_topics=target_topics)
    logger.info("Topic filter: %s -> %s", before, len(df))

    if DROP_LOW_QUALITY_ARTIFACTS:
        logger.info("Artifact / düşük kalite answer filtresi uygulanıyor...")
        before = len(df)
        df = apply_bad_answer_filter(df)
        logger.info("Bad answer filter: %s -> %s", before, len(df))

        logger.info("Clickbait / artifact title filtresi uygulanıyor...")
        before = len(df)
        df = apply_title_artifact_filter(df)
        logger.info("Title artifact filter: %s -> %s", before, len(df))
    else:
        logger.info("Artifact filtreleri atlandi; zero-data-loss modu aktif.")

    if df.empty:
        raise ValueError("Tüm filtrelerden sonra veri boş kaldı.")

    logger.info("Feature kolonları ekleniyor...")
    df = add_features(df)

    logger.info("Final kolonlar seçiliyor...")
    df = select_final_columns(df)

    return df


def safe_train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_questions = df[["question_id", "topic"]].drop_duplicates().copy()

    topic_counts = unique_questions["topic"].value_counts()
    can_stratify = len(topic_counts) > 1 and (topic_counts >= 2).all()

    if can_stratify:
        logger.info("Stratified split uygulanıyor...")
        train_q, val_q = train_test_split(
            unique_questions,
            test_size=test_size,
            random_state=random_state,
            stratify=unique_questions["topic"],
        )
    else:
        logger.warning("Stratify mümkün değil, random split uygulanıyor.")
        train_q, val_q = train_test_split(
            unique_questions,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    train_ids = set(train_q["question_id"])
    val_ids = set(val_q["question_id"])

    train_df = df[df["question_id"].isin(train_ids)].copy()
    val_df = df[df["question_id"].isin(val_ids)].copy()

    return train_df, val_df


def build_stats(full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    return {
        "full_shape": list(full_df.shape),
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "full_topic_distribution": full_df["topic"].value_counts().to_dict(),
        "train_topic_distribution": train_df["topic"].value_counts().to_dict(),
        "val_topic_distribution": val_df["topic"].value_counts().to_dict(),
        "topic_tier_distribution": full_df["topic_tier"].value_counts().to_dict(),
        "duplicate_question_text_count": int(full_df["question_text"].duplicated().sum()),
        "duplicate_answer_count": int(full_df["answer_text"].duplicated().sum()),
        "sanitized_answer_count": int(
            (full_df["answer_text"] != full_df["answer_for_generation"]).sum()
        ),
        "revision_status_distribution": full_df["revision_status"].value_counts().to_dict(),
        "detected_approach_distribution": full_df["detected_approach"].value_counts().to_dict(),
        "quality_score_summary": {
            "min": float(full_df["quality_score"].min()),
            "max": float(full_df["quality_score"].max()),
            "mean": float(full_df["quality_score"].mean()),
            "median": float(full_df["quality_score"].median()),
        },
        "answer_length_summary": {
            "min": int(full_df["answer_length"].min()),
            "max": int(full_df["answer_length"].max()),
            "mean": float(full_df["answer_length"].mean()),
            "median": float(full_df["answer_length"].median()),
        },
    }


def save_outputs(full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    logger.info("Output klasörü hazırlanıyor...")
    ensure_output_dir(OUT_DIR)

    logger.info("Parquet dosyaları kaydediliyor...")
    full_df.to_parquet(FULL_OUTPUT_PATH, index=False)
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_parquet(VAL_OUTPUT_PATH, index=False)

    logger.info("Stats json kaydediliyor...")
    stats = build_stats(full_df, train_df, val_df)
    with open(STATS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    logger.info("Counsel gold dataset build başladı...")

    full_df = preprocess_counsel_dataset(
        file_path=RAW_PATH,
        target_topics=TARGET_TOPICS,
        min_text_length=MIN_TEXT_LENGTH,
        remove_urls=REMOVE_URLS,
    )

    logger.info("Train/val split yapılıyor...")
    train_df, val_df = safe_train_val_split(
        full_df,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    save_outputs(full_df, train_df, val_df)

    logger.info("Tamamlandı.")
    logger.info("FULL  : %s", full_df.shape)
    logger.info("TRAIN : %s", train_df.shape)
    logger.info("VAL   : %s", val_df.shape)
    logger.info("Topic distribution: %s", full_df["topic"].value_counts().to_dict())


if __name__ == "__main__":
    main()
