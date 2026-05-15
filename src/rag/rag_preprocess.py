from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
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


TARGET_TOPICS: set[str] = set()

PRIMARY_TOPICS = {"anxiety", "depression"}
SECONDARY_TOPICS = {
    "stress",
    "self-esteem",
    "workplace-relationships",
    "behavioral-change",
}


TOPIC_NORMALIZATION_MAP = {
    "anxiety": "anxiety",
    "depression": "depression",
    "self-esteem": "self-esteem",
    "workplace-relationships": "workplace-relationships",
    "stress": "stress",
    "behavioral-change": "behavioral-change",
}

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


def load_counsel_source(file_path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(path)
        selected_sheet = (
            sheet_name
            if sheet_name in excel_file.sheet_names
            else excel_file.sheet_names[0]
        )
        return pd.read_excel(path, sheet_name=selected_sheet)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported counsel source format: {path.suffix}")


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
        raise ValueError(f"Missing required counsel columns: {missing_columns}")


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
    """
    Metni normalize eder:
    - HTML decode
    - URL temizliği
    - email/telefon temizliği
    - whitespace düzeltme
    """
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
    """
    Kalite skoru üretir.
    Şimdilik:
    - upvotes
    - views
    - answer length
    bazlı ilerliyoruz.
    """
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


def preprocess_counsel_dataset(
    file_path: str,
    sheet_name: Optional[str] = None,
    target_topics: Optional[set[str]] = None,
    min_text_length: int = 15,
    remove_urls: bool = True,
) -> pd.DataFrame:
    """
    Ham counsel datasetini RAG için hazır hale getirir.
    """
    if target_topics is None:
        target_topics = TARGET_TOPICS

    df = load_counsel_source(file_path, sheet_name=sheet_name)
    df = standardize_columns(df)

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
    df["topic"] = df["topic"].apply(normalize_topic)

    df["topic_tier"] = df["topic"].apply(get_topic_tier)

    # answer boşsa kayıt kullanılamaz
    df = df[df["answer_text"].str.strip() != ""].copy()
    df.loc[df["answer_for_generation"].str.strip() == "", "answer_for_generation"] = df["answer_text"]

    # question_text boşsa title fallback
    df["question_text"] = df.apply(
        lambda row: row["question_text"]
        if row["question_text"].strip()
        else row["question_title"],
        axis=1,
    )

    # minimum uzunluk filtresi
    df = df[
        df["question_text"].apply(lambda x: has_min_length(x, min_text_length))
        & df["answer_text"].apply(lambda x: has_min_length(x, min_text_length))
    ].copy()

    # topic filtre
    if target_topics:
        df = df[df["topic"].isin(target_topics)].copy()

    # answer length
    df["answer_length"] = df["answer_text"].str.len()

    # doc_id: aynı question_id için birden fazla cevap olabilir
    df["doc_id"] = (
        df["question_id"].astype(str)
        + "_"
        + df.groupby("question_id").cumcount().astype(str)
    )

    # rag document
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

    # quality score
    df = compute_quality_score(df)

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


def get_topic_tier(topic: str) -> str:
    if topic in PRIMARY_TOPICS:
        return "primary"
    if topic in SECONDARY_TOPICS or topic != "other":
        return "secondary"
    return "other"
