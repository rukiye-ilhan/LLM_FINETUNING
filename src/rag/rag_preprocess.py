from __future__ import annotations

import html
import re
from typing import Optional

import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")
WHITESPACE_PATTERN = re.compile(r"\s+")


TARGET_TOPICS = {
    "anxiety",
    "depression",
    "self-esteem",
    "workplace-relationships",
    "stress",
    "behavioral-change",
}

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


def normalize_text(text: Optional[str], remove_urls: bool = True) -> str:
    """
    Metni normalize eder:
    - HTML decode
    - URL temizliği
    - email/telefon temizliği
    - whitespace düzeltme
    """
    if text is None:
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


def normalize_topic(topic: Optional[str]) -> str:
    if topic is None:
        return "other"

    topic = str(topic).strip().lower()
    return TOPIC_NORMALIZATION_MAP.get(topic, "other")


def has_min_length(text: str, min_len: int = 15) -> bool:
    return isinstance(text, str) and len(text.strip()) >= min_len


def build_rag_document(
    question_title: str,
    topic: str,
    question_text: str,
    answer_text: str,
) -> str:
    parts = [
        f"Title: {question_title}",
        f"Topic: {topic}",
        f"Question: {question_text}",
        f"Answer: {answer_text}",
    ]
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
    target_topics: Optional[set[str]] = None,
    min_text_length: int = 15,
    remove_urls: bool = True,
) -> pd.DataFrame:
    """
    Ham counsel datasetini RAG için hazır hale getirir.
    """
    if target_topics is None:
        target_topics = TARGET_TOPICS

    df = pd.read_csv(file_path)

    required_columns = [
        "questionID",
        "questionTitle",
        "questionText",
        "questionLink",
        "topic",
        "therapistInfo",
        "therapistURL",
        "answerText",
        "upvotes",
        "views",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik kolonlar bulundu: {missing_columns}")

    df = df.rename(
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

    df["question_title"] = df["question_title"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["question_text"] = df["question_text"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["answer_text"] = df["answer_text"].apply(
        lambda x: normalize_text(x, remove_urls=remove_urls)
    )
    df["topic"] = df["topic"].apply(normalize_topic)

    df["topic_tier"] = df["topic"].apply(get_topic_tier)

    # answer boşsa kayıt kullanılamaz
    df = df[df["answer_text"].str.strip() != ""].copy()

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
            answer_text=row["answer_text"],
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
    "topic",
    "topic_tier",
    "upvotes",
    "views",
    "answer_length",
    "quality_score",
    "rag_document",
]

    return df[keep_columns].reset_index(drop=True)

def get_topic_tier(topic: str) -> str:
    if topic in PRIMARY_TOPICS:
        return "primary"
    if topic in SECONDARY_TOPICS:
        return "secondary"
    return "other"