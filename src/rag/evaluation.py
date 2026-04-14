from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_corpus_evaluation_report(corpus_df: pd.DataFrame) -> Dict[str, Any]:
    total_rows = len(corpus_df)

    if total_rows == 0:
        return {
            "total_rows": 0,
            "avg_quality_score": 0.0,
            "avg_answer_length": 0.0,
            "duplicate_title_ratio": 0.0,
            "primary_topic_ratio": 0.0,
            "secondary_topic_ratio": 0.0,
            "topic_distribution": {},
        }

    duplicate_title_count = corpus_df["question_title"].duplicated().sum()
    duplicate_title_ratio = duplicate_title_count / total_rows

    primary_count = (corpus_df["topic_tier"] == "primary").sum()
    secondary_count = (corpus_df["topic_tier"] == "secondary").sum()

    report = {
        "total_rows": int(total_rows),
        "avg_quality_score": round(float(corpus_df["quality_score"].mean()), 4),
        "avg_answer_length": round(float(corpus_df["answer_length"].mean()), 2),
        "duplicate_title_ratio": round(float(duplicate_title_ratio), 4),
        "primary_topic_ratio": round(float(primary_count / total_rows), 4),
        "secondary_topic_ratio": round(float(secondary_count / total_rows), 4),
        "topic_distribution": corpus_df["topic"].value_counts().to_dict(),
        "topic_tier_distribution": corpus_df["topic_tier"].value_counts().to_dict(),
    }

    return report