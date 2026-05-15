from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.rag.rag_preprocess import (
    REQUIRED_CANONICAL_COLUMNS,
    find_source_column,
)


REQUIRED_COLUMNS = REQUIRED_CANONICAL_COLUMNS


def get_canonical_series(df: pd.DataFrame, canonical_name: str) -> pd.Series:
    source_column = find_source_column(df, canonical_name)
    if source_column is None:
        return pd.Series(dtype=str)

    return df[source_column]


def run_data_quality_checks(
    df: pd.DataFrame,
    min_rows_required: int = 100,
    min_avg_answer_length: int = 80,
) -> Dict[str, Any]:
    missing_columns: List[str] = [
        col for col in REQUIRED_COLUMNS if find_source_column(df, col) is None
    ]

    answer_text_series = get_canonical_series(df, "answer_text")
    question_text_series = get_canonical_series(df, "question_text")
    topic_series = get_canonical_series(df, "topic")

    answer_lengths = answer_text_series.fillna("").astype(str).str.len()
    avg_answer_length = float(answer_lengths.mean()) if len(answer_lengths) > 0 else 0.0

    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "null_question_text_count": int(question_text_series.isna().sum()) if len(question_text_series) else None,
        "null_answer_text_count": int(answer_text_series.isna().sum()) if len(answer_text_series) else None,
        "avg_answer_length": round(avg_answer_length, 2),
        "topic_distribution": topic_series.fillna("NULL").astype(str).value_counts().to_dict() if len(topic_series) else {},
        "checks": {
            "enough_rows": len(df) >= min_rows_required,
            "enough_avg_answer_length": avg_answer_length >= min_avg_answer_length,
            "required_columns_present": len(missing_columns) == 0,
        },
    }

    report["overall_pass"] = all(
        value is True for value in report["checks"].values()
    )

    return report
