from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
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


def run_data_quality_checks(
    df: pd.DataFrame,
    min_rows_required: int = 100,
    min_avg_answer_length: int = 80,
) -> Dict[str, Any]:
    missing_columns: List[str] = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    answer_text_series = df["answerText"] if "answerText" in df.columns else pd.Series(dtype=str)
    question_text_series = df["questionText"] if "questionText" in df.columns else pd.Series(dtype=str)
    topic_series = df["topic"] if "topic" in df.columns else pd.Series(dtype=str)

    answer_lengths = answer_text_series.fillna("").astype(str).str.len()
    avg_answer_length = float(answer_lengths.mean()) if len(answer_lengths) > 0 else 0.0

    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "null_question_text_count": int(question_text_series.isna().sum()) if "questionText" in df.columns else None,
        "null_answer_text_count": int(answer_text_series.isna().sum()) if "answerText" in df.columns else None,
        "avg_answer_length": round(avg_answer_length, 2),
        "topic_distribution": topic_series.fillna("NULL").astype(str).value_counts().to_dict() if "topic" in df.columns else {},
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