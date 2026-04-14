from __future__ import annotations

import json
from pathlib import Path

from src.rag.rag_preprocess import preprocess_counsel_dataset


RAW_DATA_PATH = "data/raw/counsel_chat_orijinal.csv"
OUTPUT_DIR = "data/processed"
PARQUET_OUTPUT = "rag_corpus.parquet"
JSONL_OUTPUT = "rag_corpus.jsonl"


def save_as_jsonl(records, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_counsel_dataset(
        file_path=RAW_DATA_PATH,
        target_topics={
            "anxiety",
            "depression",
            "self-esteem",
            "workplace-relationships",
            "stress",
            "behavioral-change",
        },
        min_text_length=15,
        remove_urls=True,
    )

    parquet_path = output_dir / PARQUET_OUTPUT
    jsonl_path = output_dir / JSONL_OUTPUT

    df.to_parquet(parquet_path, index=False)
    save_as_jsonl(df.to_dict(orient="records"), jsonl_path)

    print(f"[INFO] RAG corpus oluşturuldu: {len(df)} kayıt")
    print(f"[INFO] Parquet çıktı: {parquet_path}")
    print(f"[INFO] JSONL çıktı: {jsonl_path}")

    print("\n[INFO] Topic dağılımı:")
    print(df["topic"].value_counts())

    print("\n[INFO] Quality score özeti:")
    print(df["quality_score"].describe())


if __name__ == "__main__":
    main()