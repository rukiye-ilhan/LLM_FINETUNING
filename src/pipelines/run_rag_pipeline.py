from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.config import ensure_directory, load_yaml_config
from src.common.logger import get_logger
from src.rag.data_quality import run_data_quality_checks
from src.rag.rag_preprocess import preprocess_counsel_dataset
from src.rag.embedder import TextEmbedder
from src.rag.vectordb import QdrantVectorDB
from src.rag.incremental_indexer import run_incremental_index_update
from src.rag.evaluation import build_corpus_evaluation_report


def save_json(data, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_as_jsonl(records, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("[INFO] Dynamic RAG pipeline başladı...")

    config = load_yaml_config("configs/rag_config.yaml")

    reports_dir = ensure_directory(config["paths"]["reports_dir"])
    processed_dir = ensure_directory(config["paths"]["processed_dir"])
    pipeline_runs_dir = ensure_directory(config["paths"]["pipeline_runs_dir"])

    logger = get_logger("DynamicRAGPipeline", log_dir=pipeline_runs_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_directory(pipeline_runs_dir / run_id)

    raw_data_path = config["paths"]["raw_data_path"]
    logger.info(f"Raw veri okunuyor: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)

    logger.info("Data quality checks çalıştırılıyor...")
    quality_report = run_data_quality_checks(
        raw_df,
        min_rows_required=config["quality"]["min_rows_required"],
        min_avg_answer_length=config["quality"]["min_avg_answer_length"],
    )

    quality_report_path = reports_dir / config["outputs"]["quality_report_name"]
    save_json(quality_report, quality_report_path)
    save_json(quality_report, run_dir / config["outputs"]["quality_report_name"])

    logger.info(f"Quality overall_pass = {quality_report['overall_pass']}")

    if config["quality"]["fail_on_missing_required_columns"]:
        if not quality_report["checks"]["required_columns_present"]:
            raise ValueError(
                f"Eksik required kolonlar var: {quality_report['missing_required_columns']}"
            )

    logger.info("RAG corpus üretiliyor...")
    corpus_df = preprocess_counsel_dataset(
        file_path=raw_data_path,
        target_topics=set(config["rag"]["target_topics"]),
        min_text_length=config["rag"]["min_text_length"],
        remove_urls=config["rag"]["remove_urls"],
    )

    parquet_path = processed_dir / config["outputs"]["corpus_parquet_name"]
    jsonl_path = processed_dir / config["outputs"]["corpus_jsonl_name"]

    corpus_df.to_parquet(parquet_path, index=False)
    save_as_jsonl(corpus_df.to_dict(orient="records"), jsonl_path)

    logger.info(f"Corpus üretildi. Satır sayısı: {len(corpus_df)}")
    logger.info(f"Corpus parquet: {parquet_path}")
    logger.info(f"Corpus jsonl: {jsonl_path}")
    evaluation_report = build_corpus_evaluation_report(corpus_df)

    evaluation_report_path = reports_dir / config["outputs"]["evaluation_report_name"]
    save_json(evaluation_report, evaluation_report_path)
    save_json(evaluation_report, run_dir / config["outputs"]["evaluation_report_name"])

    logger.info(f"Evaluation report kaydedildi: {evaluation_report_path}")


    indexing_enabled = config["indexing"]["enabled"]
    indexing_mode = config["indexing"]["mode"]

    indexed_count = 0
    incremental_report = None

    if indexing_enabled and indexing_mode == "full_reindex":
        logger.info("Full reindex başlıyor...")

        embedder = TextEmbedder(model_name=config["embedding"]["model_name"])
        vector_size = embedder.get_embedding_dimension()

        vectordb = QdrantVectorDB(
            collection_name=config["rag"]["collection_name"],
            vector_size=vector_size,
            db_path=config["paths"]["vector_db_path"],
        )

        vectordb.recreate_collection()

        embeddings = embedder.encode_texts(
            corpus_df["rag_document"].tolist(),
            batch_size=config["embedding"]["batch_size"],
        )

        ids = list(range(len(corpus_df)))

        payloads = []
        for _, row in corpus_df.iterrows():
            payloads.append(
                {
                    "doc_id": row["doc_id"],
                    "question_id": int(row["question_id"]),
                    "question_title": row["question_title"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                    "topic": row["topic"],
                    "topic_tier": row["topic_tier"],
                    "upvotes": int(row["upvotes"]),
                    "views": int(row["views"]),
                    "answer_length": int(row["answer_length"]),
                    "quality_score": float(row["quality_score"]),
                    "rag_document": row["rag_document"],
                }
            )

        vectordb.upsert_documents(ids=ids, embeddings=embeddings, payloads=payloads)
        indexed_count = len(payloads)
        logger.info(f"Indexing tamamlandı. Indexed count = {indexed_count}")

    elif indexing_enabled and indexing_mode == "incremental_update":
        logger.info("Incremental update başlıyor...")

        incremental_report = run_incremental_index_update(
            corpus_df=corpus_df,
            collection_name=config["rag"]["collection_name"],
            vector_db_path=config["paths"]["vector_db_path"],
            embedding_model_name=config["embedding"]["model_name"],
            embedding_batch_size=config["embedding"]["batch_size"],
            registry_path=config["indexing"]["registry_path"],
        )

        indexed_count = int(incremental_report["delta_indexed_count"])
        logger.info(f"Incremental update tamamlandı: {incremental_report}")

    elif indexing_enabled and indexing_mode == "skip_for_now":
        logger.info("Indexing açık ama mode=skip_for_now; indexing atlandı.")
    else:
        logger.info("Indexing kapalı.")

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "project_name": config["project"]["name"],
        "environment": config["project"]["environment"],
        "raw_data_path": raw_data_path,
        "processed_corpus_path": str(parquet_path),
        "processed_jsonl_path": str(jsonl_path),
        "raw_row_count": int(len(raw_df)),
        "processed_row_count": int(len(corpus_df)),
        "indexed_count": int(indexed_count),
        "collection_name": config["rag"]["collection_name"],
        "indexing_enabled": bool(indexing_enabled),
        "indexing_mode": indexing_mode,
        "quality_overall_pass": bool(quality_report["overall_pass"]),
        "topic_distribution_processed": corpus_df["topic"].value_counts().to_dict(),
        "incremental_report": incremental_report,
        "evaluation_report": evaluation_report,
    }

    save_json(run_metadata, run_dir / config["outputs"]["run_metadata_name"])
    logger.info(f"Run metadata kaydedildi: {run_dir / config['outputs']['run_metadata_name']}")
    logger.info("Dynamic RAG pipeline tamamlandı.")
    print("[DONE] Dynamic RAG pipeline tamamlandı.")


if __name__ == "__main__":
    main()