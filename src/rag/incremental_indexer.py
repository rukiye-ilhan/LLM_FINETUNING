from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.rag.embedder import TextEmbedder
from src.rag.vectordb import QdrantVectorDB
from src.rag.corpus_registry import (
    build_registry_from_corpus,
    diff_registry,
    load_registry,
    save_registry,
)
from src.rag.id_utils import stable_numeric_id


def build_payloads_from_df(df: pd.DataFrame):
    payloads = []
    for _, row in df.iterrows():
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
    return payloads


def run_incremental_index_update(
    corpus_df: pd.DataFrame,
    collection_name: str,
    vector_db_path: str,
    embedding_model_name: str,
    embedding_batch_size: int,
    registry_path: str,
) -> Dict[str, Any]:
    old_registry = load_registry(registry_path)
    new_registry = build_registry_from_corpus(corpus_df)

    new_doc_ids, changed_doc_ids, unchanged_doc_ids, deleted_doc_ids = diff_registry(
        old_registry, new_registry
    )

    delta_doc_ids = set(new_doc_ids + changed_doc_ids)
    delta_df = corpus_df[corpus_df["doc_id"].astype(str).isin(delta_doc_ids)].copy()

    embedder = TextEmbedder(model_name=embedding_model_name)
    vector_size = embedder.get_embedding_dimension()

    vectordb = QdrantVectorDB(
    collection_name=COLLECTION_NAME,
    vector_size=vector_size,
    host="localhost",
    port=6333,
)

    indexed_count = 0
    deleted_count = 0

    if len(delta_df) > 0:
        embeddings = embedder.encode_texts(
            delta_df["rag_document"].tolist(),
            batch_size=embedding_batch_size,
        )

        payloads = build_payloads_from_df(delta_df)

        ids = [
            stable_numeric_id(str(doc_id))
            for doc_id in delta_df["doc_id"].astype(str).tolist()
        ]

        vectordb.upsert_documents(ids=ids, embeddings=embeddings, payloads=payloads)
        indexed_count = len(delta_df)

    if len(deleted_doc_ids) > 0:
        delete_ids = [stable_numeric_id(str(doc_id)) for doc_id in deleted_doc_ids]
        vectordb.delete_points(delete_ids)
        deleted_count = len(delete_ids)

    save_registry(new_registry, registry_path)

    return {
        "old_registry_count": len(old_registry),
        "new_registry_count": len(new_registry),
        "new_doc_count": len(new_doc_ids),
        "changed_doc_count": len(changed_doc_ids),
        "unchanged_doc_count": len(unchanged_doc_ids),
        "deleted_doc_count": len(deleted_doc_ids),
        "delta_indexed_count": indexed_count,
        "delta_deleted_count": deleted_count,
    }