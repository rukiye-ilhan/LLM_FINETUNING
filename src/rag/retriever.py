from __future__ import annotations

from typing import Any, Dict, List

from src.rag.embedder import TextEmbedder
from src.rag.vectordb import QdrantVectorDB


class RagRetriever:
    def __init__(
        self,
        collection_name: str = "counsel_rag",
        db_path: str = "data/vector_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedder = TextEmbedder(model_name=model_name)
        self.vectordb = QdrantVectorDB(
            collection_name=collection_name,
            vector_size=self.embedder.get_embedding_dimension(),
            db_path=db_path,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedder.encode_query(query)

        results = self.vectordb.search(
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        formatted_results = []
        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            formatted_results.append(
                {
                    "rank": rank,
                    "score": float(result.score),
                    "doc_id": payload.get("doc_id"),
                    "question_id": payload.get("question_id"),
                    "topic": payload.get("topic"),
                    "topic_tier": payload.get("topic_tier"),
                    "question_title": payload.get("question_title"),
                    "question_text": payload.get("question_text"),
                    "answer_text": payload.get("answer_text"),
                    "quality_score": payload.get("quality_score"),
                    "upvotes": payload.get("upvotes"),
                    "views": payload.get("views"),
                }
            )

        return formatted_results