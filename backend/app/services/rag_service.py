import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

from backend.app.core.config import (
    COUNSEL_FULL_PATH,
    EMBED_MODEL_NAME,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TOP_K,
)


class RagService:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.local_df = pd.read_parquet(COUNSEL_FULL_PATH)
        docs = self.local_df["rag_document"].fillna("").tolist()
        self.local_embeddings = self.embedder.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _infer_topic_hints(self, query: str) -> list[str]:
        q = query.lower()
        hints = []

        if any(x in q for x in ["work", "manager", "office", "job", "coworker", "boss"]):
            hints.append("workplace-relationships")
        if any(x in q for x in ["anxiety", "anxious", "panic", "worried", "overwhelmed"]):
            hints.append("anxiety")
        if any(x in q for x in ["stress", "stressed", "pressure"]):
            hints.append("stress")
        if any(x in q for x in ["not good enough", "self-doubt", "insecure", "worthless"]):
            hints.append("self-esteem")
        if any(x in q for x in ["depressed", "hopeless", "empty", "sad"]):
            hints.append("depression")

        return list(dict.fromkeys(hints))

    def _build_context_result(self, rows: list[dict]) -> dict:
        context_chunks = []
        topics_used = []

        for item in rows:
            topic = item.get("topic", "unknown")
            score = item.get("score", 0.0)
            quality = item.get("quality_score", 0.0)
            doc = item.get("rag_document", "")

            topics_used.append(topic)

            context_chunks.append(
                f"[Document]\n"
                f"Topic: {topic}\n"
                f"Score: {score:.4f}\n"
                f"Quality: {quality:.4f}\n"
                f"{doc}"
            )

        return {
            "context_text": "\n\n".join(context_chunks),
            "document_count": len(rows),
            "topics_used": list(dict.fromkeys(topics_used)),
            "documents": rows,
        }

    def _qdrant_collection_exists(self) -> bool:
        collections = self.qdrant_client.get_collections().collections
        names = [c.name for c in collections]
        return QDRANT_COLLECTION_NAME in names

    def _retrieve_from_qdrant(self, query: str, top_k: int) -> dict:
        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        topic_hints = self._infer_topic_hints(query)

        query_filter = None
        if topic_hints:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="topic",
                        match=MatchAny(any=topic_hints)
                    )
                ]
            )

        results = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
        )

        rows = []
        for r in results:
            payload = r.payload or {}
            rows.append({
                "topic": payload.get("topic", "unknown"),
                "quality_score": float(payload.get("quality_score", 0.0)),
                "rag_document": payload.get("rag_document", ""),
                "score": float(r.score),
            })

        return self._build_context_result(rows)

    def _retrieve_from_local(self, query: str, top_k: int) -> dict:
        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        sims = np.dot(self.local_embeddings, query_vec)

        temp_df = self.local_df.copy()
        temp_df["semantic_score"] = sims

        topic_hints = self._infer_topic_hints(query)
        if topic_hints:
            filtered = temp_df[temp_df["topic"].isin(topic_hints)].copy()
            if not filtered.empty:
                temp_df = filtered

        temp_df["final_score"] = (
            0.80 * temp_df["semantic_score"] +
            0.20 * temp_df["quality_score"]
        )

        temp_df = temp_df.sort_values("final_score", ascending=False).head(top_k)

        rows = []
        for _, row in temp_df.iterrows():
            rows.append({
                "topic": row.get("topic", "unknown"),
                "quality_score": float(row.get("quality_score", 0.0)),
                "rag_document": row.get("rag_document", ""),
                "score": float(row.get("final_score", 0.0)),
            })

        return self._build_context_result(rows)

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        if self._qdrant_collection_exists():
            return self._retrieve_from_qdrant(query, top_k)

        return self._retrieve_from_local(query, top_k)