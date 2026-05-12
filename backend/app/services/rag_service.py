from __future__ import annotations

import re
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

        self.follow_up_markers = [
            "this", "that", "these", "those", "it", "they",
            "them", "he", "she", "when i do this", "when i do these things",
            "will i feel better", "does this help", "will that help"
        ]

        self.bad_context_patterns = [
            r"the other .* post answers?",
            r"the other .* answers?",
            r"as mentioned above",
            r"see the illustration below",
            r"see below",
            r"below:",
            r"working with me you will learn",
            r"other post",
            r"other answers",
        ]

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

    def _is_follow_up_query(self, query: str) -> bool:
        q = query.lower().strip()

        if len(q.split()) <= 8:
            if any(marker in q for marker in self.follow_up_markers):
                return True

        if q.startswith(("and ", "but ", "so ", "then ")):
            return True

        return False

    def build_retrieval_query(self, user_query: str, chat_history: str | None = None) -> str:
        """
        Follow-up sorularda retrieval'i sadece son mesaja göre yapma.
        Kısa konuşma geçmişinden sinyal ekle.
        """
        if not chat_history or not self._is_follow_up_query(user_query):
            return user_query

        history_lines = []
        for line in chat_history.splitlines():
            line = line.strip()
            if line.startswith("User:"):
                history_lines.append(line.replace("User:", "").strip())

        recent_user_context = " ".join(history_lines[-2:]).strip()

        if recent_user_context:
            return f"{recent_user_context} {user_query}"

        return user_query

    def _sanitize_document_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text

        for pattern in self.bad_context_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _build_context_result(self, rows: list[dict]) -> dict:
        context_chunks = []
        topics_used = []

        for item in rows:
            topic = item.get("topic", "unknown")
            score = item.get("score", 0.0)
            quality = item.get("quality_score", 0.0)
            doc = self._sanitize_document_text(item.get("rag_document", ""))

            if not doc:
                continue

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
            "document_count": len(context_chunks),
            "topics_used": list(dict.fromkeys(topics_used)),
            "documents": rows,
        }

    def _qdrant_collection_exists(self) -> bool:
        try:
            collections = self.qdrant_client.get_collections().collections
            names = [c.name for c in collections]
            return QDRANT_COLLECTION_NAME in names
        except Exception:
            return False

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
            limit=max(top_k * 2, 8),
            query_filter=query_filter,
        )

        rows = []
        for r in results:
            payload = r.payload or {}
            score = float(r.score)

            # Çok düşük relevance sonuçları agresif şekilde alma
            if score < 0.18:
                continue

            rows.append({
                "topic": payload.get("topic", "unknown"),
                "quality_score": float(payload.get("quality_score", 0.0)),
                "rag_document": payload.get("rag_document", ""),
                "score": score,
            })

        rows = rows[:top_k]
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

        # Çok alakasız sonuçları ele
        temp_df = temp_df[temp_df["final_score"] >= 0.20].copy()

        if temp_df.empty:
            # hiçbir şey kalmazsa tekrar geniş arama yap
            temp_df = self.local_df.copy()
            temp_df["semantic_score"] = sims
            temp_df["final_score"] = (
                0.85 * temp_df["semantic_score"] +
                0.15 * temp_df["quality_score"]
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

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        chat_history: str | None = None,
    ) -> dict:
        retrieval_query = self.build_retrieval_query(
            user_query=query,
            chat_history=chat_history,
        )

        if self._qdrant_collection_exists():
            try:
                return self._retrieve_from_qdrant(retrieval_query, top_k)
            except Exception:
                pass

        return self._retrieve_from_local(retrieval_query, top_k)