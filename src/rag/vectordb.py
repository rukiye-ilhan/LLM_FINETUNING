from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantVectorDB:
    def __init__(
        self,
        collection_name: str = "counsel_rag",
        vector_size: int = 384,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(host=host, port=port)

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        return self.collection_name in names

    def ensure_collection(self) -> None:
        if not self.collection_exists():
            self.recreate_collection()

    def upsert_documents(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 64,
    ) -> None:
        if not (len(ids) == len(embeddings) == len(payloads)):
            raise ValueError("ids, embeddings ve payloads uzunlukları eşit olmalı.")

        self.ensure_collection()

        for start_idx in range(0, len(ids), batch_size):
            end_idx = start_idx + batch_size

            batch_ids = ids[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_payloads = payloads[start_idx:end_idx]

            points = [
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
                for idx, vector, payload in zip(batch_ids, batch_embeddings, batch_payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def delete_points(self, ids: List[int]) -> None:
        if not ids:
            return

        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return response.points