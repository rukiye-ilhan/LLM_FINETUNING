from __future__ import annotations

import pandas as pd

from src.rag.embedder import TextEmbedder
from src.rag.vectordb import QdrantVectorDB


CORPUS_PATH = "data/processed/rag_corpus.parquet"
COLLECTION_NAME = "counsel_rag"


def main() -> None:
    print("[INFO] RAG corpus yükleniyor...")
    df = pd.read_parquet(CORPUS_PATH)

    print(f"[INFO] Kayıt sayısı: {len(df)}")

    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_size = embedder.get_embedding_dimension()

    vectordb = QdrantVectorDB(
    collection_name=COLLECTION_NAME,
    vector_size=vector_size,
    host="localhost",
    port=6333,
)

    print("[INFO] Collection yeniden oluşturuluyor...")
    vectordb.recreate_collection()

    print("[INFO] Doküman embedding'leri üretiliyor...")
    embeddings = embedder.encode_texts(df["rag_document"].tolist(), batch_size=32)

    # Qdrant local mode için integer id kullanıyoruz
    ids = list(range(len(df)))

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

    print("[INFO] Qdrant'a upsert ediliyor...")
    vectordb.upsert_documents(ids=ids, embeddings=embeddings, payloads=payloads)

    print("[DONE] Corpus indexing tamamlandı.")


if __name__ == "__main__":
    main()