from __future__ import annotations

from src.rag.retriever import RagRetriever


def main() -> None:
    query = "I feel anxious and overwhelmed and I cannot manage my stress lately"

    retriever = RagRetriever(
        collection_name="counsel_rag",
        db_path="data/vector_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    results = retriever.retrieve(query=query, top_k=5)

    print(f"\n[QUERY] {query}\n")
    print("[TOP RESULTS]\n")

    for item in results:
        answer_preview = (item["answer_text"] or "")[:300].replace("\n", " ")

        print(f"Rank         : {item['rank']}")
        print(f"Score        : {item['score']:.4f}")
        print(f"Doc ID       : {item['doc_id']}")
        print(f"Topic        : {item['topic']}")
        print(f"Title        : {item['question_title']}")
        print(f"Quality      : {item['quality_score']}")
        print(f"Upvotes      : {item['upvotes']}")
        print(f"Views        : {item['views']}")
        print(f"Answer Prev. : {answer_preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()