from __future__ import annotations

from src.rag.retriever import RagRetriever
from src.rag.reranker import RagReranker


def print_results(title: str, results):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}\n")

    for item in results:
        answer_preview = (item.get("answer_text") or "")[:250].replace("\n", " ")

        print(f"Rank           : {item.get('rank', item.get('reranked_rank'))}")
        print(f"Score          : {item.get('score', '-')}")
        print(f"Final Score    : {item.get('final_score', '-')}")
        print(f"Topic          : {item.get('topic')}")
        print(f"Title          : {item.get('question_title')}")
        print(f"Quality        : {item.get('quality_score')}")
        print(f"Topic Bonus    : {item.get('topic_bonus', '-')}")
        print(f"Dup Penalty    : {item.get('duplicate_penalty_applied', '-')}")
        print(f"Answer Preview : {answer_preview}")
        print("-" * 100)


def main() -> None:
    query = "I feel anxious and overwhelmed and I cannot manage my stress lately"

    retriever = RagRetriever(
        collection_name="counsel_rag",
        db_path="data/vector_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    reranker = RagReranker(
        semantic_weight=0.75,
        quality_weight=0.20,
        topic_weight=0.05,
        duplicate_title_penalty=0.08,
    )

    retrieved_results = retriever.retrieve(query=query, top_k=10)
    reranked_results = reranker.rerank(query=query, results=retrieved_results, final_top_k=5)

    print(f"\n[QUERY] {query}\n")

    print_results("[RAW RETRIEVAL RESULTS]", retrieved_results)
    print_results("[RERANKED RESULTS]", reranked_results)


if __name__ == "__main__":
    main()