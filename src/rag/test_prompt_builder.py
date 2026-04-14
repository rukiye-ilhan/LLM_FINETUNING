from __future__ import annotations

from src.rag.retriever import RagRetriever
from src.rag.reranker import RagReranker
from src.rag.context_builder import RagContextBuilder
from src.rag.prompt_builder import RagPromptBuilder


def main() -> None:
    query = "I feel anxious and overwhelmed and I cannot manage my stress lately"

    retriever = RagRetriever(
        collection_name="counsel_rag",
        db_path="data/vector_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    reranker = RagReranker(
        semantic_weight=0.65,
        quality_weight=0.20,
        topic_weight=0.15,
        duplicate_title_penalty=0.10,
        mismatch_penalty=0.06,
    )

    context_builder = RagContextBuilder(
        max_documents=4,
        max_chars_per_doc=900,
        drop_duplicate_titles=True,
        max_same_topic=2,
    )

    prompt_builder = RagPromptBuilder()

    retrieved = retriever.retrieve(query=query, top_k=15)
    reranked = reranker.rerank(query=query, results=retrieved, final_top_k=8)
    context_result = context_builder.build_context(reranked)
    prompt_result = prompt_builder.build_prompt(user_query=query, context_result=context_result)

    print("\n[SYSTEM PROMPT]\n")
    print(prompt_result["system_prompt"])

    print("\n" + "=" * 100 + "\n")

    print("[USER PROMPT]\n")
    print(prompt_result["user_prompt"])


if __name__ == "__main__":
    main()