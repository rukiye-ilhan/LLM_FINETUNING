from __future__ import annotations

from typing import Any, Dict, List


class RagContextBuilder:
    def __init__(
        self,
        max_documents: int = 4,
        max_chars_per_doc: int = 900,
        drop_duplicate_titles: bool = True,
        max_same_topic: int = 2,
    ):
        self.max_documents = max_documents
        self.max_chars_per_doc = max_chars_per_doc
        self.drop_duplicate_titles = drop_duplicate_titles
        self.max_same_topic = max_same_topic

    def _truncate(self, text: str, max_chars: int) -> str:
        if not text:
            return ""

        text = str(text).replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
        return truncated + "..."

    def _format_document(self, item: Dict[str, Any], idx: int) -> str:
        title = item.get("question_title", "")
        topic = item.get("topic", "")
        topic_tier = item.get("topic_tier", "")
        question = self._truncate(item.get("question_text", ""), 300)
        answer = self._truncate(item.get("answer_text", ""), self.max_chars_per_doc)
        score = item.get("final_score", item.get("score", 0.0))
        quality = item.get("quality_score", 0.0)

        return (
            f"[Document {idx}]\n"
            f"Title: {title}\n"
            f"Topic: {topic}\n"
            f"Topic Tier: {topic_tier}\n"
            f"Score: {score}\n"
            f"Quality: {quality}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

    def build_context(self, reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected = []
        seen_titles = set()
        topic_counts = {}

        # Önce primary, sonra secondary
        primary_candidates = [x for x in reranked_results if x.get("topic_tier") == "primary"]
        secondary_candidates = [x for x in reranked_results if x.get("topic_tier") == "secondary"]
        ordered_candidates = primary_candidates + secondary_candidates

        for item in ordered_candidates:
            title = (item.get("question_title") or "").strip().lower()
            topic = item.get("topic", "")

            if self.drop_duplicate_titles and title in seen_titles:
                continue

            current_topic_count = topic_counts.get(topic, 0)
            if current_topic_count >= self.max_same_topic:
                continue

            selected.append(item)
            seen_titles.add(title)
            topic_counts[topic] = current_topic_count + 1

            if len(selected) >= self.max_documents:
                break

        context_blocks = [
            self._format_document(item, idx=i + 1)
            for i, item in enumerate(selected)
        ]

        full_context = "\n\n".join(context_blocks)

        return {
            "selected_documents": selected,
            "context_text": full_context,
            "document_count": len(selected),
            "topics_used": [x.get("topic") for x in selected],
        }