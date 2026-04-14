from __future__ import annotations

from typing import Any, Dict, List


class RagReranker:
    def __init__(
        self,
        semantic_weight: float = 0.65,
        quality_weight: float = 0.20,
        topic_weight: float = 0.15,
        duplicate_title_penalty: float = 0.10,
        mismatch_penalty: float = 0.06,
    ):
        self.semantic_weight = semantic_weight
        self.quality_weight = quality_weight
        self.topic_weight = topic_weight
        self.duplicate_title_penalty = duplicate_title_penalty
        self.mismatch_penalty = mismatch_penalty

    def infer_query_topics(self, query: str) -> set[str]:
        query_lower = query.lower()
        matched_topics = set()

        anxiety_keywords = {
            "anxiety", "anxious", "panic", "panicked", "worry", "worried",
            "overwhelmed", "fear", "afraid", "nervous"
        }
        stress_keywords = {
            "stress", "stressed", "pressure", "burnout", "burned out", "tense"
        }
        depression_keywords = {
            "depression", "depressed", "hopeless", "empty", "sad", "down", "low"
        }
        self_esteem_keywords = {
            "worthless", "confidence", "self-esteem", "self esteem",
            "insecure", "not good enough", "value myself"
        }
        workplace_keywords = {
            "work", "job", "office", "manager", "boss", "coworker", "colleague"
        }
        behavioral_keywords = {
            "habit", "change", "routine", "motivation", "discipline", "behavior"
        }

        if any(k in query_lower for k in anxiety_keywords):
            matched_topics.add("anxiety")

        if any(k in query_lower for k in stress_keywords):
            matched_topics.add("stress")

        if any(k in query_lower for k in depression_keywords):
            matched_topics.add("depression")

        if any(k in query_lower for k in self_esteem_keywords):
            matched_topics.add("self-esteem")

        if any(k in query_lower for k in workplace_keywords):
            matched_topics.add("workplace-relationships")

        if any(k in query_lower for k in behavioral_keywords):
            matched_topics.add("behavioral-change")

        return matched_topics

    def compute_topic_bonus(self, doc_topic: str, doc_topic_tier: str, query_topics: set[str]) -> float:
        if not query_topics:
            return 0.0

        if doc_topic in query_topics:
            if doc_topic_tier == "primary":
                return 1.0
            if doc_topic_tier == "secondary":
                return 0.65

        # query anxiety/stress ise depression'ı tamamen dışlama ama hafif destekle
        if "anxiety" in query_topics and doc_topic == "depression":
            return 0.25

        return 0.0

    def compute_mismatch_penalty(self, doc_topic: str, query_topics: set[str]) -> float:
        if not query_topics:
            return 0.0

        if doc_topic in query_topics:
            return 0.0

        # anxiety/stress sorgusunda workplace gereksiz yükselmesin
        if "stress" in query_topics and doc_topic == "workplace-relationships":
            return self.mismatch_penalty

        if "anxiety" in query_topics and doc_topic == "behavioral-change":
            return self.mismatch_penalty / 2

        return 0.0

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        final_top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_topics = self.infer_query_topics(query)

        rescored = []
        seen_titles = set()

        for item in results:
            semantic_score = float(item.get("score", 0.0))
            quality_score = float(item.get("quality_score", 0.0) or 0.0)
            topic = item.get("topic", "")
            topic_tier = item.get("topic_tier", "other")
            title = (item.get("question_title") or "").strip().lower()

            topic_bonus = self.compute_topic_bonus(
                doc_topic=topic,
                doc_topic_tier=topic_tier,
                query_topics=query_topics,
            )

            mismatch_penalty = self.compute_mismatch_penalty(
                doc_topic=topic,
                query_topics=query_topics,
            )

            final_score = (
                self.semantic_weight * semantic_score
                + self.quality_weight * quality_score
                + self.topic_weight * topic_bonus
                - mismatch_penalty
            )

            duplicate_penalty_applied = False
            if title and title in seen_titles:
                final_score -= self.duplicate_title_penalty
                duplicate_penalty_applied = True

            seen_titles.add(title)

            enriched = dict(item)
            enriched["semantic_score"] = semantic_score
            enriched["topic_bonus"] = round(topic_bonus, 4)
            enriched["mismatch_penalty"] = round(mismatch_penalty, 4)
            enriched["final_score"] = round(final_score, 4)
            enriched["duplicate_penalty_applied"] = duplicate_penalty_applied

            rescored.append(enriched)

        rescored.sort(key=lambda x: x["final_score"], reverse=True)

        for rank, item in enumerate(rescored, start=1):
            item["reranked_rank"] = rank

        return rescored[:final_top_k]