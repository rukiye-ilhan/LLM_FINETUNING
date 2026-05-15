from __future__ import annotations

import re
import unicodedata

from backend.app.core.config import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    PROMPT_HISTORY_MESSAGE_LIMIT,
    RAG_ALLOW_EXPLICIT_TOPIC_SHIFT,
    RAG_RETRIEVAL_POLICY,
)
from backend.app.services.emotion_service import EmotionService
from backend.app.services.rag_service import RagService
from backend.app.services.llm_service import LLMService
from backend.app.services.chat_memory_service import ChatMemoryService
from backend.app.services.safety_service import SafetyService


class ChatOrchestrator:
    topic_shift_markers = [
        "new topic",
        "different issue",
        "another issue",
        "another problem",
        "unrelated",
        "can we talk about",
        "let's talk about",
        "lets talk about",
        "switch topic",
        "change topic",
    ]
    closing_markers = [
        "that's enough for today",
        "thats enough for today",
        "that is enough for today",
        "enough for today",
        "that's all for today",
        "thats all for today",
        "that is all for today",
        "we can stop here",
        "let's stop here",
        "lets stop here",
        "i want to stop",
        "i'll stop here",
        "ill stop here",
        "see you",
        "talk later",
        "goodbye",
        "bye",
    ]
    gratitude_markers = [
        "thank you",
        "thanks",
        "you reassured me",
        "you helped me",
        "this helped",
        "that helped",
        "your instructions were correct",
        "your advice helped",
        "i feel better",
    ]

    def __init__(self):
        self.emotion_service = EmotionService()
        self.rag_service = RagService()
        self.llm_service = LLMService()
        self.memory_service = ChatMemoryService()
        self.safety_service = SafetyService()

    def _format_history(
        self,
        messages: list[dict],
        limit: int = PROMPT_HISTORY_MESSAGE_LIMIT,
    ) -> str:
        if not messages:
            return "No previous conversation."

        lines = []
        for msg in messages[-limit:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"].strip()
            if content:
                lines.append(f"{role}: {content}")

        if not lines:
            return "No previous conversation."

        return "\n".join(lines)

    def _has_prior_user_turn(self, messages: list[dict]) -> bool:
        return any(
            msg["role"] == "user" and msg["content"].strip()
            for msg in messages
        )

    def _ascii_fold(self, text: str) -> str:
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    def _is_explicit_topic_shift(self, user_query: str) -> bool:
        normalized = self._ascii_fold(user_query.lower().strip())
        return any(marker in normalized for marker in self.topic_shift_markers)

    def _is_closing_message(self, user_query: str) -> bool:
        normalized = self._ascii_fold(user_query.lower().strip())

        if any(marker in normalized for marker in self.closing_markers):
            return True

        has_gratitude = any(marker in normalized for marker in self.gratitude_markers)
        is_short = len(normalized.split()) <= 18
        has_question = "?" in normalized

        return has_gratitude and is_short and not has_question

    def _build_closing_response(self, user_query: str) -> str:
        normalized = self._ascii_fold(user_query.lower().strip())

        if any(marker in normalized for marker in self.gratitude_markers):
            return (
                "I'm glad this felt reassuring. "
                "It makes sense to pause here for today. "
                "Take care of yourself, and you can come back to this whenever you feel ready."
            )

        return (
            "Of course. We can stop here for today. "
            "Take care of yourself, and you can return to the conversation whenever you want."
        )

    def _should_use_retrieval(
        self,
        user_query: str,
        previous_messages: list[dict],
    ) -> tuple[bool, str]:
        if RAG_RETRIEVAL_POLICY == "always":
            return True, "policy_always"

        if RAG_RETRIEVAL_POLICY == "never":
            return False, "policy_never"

        if not self._has_prior_user_turn(previous_messages):
            return True, "first_user_turn"

        if (
            RAG_RETRIEVAL_POLICY == "first_turn_or_topic_shift"
            and RAG_ALLOW_EXPLICIT_TOPIC_SHIFT
            and self._is_explicit_topic_shift(user_query)
        ):
            return True, "explicit_topic_shift"

        return False, "conversation_memory"

    def _build_retrieval_context_summary(self, messages: list[dict]) -> str:
        if not messages:
            return ""

        user_messages = [
            msg["content"].strip()
            for msg in messages
            if msg["role"] == "user" and msg["content"].strip()
        ]

        if not user_messages:
            return ""

        return " ".join(user_messages[-2:])

    def _compact_text(self, text: str, max_chars: int) -> str:
        compact = " ".join((text or "").split())

        if len(compact) <= max_chars:
            return compact

        return compact[: max_chars - 3].rstrip() + "..."

    def _append_unique(self, values: list[str], new_value: str, max_items: int) -> list[str]:
        cleaned = self._compact_text(new_value, 180)
        if not cleaned:
            return values[-max_items:]

        merged = [item for item in values if item != cleaned]
        merged.append(cleaned)
        return merged[-max_items:]

    def _extract_rag_field(self, document: str, field_name: str) -> str:
        pattern = rf"^{re.escape(field_name)}:\s*(.+)$"
        for line in (document or "").splitlines():
            match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _clean_guidance_phrase(self, text: str) -> str:
        phrase = " ".join((text or "").split()).strip()
        phrase = re.sub(r"^\s*(yeah|yes|look|well)[,.\s]+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\b(best of luck|good luck)\s*!?\s*$", "", phrase, flags=re.IGNORECASE)
        phrase = phrase.strip(" -")
        return phrase

    def _extract_guidance_sentences(self, answer_text: str, max_items: int = 3) -> list[str]:
        cleaned = self._clean_guidance_phrase(answer_text)
        if not cleaned:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        guidance = []
        banned = [
            "i'm not a professional",
            "im not a professional",
            "i've heard",
            "ive heard",
            "best of luck",
            "good luck",
            "milkshake",
            "other post",
            "other answers",
        ]

        for sentence in sentences:
            sentence = self._clean_guidance_phrase(sentence)
            if len(sentence.split()) < 5:
                continue
            if any(pattern in sentence.lower() for pattern in banned):
                continue

            guidance.append(self._compact_text(sentence, 150))
            if len(guidance) >= max_items:
                break

        return guidance

    def _format_approach_label(self, approach: str) -> str:
        if not approach:
            return ""
        return approach.replace("_", " ").replace("-", " ").strip()

    def _build_retrieved_guidance(
        self,
        user_query: str,
        predicted_emotion: str,
        tone: str,
        topics_used: list[str],
        documents: list[dict],
    ) -> str:
        if not documents:
            return "No new retrieved guidance for this turn."

        approaches = []
        principles = []

        for item in documents:
            raw_doc = item.get("rag_document", "")
            approach = self._format_approach_label(
                self._extract_rag_field(raw_doc, "Approach")
            )
            answer_text = self._extract_rag_field(raw_doc, "Answer")

            if approach:
                approaches.append(approach)

            for sentence in self._extract_guidance_sentences(answer_text):
                principles.append(sentence)

        approaches = list(dict.fromkeys(approaches))[:3]
        principles = list(dict.fromkeys(principles))[:4]

        if not principles:
            principles = [
                "Validate the user's feeling without over-identifying with it.",
                "Offer one or two concrete next steps rather than a long list.",
                "Keep the response tied to the user's current situation.",
            ]

        lines = [
            "Use this retrieval result as background guidance, not as text to copy.",
            f"Current user need: {self._compact_text(user_query, 180)}",
            f"Predicted emotion: {predicted_emotion}",
            f"Response tone: {tone}",
        ]

        if topics_used:
            lines.append("Retrieved topics: " + ", ".join(topics_used[:4]))

        if approaches:
            lines.append("Useful counseling approaches: " + ", ".join(approaches))

        lines.append("Guidance principles:")
        for principle in principles:
            lines.append(f"- {principle}")

        lines.extend(
            [
                "Answer naturally in your own words.",
                "Do not mention retrieved documents, forum posts, scores, titles, or source labels.",
            ]
        )

        return "\n".join(lines)

    def _extract_open_questions(self, answer: str) -> list[str]:
        questions = []

        for part in answer.split("?"):
            text = part.strip()
            if not text:
                continue
            tail = text.split(".")[-1].strip()
            if tail:
                questions.append(self._compact_text(tail + "?", 160))

        return questions[-2:]

    def _format_conversation_state(self, state: dict) -> str:
        if not state or int(state.get("turn_count", 0) or 0) == 0:
            return "No established conversation state yet."

        lines = []

        field_labels = [
            ("main_issue", "Main issue"),
            ("current_emotion", "Current emotion"),
            ("user_goal", "User goal"),
            ("summary", "Running summary"),
        ]

        for key, label in field_labels:
            value = str(state.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")

        previous_advice = state.get("previous_advice") or []
        if previous_advice:
            lines.append("Previous advice given: " + " | ".join(previous_advice[-4:]))

        open_questions = state.get("open_questions") or []
        if open_questions:
            lines.append("Open questions: " + " | ".join(open_questions[-3:]))

        last_topics = state.get("last_retrieved_topics") or []
        if last_topics:
            lines.append("Last retrieved topics: " + ", ".join(last_topics[-4:]))

        return "\n".join(lines) if lines else "No established conversation state yet."

    def _build_updated_conversation_state(
        self,
        state: dict,
        user_query: str,
        answer: str,
        predicted_emotion: str,
        topics_used: list[str],
        retrieval_strategy: str,
    ) -> dict:
        updated = dict(state or {})
        turn_count = int(updated.get("turn_count", 0) or 0) + 1

        if not updated.get("main_issue") or retrieval_strategy == "explicit_topic_shift":
            updated["main_issue"] = self._compact_text(user_query, 260)

        if not updated.get("user_goal"):
            updated["user_goal"] = self._compact_text(user_query, 220)

        advice_snapshot = self._compact_text(answer, 220)
        previous_advice = updated.get("previous_advice") or []
        updated["previous_advice"] = self._append_unique(
            previous_advice,
            advice_snapshot,
            max_items=4,
        )

        open_questions = updated.get("open_questions") or []
        for question in self._extract_open_questions(answer):
            open_questions = self._append_unique(open_questions, question, max_items=3)

        if topics_used:
            updated["last_retrieved_topics"] = list(dict.fromkeys(topics_used))[-4:]
        else:
            updated["last_retrieved_topics"] = updated.get("last_retrieved_topics") or []

        updated["current_emotion"] = predicted_emotion
        updated["open_questions"] = open_questions
        updated["turn_count"] = turn_count

        summary_base = str(updated.get("summary") or "").strip()
        latest = (
            f"Turn {turn_count}: user said {self._compact_text(user_query, 160)}. "
            f"Assistant responded by focusing on {self._compact_text(answer, 180)}."
        )
        updated["summary"] = self._compact_text(
            f"{summary_base} {latest}".strip(),
            900,
        )

        return updated

    def _needs_answer_regeneration(self, answer: str) -> bool:
        if not answer or not answer.strip():
            return True

        lowered = answer.lower()
        normalized = lowered.rstrip(" .!?,;:")
        banned_patterns = [
            "the other post answers",
            "the other answers",
            "see below",
            "illustration below",
            "working with me you will learn",
            "9 ways",
            "other post",
            "other answers",
            "i'm not a professional",
            "im not a professional",
            "not a professional",
            "yeah i get",
            "i get the same problem",
            "i've heard",
            "ive heard",
            "look i'm",
            "look im",
            "a powernap",
            "half hour of sleep",
            "best of luck",
            "aka ",
            "milkshake",
        ]

        if any(pattern in lowered for pattern in banned_patterns):
            return True

        if len(answer.split()) > 150:
            return True

        incomplete_endings = (
            "until",
            "because",
            "although",
            "unless",
            "while",
            "if",
            "but",
            "and",
            "or",
            "so",
            "to",
            "for",
            "with",
            "without",
            "about",
            "into",
            "through",
            "how to",
            "what to",
            "where to",
            "who to",
            "whether",
        )
        if normalized.endswith(incomplete_endings):
            return True

        return lowered.startswith(("yeah ", "look ", "well look", "well, look"))

    def _strip_answer_artifacts(self, answer: str) -> str:
        text = (answer or "").strip()
        text = re.sub(
            r"\s*(best of luck|good luck)\s*!?\s*[:;]?-?\)?\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"\s*[:;]-?\)\s*$", "", text).strip()
        return text

    def _is_repetitive_answer(self, answer: str, previous_messages: list[dict]) -> bool:
        previous_answers = [
            msg["content"].strip()
            for msg in previous_messages
            if msg["role"] == "assistant" and msg["content"].strip()
        ]
        if not previous_answers:
            return False

        current_tokens = set(re.findall(r"[a-zA-Z']+", answer.lower()))
        if len(current_tokens) < 8:
            return False

        for previous in previous_answers[-2:]:
            previous_tokens = set(re.findall(r"[a-zA-Z']+", previous.lower()))
            if not previous_tokens:
                continue

            overlap = len(current_tokens & previous_tokens) / max(len(current_tokens), 1)
            length_ratio = min(len(answer), len(previous)) / max(len(answer), len(previous), 1)
            if overlap >= 0.72 and length_ratio >= 0.65:
                return True

        return False

    def _fallback_answer(
        self,
        user_query: str,
        predicted_emotion: str,
    ) -> str:
        lowered_query = user_query.lower()

        if any(term in lowered_query for term in ["who", "talk to", "talking to", "speak to"]):
            return (
                "A good first choice is someone you trust and feel emotionally safe with, like a friend who listens without judging. "
                "If the issue feels too heavy, keeps coming back, or starts affecting daily life, a counselor or mental health professional can add more structured support. "
                "You can start with the person who feels easiest to open up to, then decide whether you need more help."
            )

        if predicted_emotion == "anxiety":
            if any(term in lowered_query for term in ["friend", "she listens", "he listens", "they listen"]):
                return (
                    "Talking with your friend can be a very helpful first step, especially if she listens without judgment. "
                    "It may not solve everything on its own, but it can help you feel less alone and understand what is weighing on you. "
                    "If the overwhelm keeps returning or starts affecting your daily life, adding support from a counselor or another trusted professional could give you more tools."
                )

            if (
                any(term in lowered_query for term in ["work", "job", "manager", "office"])
                and any(term in lowered_query for term in ["not good enough", "self-doubt", "worthless", "insecure"])
            ):
                return (
                    "It makes sense that pressure at work would make the 'not good enough' thought feel louder. "
                    "Try separating the fact in front of you from the self-critical story: name the task, then name the thought as a thought rather than a verdict. "
                    "Choose one small next action, take a brief reset if your body feels activated, and come back to that one step instead of trying to solve everything at once. "
                    "After work, do one grounding thing that helps you recover so the job does not take up the whole day."
                )

            return (
                "It sounds like you are carrying a lot right now. "
                "Try slowing things down and focusing on one small step at a time. "
                "A short pause, steady breathing, and naming what is overwhelming you "
                "can help you feel a little more grounded."
            )

        if predicted_emotion == "distress":
            return (
                "I’m sorry that this feels so heavy. "
                "You do not need to solve everything at once. "
                "Try giving yourself a moment of compassion, and focus on one small, manageable step."
            )

        if predicted_emotion == "self-esteem":
            return (
                "It sounds like this is affecting how you see yourself. "
                "Try to notice the self-critical thought without fully accepting it as truth. "
                "A more helpful next step may be to ask what evidence supports you, not just what hurts you."
            )

        return (
            "Thank you for sharing that. "
            "Let’s slow it down and focus on what feels most important right now. "
            "You do not have to handle everything at once."
        )

    def build_prompt(
        self,
        user_query: str,
        chat_history: str,
        conversation_state: str,
        retrieved_guidance: str,
        predicted_emotion: str,
        tone: str,
        topics_used: list[str],
        document_count: int,
        retrieval_strategy: str,
    ) -> str:
        if retrieved_guidance.strip() and retrieved_guidance != "No new retrieved guidance for this turn.":
            guidance_block = retrieved_guidance
            retrieval_guidance = (
                "A new retrieval result has been converted into short guidance for this turn. "
                "Use the guidance as background, then answer the user naturally."
            )
        else:
            guidance_block = "No new retrieved guidance for this turn."
            retrieval_guidance = (
                "No new retrieval was performed for this turn. "
                "Continue from the conversation history and the current user message."
            )

        return (
            "You are an empathetic assistant.\n"
            f"{retrieval_guidance}\n"
            "Stay consistent with the ongoing conversation.\n"
            "Answer the current user message directly; do not repeat a previous assistant answer.\n"
            "Use prior advice as background only, and add a useful next step when the user asks a follow-up.\n"
            "Ignore forum-style references such as other posts, other answers, or illustration notes.\n"
            "When retrieved guidance is present, use it as counseling direction rather than source text.\n"
            "Answer as a direct supportive assistant, not as a forum commenter.\n"
            "Do not say you are not a professional, do not say you have heard things, and do not use casual openings like 'yeah' or 'look'.\n"
            "Be grounded, warm, practical, and concise.\n"
            "Answer in 2 to 4 complete sentences and stay under 90 words unless safety requires more.\n"
            "Do not claim to be a therapist or give diagnosis.\n"
            "Do not use titles or extra sections.\n"
            "Do not copy retrieved wording verbatim.\n\n"
            f"Conversation State:\n{conversation_state}\n\n"
            f"Conversation History:\n{chat_history}\n\n"
            f"Retrieved Document Count: {document_count}\n"
            f"Retrieved Topics: {topics_used}\n\n"
            f"Retrieval Strategy: {retrieval_strategy}\n\n"
            f"Retrieved Guidance:\n{guidance_block}\n\n"
            f"Current User Message: {user_query}\n"
            f"Predicted Emotion: {predicted_emotion}\n"
            f"Response Tone: {tone}\n\n"
            "Answer:"
        )

    def run(self, user_query: str, user_id: str, chat_id: str | None = None) -> dict:
        user_query = user_query.strip()

        if not user_query:
            raise ValueError("Message cannot be empty.")

        is_new_chat = False

        if chat_id:
            chat = self.memory_service.get_chat(chat_id=chat_id, user_id=user_id)
            if chat is None:
                chat = self.memory_service.create_chat(
                    first_message=user_query,
                    user_id=user_id,
                )
                chat_id = chat["chat_id"]
                is_new_chat = True
        else:
            chat = self.memory_service.create_chat(
                first_message=user_query,
                user_id=user_id,
            )
            chat_id = chat["chat_id"]
            is_new_chat = True

        safety_result = self.safety_service.check(user_query)
        predicted_emotion = self.emotion_service.predict_emotion(user_query)

        previous_messages = self.memory_service.get_messages(
            chat_id=chat_id,
            user_id=user_id,
            limit=CHAT_HISTORY_MESSAGE_LIMIT,
        )
        chat_history = self._format_history(previous_messages)
        conversation_state = self.memory_service.get_chat_state(
            chat_id=chat_id,
            user_id=user_id,
        )
        conversation_state_text = self._format_conversation_state(conversation_state)
        retrieval_context_summary = self._build_retrieval_context_summary(previous_messages)
        retrieval_used = False
        retrieval_strategy = "not_evaluated"
        retrieved_guidance = ""

        if safety_result["flagged"]:
            tone = "calm, supportive, safety-focused"
            answer = self.safety_service.build_safe_response(user_query)
            topics_used = []
            document_count = 0
            retrieval_strategy = "safety_skip"
            retrieved_guidance = "No new retrieved guidance for this turn."
        elif self._is_closing_message(user_query):
            tone = "warm, reassuring, closing"
            answer = self._build_closing_response(user_query)
            topics_used = []
            document_count = 0
            retrieval_strategy = "conversation_closure"
            retrieved_guidance = "No new retrieved guidance for this turn."
        else:
            should_retrieve, retrieval_strategy = self._should_use_retrieval(
                user_query=user_query,
                previous_messages=previous_messages,
            )

            if should_retrieve:
                retrieval_history = (
                    retrieval_context_summary
                    if retrieval_context_summary
                    else chat_history
                )

                context_result = self.rag_service.retrieve(
                    query=user_query,
                    chat_history=retrieval_history,
                )

                topics_used = context_result.get("topics_used", [])
                context_text = context_result.get("context_text", "")
                document_count = context_result.get("document_count", 0)
                retrieval_used = document_count > 0
            else:
                topics_used = []
                context_text = ""
                document_count = 0

            tone = self.emotion_service.choose_tone(predicted_emotion, topics_used)
            retrieved_guidance = self._build_retrieved_guidance(
                user_query=user_query,
                predicted_emotion=predicted_emotion,
                tone=tone,
                topics_used=topics_used,
                documents=context_result.get("documents", []) if should_retrieve else [],
            )

            prompt = self.build_prompt(
                user_query=user_query,
                chat_history=chat_history,
                conversation_state=conversation_state_text,
                retrieved_guidance=retrieved_guidance,
                predicted_emotion=predicted_emotion,
                tone=tone,
                topics_used=topics_used,
                document_count=document_count,
                retrieval_strategy=retrieval_strategy,
            )

            answer = self.llm_service.generate(prompt)
            answer = self._strip_answer_artifacts(answer)

            if self._needs_answer_regeneration(answer) or self._is_repetitive_answer(
                answer=answer,
                previous_messages=previous_messages,
            ):
                answer = self._fallback_answer(
                    user_query=user_query,
                    predicted_emotion=predicted_emotion,
                )

            if not answer or not answer.strip():
                answer = self._fallback_answer(
                    user_query=user_query,
                    predicted_emotion=predicted_emotion,
                )

        self.memory_service.add_message(
            chat_id=chat_id,
            user_id=user_id,
            role="user",
            content=user_query,
            emotion=predicted_emotion,
            tone=None,
        )

        self.memory_service.add_message(
            chat_id=chat_id,
            user_id=user_id,
            role="assistant",
            content=answer,
            emotion=predicted_emotion,
            tone=tone,
        )

        updated_state = self._build_updated_conversation_state(
            state=conversation_state,
            user_query=user_query,
            answer=answer,
            predicted_emotion=predicted_emotion,
            topics_used=topics_used,
            retrieval_strategy=retrieval_strategy,
        )
        self.memory_service.upsert_chat_state(
            chat_id=chat_id,
            user_id=user_id,
            state=updated_state,
        )

        final_chat = self.memory_service.get_chat(chat_id=chat_id, user_id=user_id)

        return {
            "chat_id": chat_id,
            "chat_title": final_chat["title"],
            "is_new_chat": is_new_chat,
            "answer": answer,
            "predicted_emotion": predicted_emotion,
            "tone": tone,
            "retrieved_topics": topics_used,
            "retrieved_document_count": document_count,
            "retrieval_used": retrieval_used,
            "retrieval_strategy": retrieval_strategy,
            "memory_turn_count": updated_state.get("turn_count", 0),
            "safety_flag": safety_result["flagged"],
            "safety_reason": safety_result["reason"],
        }

    def list_sessions(self, user_id: str) -> list[dict]:
        return self.memory_service.list_chats(user_id=user_id)

    def get_messages(self, chat_id: str, user_id: str) -> list[dict]:
        chat = self.memory_service.get_chat(chat_id=chat_id, user_id=user_id)
        if chat is None:
            raise ValueError("Chat session not found for this user.")

        return self.memory_service.get_messages(chat_id=chat_id, user_id=user_id)
