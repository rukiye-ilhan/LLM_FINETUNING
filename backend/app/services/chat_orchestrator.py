from __future__ import annotations

from backend.app.services.emotion_service import EmotionService
from backend.app.services.rag_service import RagService
from backend.app.services.llm_service import LLMService
from backend.app.services.chat_memory_service import ChatMemoryService
from backend.app.services.safety_service import SafetyService


class ChatOrchestrator:
    def __init__(self):
        self.emotion_service = EmotionService()
        self.rag_service = RagService()
        self.llm_service = LLMService()
        self.memory_service = ChatMemoryService()
        self.safety_service = SafetyService()

    def _format_history(self, messages: list[dict]) -> str:
        if not messages:
            return "No previous conversation."

        lines = []
        for msg in messages[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"].strip()
            if content:
                lines.append(f"{role}: {content}")

        if not lines:
            return "No previous conversation."

        return "\n".join(lines)

    def _build_retrieval_context_summary(self, messages: list[dict]) -> str:
        """
        Retrieval için full history değil, kısa ve anlamlı user-side bağlam çıkar.
        """
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

    def _needs_answer_regeneration(self, answer: str) -> bool:
        if not answer or not answer.strip():
            return True

        lowered = answer.lower()

        banned_patterns = [
            "the other post answers",
            "the other answers",
            "see below",
            "illustration below",
            "working with me you will learn",
            "9 ways",
            "other post",
            "other answers",
        ]

        return any(pattern in lowered for pattern in banned_patterns)

    def _fallback_answer(
        self,
        user_query: str,
        predicted_emotion: str,
    ) -> str:
        if predicted_emotion == "anxiety":
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
        context_text: str,
        predicted_emotion: str,
        tone: str,
        topics_used: list[str],
        document_count: int,
    ) -> str:
        return (
            "You are an empathetic assistant.\n"
            "Use the retrieved context when it is relevant.\n"
            "Stay consistent with the ongoing conversation.\n"
            "Ignore forum-style references such as other posts, other answers, or illustration notes.\n"
            "Rewrite the retrieved content naturally for the current user.\n"
            "If the retrieved text contains thread artifacts, do not repeat them.\n"
            "Be grounded, warm, practical, and concise.\n"
            "Do not claim to be a therapist or give diagnosis.\n"
            "Do not use titles or extra sections.\n"
            "Do not copy the context verbatim.\n\n"
            f"Conversation History:\n{chat_history}\n\n"
            f"Retrieved Context Count: {document_count}\n"
            f"Retrieved Topics: {topics_used}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"Current User Message: {user_query}\n"
            f"Predicted Emotion: {predicted_emotion}\n"
            f"Response Tone: {tone}\n\n"
            "Answer:"
        )

    def run(self, user_query: str, chat_id: str | None = None) -> dict:
        user_query = user_query.strip()

        if not user_query:
            raise ValueError("Message cannot be empty.")

        is_new_chat = False

        if chat_id:
            chat = self.memory_service.get_chat(chat_id)
            if chat is None:
                chat = self.memory_service.create_chat(user_query)
                chat_id = chat["chat_id"]
                is_new_chat = True
        else:
            chat = self.memory_service.create_chat(user_query)
            chat_id = chat["chat_id"]
            is_new_chat = True

        safety_result = self.safety_service.check(user_query)
        predicted_emotion = self.emotion_service.predict_emotion(user_query)

        previous_messages = self.memory_service.get_messages(chat_id, limit=20)
        chat_history = self._format_history(previous_messages)
        retrieval_context_summary = self._build_retrieval_context_summary(previous_messages)

        if safety_result["flagged"]:
            tone = "calm, supportive, safety-focused"
            answer = self.safety_service.build_safe_response(user_query)
            topics_used = []
            document_count = 0
        else:
            retrieval_history = retrieval_context_summary if retrieval_context_summary else chat_history

            context_result = self.rag_service.retrieve(
                query=user_query,
                chat_history=retrieval_history,
            )

            topics_used = context_result.get("topics_used", [])
            context_text = context_result.get("context_text", "")
            document_count = context_result.get("document_count", 0)

            tone = self.emotion_service.choose_tone(predicted_emotion, topics_used)

            prompt = self.build_prompt(
                user_query=user_query,
                chat_history=chat_history,
                context_text=context_text,
                predicted_emotion=predicted_emotion,
                tone=tone,
                topics_used=topics_used,
                document_count=document_count,
            )

            answer = self.llm_service.generate(prompt)

            if self._needs_answer_regeneration(answer):
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
            role="user",
            content=user_query,
            emotion=predicted_emotion,
            tone=None,
        )

        self.memory_service.add_message(
            chat_id=chat_id,
            role="assistant",
            content=answer,
            emotion=predicted_emotion,
            tone=tone,
        )

        final_chat = self.memory_service.get_chat(chat_id)

        return {
            "chat_id": chat_id,
            "chat_title": final_chat["title"],
            "is_new_chat": is_new_chat,
            "answer": answer,
            "predicted_emotion": predicted_emotion,
            "tone": tone,
            "retrieved_topics": topics_used,
            "retrieved_document_count": document_count,
            "safety_flag": safety_result["flagged"],
            "safety_reason": safety_result["reason"],
        }

    def list_sessions(self) -> list[dict]:
        return self.memory_service.list_chats()

    def get_messages(self, chat_id: str) -> list[dict]:
        chat = self.memory_service.get_chat(chat_id)
        if chat is None:
            raise ValueError("Chat session not found.")
        return self.memory_service.get_messages(chat_id)