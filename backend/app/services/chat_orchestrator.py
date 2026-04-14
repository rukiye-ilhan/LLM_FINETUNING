from backend.app.services.emotion_service import EmotionService
from backend.app.services.rag_service import RagService
from backend.app.services.llm_service import LLMService


class ChatOrchestrator:
    def __init__(self):
        self.emotion_service = EmotionService()
        self.rag_service = RagService()
        self.llm_service = LLMService()

    def build_prompt(
        self,
        user_query: str,
        context_text: str,
        predicted_emotion: str,
        tone: str,
        topics_used: list[str],
        document_count: int,
    ) -> str:
        return (
            "Use only the context below to answer the user.\n"
            "Be empathetic, grounded, concise, and practical.\n"
            "Do not continue with another example.\n"
            "Do not write titles, labels, or extra sections.\n"
            "Do not claim to be a therapist or give diagnosis.\n\n"
            f"Retrieved Context Count: {document_count}\n"
            f"Retrieved Topics: {topics_used}\n\n"
            f"Context:\n{context_text}\n\n"
            f"User: {user_query}\n"
            f"Emotion: {predicted_emotion}\n"
            f"Tone: {tone}\n\n"
            "Answer:"
        )

    def run(self, user_query: str) -> dict:
        predicted_emotion = self.emotion_service.predict_emotion(user_query)

        context_result = self.rag_service.retrieve(user_query)
        topics_used = context_result.get("topics_used", [])
        context_text = context_result.get("context_text", "")
        document_count = context_result.get("document_count", 0)

        tone = self.emotion_service.choose_tone(predicted_emotion, topics_used)

        prompt = self.build_prompt(
            user_query=user_query,
            context_text=context_text,
            predicted_emotion=predicted_emotion,
            tone=tone,
            topics_used=topics_used,
            document_count=document_count,
        )

        answer = self.llm_service.generate(prompt)

        return {
            "answer": answer,
            "predicted_emotion": predicted_emotion,
            "tone": tone,
            "retrieved_topics": topics_used,
            "retrieved_document_count": document_count,
        }