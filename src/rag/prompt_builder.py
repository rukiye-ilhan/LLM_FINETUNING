from __future__ import annotations

from typing import Dict, Optional


class RagPromptBuilder:
    def __init__(self):
        self.system_prompt = (
            "You are a supportive and careful AI assistant. "
            "Use the provided retrieved context as your primary grounding source. "
            "Do not invent facts that are not supported by the retrieved documents. "
            "Do not claim to be a therapist or give medical diagnosis. "
            "Respond in a calm, empathetic, practical, and concise way. "
            "If the retrieved context is only partially relevant, say so carefully and give a grounded, safe response."
        )

    def build_prompt(
        self,
        user_query: str,
        context_result: Dict,
        predicted_emotion: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> Dict[str, str]:
        context_text = context_result.get("context_text", "")
        document_count = context_result.get("document_count", 0)
        topics_used = context_result.get("topics_used", [])

        predicted_emotion = predicted_emotion or "unknown"
        tone = tone or "warm, validating, supportive"

        user_prompt = (
            f"User Query:\n{user_query}\n\n"
            f"Predicted Emotion: {predicted_emotion}\n"
            f"Response Tone: {tone}\n\n"
            f"Retrieved Context Count: {document_count}\n"
            f"Retrieved Topics: {topics_used}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            "Instructions:\n"
            "1. First briefly acknowledge the user's emotional state naturally.\n"
            "2. Match the response tone to the provided tone guidance.\n"
            "3. Then provide a grounded response using the retrieved context.\n"
            "4. Do not copy the context verbatim unless necessary.\n"
            "5. Do not go beyond the retrieved context.\n"
            "6. Focus on practical, safe, and readable guidance.\n"
            "7. If context is weak or only partially relevant, say so carefully.\n"
        )

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
        }