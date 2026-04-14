from typing import Dict, List


EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anxiety": [
        "anxious", "anxiety", "overwhelmed", "panic", "worried",
        "worry", "nervous", "stress", "stressed", "fear", "afraid"
    ],
    "distress": [
        "sad", "hopeless", "empty", "down", "lonely", "crying",
        "worthless", "depressed", "hurt", "upset"
    ],
    "anger": [
        "angry", "furious", "annoyed", "mad", "resentful", "irritated"
    ],
    "self-esteem": [
        "not good enough", "worthless", "insecure", "not enough",
        "self-doubt", "hate myself"
    ],
}


EMOTION_TO_TONE = {
    "anxiety": "calm, reassuring, grounded",
    "distress": "warm, validating, gentle",
    "anger": "calm, respectful, de-escalating",
    "self-esteem": "affirming, non-judgmental, supportive",
    "mixed": "warm, validating, supportive",
}


class EmotionService:
    def predict_emotion(self, text: str) -> str:
        q = text.lower()

        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in q:
                    return emotion

        return "mixed"

    def choose_tone(self, predicted_emotion: str, retrieved_topics: list[str]) -> str:
        if predicted_emotion in EMOTION_TO_TONE:
            return EMOTION_TO_TONE[predicted_emotion]

        return "warm, validating, supportive"