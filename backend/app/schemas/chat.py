from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    answer: str
    predicted_emotion: str
    tone: str
    retrieved_topics: List[str]
    retrieved_document_count: int