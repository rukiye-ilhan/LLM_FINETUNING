from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    chat_id: str
    chat_title: str
    is_new_chat: bool

    answer: str
    predicted_emotion: str
    tone: str
    retrieved_topics: List[str]
    retrieved_document_count: int

    safety_flag: bool
    safety_reason: Optional[str] = None


class ChatSessionItem(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessageItem(BaseModel):
    role: str
    content: str
    emotion: Optional[str] = None
    tone: Optional[str] = None
    created_at: str