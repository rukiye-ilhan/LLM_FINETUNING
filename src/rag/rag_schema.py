from dataclasses import dataclass
from typing import Optional


@dataclass
class RagRawRecord:
    question_id: int
    question_title: str
    question_text: Optional[str]
    question_link: str
    topic: str
    therapist_info: Optional[str]
    therapist_url: Optional[str]
    answer_text: Optional[str]
    upvotes: int
    views: int


@dataclass
class RagProcessedRecord:
    doc_id: str
    question_id: int
    question_title: str
    question_text: str
    answer_text: str
    topic: str
    upvotes: int
    views: int
    answer_length: int
    quality_score: float
    rag_document: str