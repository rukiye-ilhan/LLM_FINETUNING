from fastapi import APIRouter

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_orchestrator import ChatOrchestrator

router = APIRouter()
chat_orchestrator = ChatOrchestrator()


@router.post("/chat/message", response_model=ChatResponse)
def chat_message(request: ChatRequest):
    result = chat_orchestrator.run(request.message)
    return ChatResponse(**result)