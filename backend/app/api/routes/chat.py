from fastapi import APIRouter, HTTPException

from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionItem,
    ChatMessageItem,
)
from backend.app.services.chat_orchestrator import ChatOrchestrator

router = APIRouter()
chat_orchestrator = ChatOrchestrator()


@router.post("/chat/message", response_model=ChatResponse)
def chat_message(request: ChatRequest):
    try:
        result = chat_orchestrator.run(
            user_query=request.message,
            chat_id=request.chat_id,
        )
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/chat/sessions", response_model=list[ChatSessionItem])
def get_chat_sessions():
    try:
        sessions = chat_orchestrator.list_sessions()
        return [ChatSessionItem(**item) for item in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load chat sessions: {str(e)}")


@router.get("/chat/{chat_id}/messages", response_model=list[ChatMessageItem])
def get_chat_messages(chat_id: str):
    try:
        messages = chat_orchestrator.get_messages(chat_id)
        return [ChatMessageItem(**item) for item in messages]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load chat messages: {str(e)}")