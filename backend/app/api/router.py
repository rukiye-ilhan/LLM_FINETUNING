from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.auth import router as auth_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(chat_router, tags=["chat"])