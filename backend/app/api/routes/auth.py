from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends

from backend.app.api.dependencies.auth import get_current_user
from backend.app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)
from backend.app.services.auth_service import AuthService

router = APIRouter(prefix="/auth")
auth_service = AuthService()


@router.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    try:
        result = auth_service.register(
            name=request.name,
            email=request.email,
            password=request.password,
        )
        return AuthResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest):
    try:
        result = auth_service.login(
            email=request.email,
            password=request.password,
        )
        return AuthResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        user_id=current_user["user_id"],
        name=current_user["name"],
        email=current_user["email"],
    )
