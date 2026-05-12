from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.services.auth_service import AuthService

security = HTTPBearer()
auth_service = AuthService()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials

    try:
        return auth_service.verify_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
