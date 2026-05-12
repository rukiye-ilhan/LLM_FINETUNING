from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


class AuthService:
    """
    Lightweight auth service for local/demo usage.

    - Stores users in SQLite.
    - Hashes passwords with PBKDF2-HMAC-SHA256.
    - Issues signed bearer tokens without adding extra dependencies.
    """

    def __init__(self, db_path: str = "data/chat_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_key = os.getenv("EMPARAG_AUTH_SECRET", "change-this-dev-secret")
        self.token_expire_minutes = int(os.getenv("EMPARAG_TOKEN_EXPIRE_MINUTES", "1440"))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _hash_password(self, password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            150_000,
        )
        return base64.urlsafe_b64encode(digest).decode("utf-8")

    def _verify_password(self, password: str, salt: str, expected_hash: str) -> bool:
        actual_hash = self._hash_password(password, salt)
        return hmac.compare_digest(actual_hash, expected_hash)

    def _b64(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _unb64(self, value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    def _sign(self, payload_b64: str) -> str:
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return self._b64(signature)

    def _create_token(self, user_id: str, email: str) -> str:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.token_expire_minutes)
        payload = {
            "sub": user_id,
            "email": email,
            "exp": int(expires_at.timestamp()),
        }
        payload_b64 = self._b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signature_b64 = self._sign(payload_b64)
        return f"{payload_b64}.{signature_b64}"

    def _get_user_by_email(self, email: str) -> Optional[dict]:
        normalized_email = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, name, email, created_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def register(self, name: str, email: str, password: str) -> dict:
        normalized_email = email.strip().lower()

        if self._get_user_by_email(normalized_email):
            raise ValueError("This email is already registered.")

        salt = secrets.token_urlsafe(24)
        password_hash = self._hash_password(password, salt)
        user_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, name, email, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, name.strip(), normalized_email, password_hash, salt, self._now()),
            )
            conn.commit()

        user = {
            "user_id": user_id,
            "name": name.strip(),
            "email": normalized_email,
        }
        return {
            "access_token": self._create_token(user_id, normalized_email),
            "token_type": "bearer",
            "user": user,
        }

    def login(self, email: str, password: str) -> dict:
        normalized_email = email.strip().lower()
        user = self._get_user_by_email(normalized_email)

        if not user or not self._verify_password(password, user["salt"], user["password_hash"]):
            raise ValueError("Invalid email or password.")

        user_response = {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
        }
        return {
            "access_token": self._create_token(user["user_id"], user["email"]),
            "token_type": "bearer",
            "user": user_response,
        }

    def verify_token(self, token: str) -> dict:
        try:
            payload_b64, signature_b64 = token.split(".", 1)
        except ValueError:
            raise ValueError("Invalid token format.")

        expected_signature = self._sign(payload_b64)
        if not hmac.compare_digest(signature_b64, expected_signature):
            raise ValueError("Invalid token signature.")

        payload = json.loads(self._unb64(payload_b64).decode("utf-8"))
        exp = int(payload.get("exp", 0))

        if exp < int(datetime.now(timezone.utc).timestamp()):
            raise ValueError("Token expired.")

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Token missing subject.")

        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found.")

        return user
