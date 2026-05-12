from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ChatMemoryService:
    def __init__(self, db_path: str = "data/chat_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    tone TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_sessions(chat_id)
                )
            """)
            conn.commit()

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def generate_title(self, first_message: str) -> str:
        text = " ".join(first_message.strip().split())
        words = text.split()[:8]
        title = " ".join(words).strip()
        if not title:
            return "New Chat"
        if len(title) > 60:
            title = title[:57] + "..."
        return title

    def create_chat(self, first_message: str) -> Dict:
        chat_id = str(uuid.uuid4())
        now = self._now()
        title = self.generate_title(first_message)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (chat_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, title, now, now),
            )
            conn.commit()

        return {
            "chat_id": chat_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def touch_chat(self, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE chat_id = ?",
                (self._now(), chat_id),
            )
            conn.commit()

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (chat_id, role, content, emotion, tone, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, role, content, emotion, tone, self._now()),
            )
            conn.commit()

        self.touch_chat(chat_id)

    def get_messages(self, chat_id: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, emotion, tone, created_at
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()

        return [dict(r) for r in rows]

    def list_chats(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]