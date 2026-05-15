from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional


class ChatMemoryService:
    def __init__(self, db_path: str = "data/chat_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    tone TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_sessions(chat_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    main_issue TEXT,
                    current_emotion TEXT,
                    user_goal TEXT,
                    previous_advice TEXT,
                    open_questions TEXT,
                    last_retrieved_topics TEXT,
                    summary TEXT,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_sessions(chat_id)
                )
                """
            )

            conn.commit()

    def _migrate_db(self) -> None:
        with self._connect() as conn:
            session_columns = conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            session_column_names = {column["name"] for column in session_columns}

            if "user_id" not in session_column_names:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT")

            message_columns = conn.execute("PRAGMA table_info(chat_messages)").fetchall()
            message_column_names = {column["name"] for column in message_columns}

            if "user_id" not in message_column_names:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN user_id TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    main_issue TEXT,
                    current_emotion TEXT,
                    user_goal TEXT,
                    previous_advice TEXT,
                    open_questions TEXT,
                    last_retrieved_topics TEXT,
                    summary TEXT,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chat_sessions(chat_id)
                )
                """
            )

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

    def create_chat(self, first_message: str, user_id: str) -> Dict:
        chat_id = str(uuid.uuid4())
        now = self._now()
        title = self.generate_title(first_message)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (chat_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, title, now, now),
            )
            conn.execute(
                """
                INSERT INTO chat_state
                (
                    chat_id,
                    user_id,
                    previous_advice,
                    open_questions,
                    last_retrieved_topics,
                    turn_count,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, "[]", "[]", "[]", 0, now),
            )
            conn.commit()

        return {
            "chat_id": chat_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def get_chat(self, chat_id: str, user_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM chat_sessions
                WHERE chat_id = ? AND user_id = ?
                """,
                (chat_id, user_id),
            ).fetchone()

        return dict(row) if row else None

    def touch_chat(self, chat_id: str, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE chat_id = ? AND user_id = ?
                """,
                (self._now(), chat_id, user_id),
            )
            conn.commit()

    def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> None:
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)

        if chat is None:
            raise ValueError("Chat session not found for this user.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages
                (chat_id, user_id, role, content, emotion, tone, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, role, content, emotion, tone, self._now()),
            )
            conn.commit()

        self.touch_chat(chat_id=chat_id, user_id=user_id)

    def get_messages(self, chat_id: str, user_id: str, limit: int = 50) -> List[Dict]:
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)

        if chat is None:
            raise ValueError("Chat session not found for this user.")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, emotion, tone, created_at
                FROM chat_messages
                WHERE chat_id = ? AND user_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (chat_id, user_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def _decode_list(self, value: Optional[str]) -> List[str]:
        if not value:
            return []

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []

        if not isinstance(decoded, list):
            return []

        return [str(item) for item in decoded if str(item).strip()]

    def get_chat_state(self, chat_id: str, user_id: str) -> Dict:
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)

        if chat is None:
            raise ValueError("Chat session not found for this user.")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM chat_state
                WHERE chat_id = ? AND user_id = ?
                """,
                (chat_id, user_id),
            ).fetchone()

        if row is None:
            return {
                "main_issue": "",
                "current_emotion": "",
                "user_goal": "",
                "previous_advice": [],
                "open_questions": [],
                "last_retrieved_topics": [],
                "summary": "",
                "turn_count": 0,
            }

        state = dict(row)
        state["previous_advice"] = self._decode_list(state.get("previous_advice"))
        state["open_questions"] = self._decode_list(state.get("open_questions"))
        state["last_retrieved_topics"] = self._decode_list(state.get("last_retrieved_topics"))
        return state

    def upsert_chat_state(self, chat_id: str, user_id: str, state: Dict) -> None:
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)

        if chat is None:
            raise ValueError("Chat session not found for this user.")

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_state
                (
                    chat_id,
                    user_id,
                    main_issue,
                    current_emotion,
                    user_goal,
                    previous_advice,
                    open_questions,
                    last_retrieved_topics,
                    summary,
                    turn_count,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    main_issue = excluded.main_issue,
                    current_emotion = excluded.current_emotion,
                    user_goal = excluded.user_goal,
                    previous_advice = excluded.previous_advice,
                    open_questions = excluded.open_questions,
                    last_retrieved_topics = excluded.last_retrieved_topics,
                    summary = excluded.summary,
                    turn_count = excluded.turn_count,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    user_id,
                    state.get("main_issue", ""),
                    state.get("current_emotion", ""),
                    state.get("user_goal", ""),
                    json.dumps(state.get("previous_advice", []), ensure_ascii=False),
                    json.dumps(state.get("open_questions", []), ensure_ascii=False),
                    json.dumps(state.get("last_retrieved_topics", []), ensure_ascii=False),
                    state.get("summary", ""),
                    int(state.get("turn_count", 0)),
                    now,
                ),
            )
            conn.commit()

    def list_chats(self, user_id: str, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chat_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]
