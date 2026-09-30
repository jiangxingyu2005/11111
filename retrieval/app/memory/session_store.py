"""会话/消息持久化：real=asyncpg(PG conversations/messages)，fake=内存 dict。

会话 = user_id + session_id 定位一行；消息 = 窗口里的原文轮次（role/content/token_count）。
对齐 store/fake.py 的 fake/real 分发 + fuse/rules.py 的 asyncpg.connect 连接方式。

retrieval 由「三库只读」变为对这两张表读写（IF-9 之外的新增写路径，见 plan 风险 2）。
会话管理接口（app/api/session.py）复用本 store 做独立 CRUD，与记忆读写共用同一张表，
按 (user_id, session_id) 隔离；会话接口不依赖 MEMORY_ENABLED，独立可用。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ..config import get_settings

log = logging.getLogger("retrieval.memory.session_store")


class ConversationExistsError(Exception):
    """创建会话时 (user_id, session_id) 已存在。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@runtime_checkable
class SessionStore(Protocol):
    # —— 问答记忆读写侧（app/memory/service.py 调用）——
    async def get_or_create_conversation(self, user_id: str, session_id: str) -> dict: ...
    async def append_message(self, conv_id: str, role: str, content: str, token_count: int) -> None: ...
    async def load_window(self, conv_id: str, limit: int) -> list[dict]: ...
    async def update_summary(self, conv_id: str, summary: str, compress_count: int) -> None: ...
    async def trim_messages(self, conv_id: str, keep_last: int) -> None: ...

    # —— 会话管理接口侧（app/api/session.py 调用，独立 CRUD）——
    async def create_conversation(self, user_id: str, session_id: str, title: str = "") -> dict: ...
    async def list_conversations(self, user_id: str) -> list[dict]: ...
    async def list_messages(self, user_id: str, session_id: str, limit: int) -> list[dict]: ...
    async def delete_conversation(self, user_id: str, session_id: str) -> bool: ...
    async def update_title(self, user_id: str, session_id: str, title: str) -> bool: ...


class FakeSessionStore:
    """内存会话/消息（fake 后端，进程内，重启即失）。"""

    def __init__(self) -> None:
        self._convs: dict[tuple[str, str], dict] = {}
        self._msgs: dict[str, list[dict]] = {}

    async def get_or_create_conversation(self, user_id: str, session_id: str) -> dict:
        key = (user_id, session_id)
        conv = self._convs.get(key)
        if conv is None:
            conv = {
                "id": uuid.uuid4().hex,
                "user_id": user_id,
                "session_id": session_id,
                "title": "",
                "summary": "",
                "compress_count": 0,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            self._convs[key] = conv
            self._msgs[conv["id"]] = []
        return conv

    async def append_message(self, conv_id: str, role: str, content: str, token_count: int) -> None:
        self._msgs.setdefault(conv_id, []).append(
            {"role": role, "content": content, "token_count": token_count,
             "created_at": _now_iso()})

    async def load_window(self, conv_id: str, limit: int) -> list[dict]:
        return list(self._msgs.get(conv_id, [])[-limit:])

    async def update_summary(self, conv_id: str, summary: str, compress_count: int) -> None:
        for conv in self._convs.values():
            if conv["id"] == conv_id:
                conv["summary"] = summary
                conv["compress_count"] = compress_count
                return

    async def trim_messages(self, conv_id: str, keep_last: int) -> None:
        msgs = self._msgs.get(conv_id)
        if msgs:
            self._msgs[conv_id] = msgs[-keep_last:]

    # —— 会话管理 CRUD ——

    async def create_conversation(self, user_id: str, session_id: str, title: str = "") -> dict:
        key = (user_id, session_id)
        if key in self._convs:
            raise ConversationExistsError(session_id)
        conv = {
            "id": uuid.uuid4().hex, "user_id": user_id, "session_id": session_id,
            "title": title, "summary": "", "compress_count": 0,
            "created_at": _now_iso(), "updated_at": _now_iso(),
        }
        self._convs[key] = conv
        self._msgs[conv["id"]] = []
        return conv

    async def list_conversations(self, user_id: str) -> list[dict]:
        out = []
        for (uid, _), conv in self._convs.items():
            if uid == user_id:
                out.append({
                    "session_id": conv["session_id"], "title": conv.get("title", ""),
                    "message_count": len(self._msgs.get(conv["id"], [])),
                    "created_at": conv.get("created_at"), "updated_at": conv.get("updated_at"),
                })
        out.sort(key=lambda c: c["updated_at"] or "", reverse=True)
        return out

    async def list_messages(self, user_id: str, session_id: str, limit: int) -> list[dict]:
        conv = self._convs.get((user_id, session_id))
        if conv is None:
            return []
        msgs = [{"role": m["role"], "content": m["content"], "created_at": m.get("created_at")}
                for m in self._msgs.get(conv["id"], [])]
        return msgs[-limit:] if limit else msgs

    async def delete_conversation(self, user_id: str, session_id: str) -> bool:
        conv = self._convs.pop((user_id, session_id), None)
        if conv is None:
            return False
        self._msgs.pop(conv["id"], None)
        return True

    async def update_title(self, user_id: str, session_id: str, title: str) -> bool:
        conv = self._convs.get((user_id, session_id))
        if conv is None:
            return False
        conv["title"] = title
        conv["updated_at"] = _now_iso()
        return True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    compress_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, session_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
"""

# 增量迁移：CREATE TABLE IF NOT EXISTS 对已存在的旧表（无 title 列）不生效，title 靠这里补齐。
_MIGRATIONS = (
    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '';",
)


class PgSessionStore:
    """PG 会话/消息（real 后端）。"""

    async def _conn(self):
        import asyncpg

        return await asyncpg.connect(get_settings().pg_dsn)

    async def _ensure_schema(self) -> None:
        conn = await self._conn()
        try:
            await conn.execute(_SCHEMA)
            for stmt in _MIGRATIONS:
                await conn.execute(stmt)
        finally:
            await conn.close()

    async def get_or_create_conversation(self, user_id: str, session_id: str) -> dict:
        await self._ensure_schema()
        cid = uuid.uuid4()
        conn = await self._conn()
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (id, user_id, session_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, session_id)
                DO UPDATE SET updated_at = now()
                RETURNING id, summary, compress_count
                """,
                cid, user_id, session_id,
            )
            return {"id": str(row["id"]), "user_id": user_id, "session_id": session_id,
                    "summary": row["summary"] or "", "compress_count": int(row["compress_count"] or 0)}
        finally:
            await conn.close()

    async def append_message(self, conv_id: str, role: str, content: str, token_count: int) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, token_count) "
                "VALUES ($1, $2, $3, $4, $5)",
                uuid.uuid4(), conv_id, role, content, token_count,
            )
        finally:
            await conn.close()

    async def load_window(self, conv_id: str, limit: int) -> list[dict]:
        conn = await self._conn()
        try:
            rows = await conn.fetch(
                "SELECT role, content, token_count FROM messages "
                "WHERE conversation_id = $1 ORDER BY created_at DESC, id DESC LIMIT $2",
                conv_id, limit,
            )
        finally:
            await conn.close()
        return [{"role": r["role"], "content": r["content"],
                 "token_count": int(r["token_count"] or 0)} for r in reversed(rows)]

    async def update_summary(self, conv_id: str, summary: str, compress_count: int) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE conversations SET summary = $1, compress_count = $2, updated_at = now() "
                "WHERE id = $3",
                summary, compress_count, conv_id,
            )
        finally:
            await conn.close()

    async def trim_messages(self, conv_id: str, keep_last: int) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "DELETE FROM messages WHERE conversation_id = $1 AND id NOT IN ("
                "  SELECT id FROM messages WHERE conversation_id = $1 "
                "  ORDER BY created_at DESC, id DESC LIMIT $2)",
                conv_id, keep_last,
            )
        finally:
            await conn.close()

    # —— 会话管理 CRUD ——

    async def create_conversation(self, user_id: str, session_id: str, title: str = "") -> dict:
        await self._ensure_schema()
        import asyncpg
        cid = uuid.uuid4()
        conn = await self._conn()
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (id, user_id, session_id, title)
                VALUES ($1, $2, $3, $4)
                RETURNING session_id, title, created_at, updated_at
                """,
                cid, user_id, session_id, title,
            )
            return dict(row)
        except asyncpg.UniqueViolationError:
            raise ConversationExistsError(session_id)
        finally:
            await conn.close()

    async def list_conversations(self, user_id: str) -> list[dict]:
        await self._ensure_schema()
        conn = await self._conn()
        try:
            rows = await conn.fetch(
                """
                SELECT c.session_id, c.title,
                       (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id)
                           AS message_count,
                       c.created_at, c.updated_at
                FROM conversations c
                WHERE c.user_id = $1
                ORDER BY c.updated_at DESC
                """,
                user_id,
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def list_messages(self, user_id: str, session_id: str, limit: int) -> list[dict]:
        await self._ensure_schema()
        conn = await self._conn()
        try:
            rows = await conn.fetch(
                """
                SELECT m.role, m.content, m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = $1 AND c.session_id = $2
                ORDER BY m.created_at ASC, m.id ASC
                LIMIT $3
                """,
                user_id, session_id, limit,
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def delete_conversation(self, user_id: str, session_id: str) -> bool:
        await self._ensure_schema()
        conn = await self._conn()
        try:
            status = await conn.execute(
                "DELETE FROM conversations WHERE user_id = $1 AND session_id = $2",
                user_id, session_id,
            )
            return status == "DELETE 1"
        finally:
            await conn.close()

    async def update_title(self, user_id: str, session_id: str, title: str) -> bool:
        await self._ensure_schema()
        conn = await self._conn()
        try:
            status = await conn.execute(
                "UPDATE conversations SET title = $3, updated_at = now() "
                "WHERE user_id = $1 AND session_id = $2",
                user_id, session_id, title,
            )
            return status == "UPDATE 1"
        finally:
            await conn.close()


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = FakeSessionStore() if get_settings().retrieval_backend == "fake" else PgSessionStore()
    return _store


def reset_session_store() -> None:
    global _store
    _store = None
