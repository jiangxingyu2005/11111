"""会话管理接口（独立 CRUD，按 user_id 隔离）。

前端传身份 token（Authorization: Bearer <token>）→ resolve_user_id → user_id，
再按 (user_id, session_id) 隔离会话。身份解析复用 app/auth.py（与记忆层同源，
由 MEMORY_AUTH_MODE 决定 passthrough / jwt）。

与 POST /api/retrieval 的关系：
  - 会话接口创建出的 session_id，供问答接口请求体里的 session_id 复用；
  - 问答接口的记忆自动写回受 MEMORY_ENABLED 控制，本会话接口独立于该开关、始终可用。
身份校验 fail-closed：无 token / 解析失败 → 401（会话管理必须有身份才能隔离，不能静默）。
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ..auth import resolve_user_id
from ..memory.session_store import ConversationExistsError, get_session_store

log = logging.getLogger("retrieval.api.session")
router = APIRouter()

SESSION_ID_MAX = 128


class SessionCreate(BaseModel):
    session_id: str | None = Field(default=None, max_length=SESSION_ID_MAX)
    title: str = Field(default="", max_length=200)


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _user_id(authorization: str | None = Header(default=None)) -> str:
    """从 Authorization: Bearer <token> 解析 user_id；失败 401。"""
    token = None
    if authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    uid = resolve_user_id(token)
    if not uid:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized",
                    "message": "缺少或无效的身份 token（Authorization: Bearer <token>）"},
        )
    return uid


@router.post("/api/sessions", status_code=201)
async def create_session(body: SessionCreate, user_id: str = Depends(_user_id)) -> dict:
    """创建会话。session_id 缺省时服务端生成并返回，前端据此在问答接口复用。"""
    store = get_session_store()
    sid = (body.session_id or "").strip() or uuid.uuid4().hex
    try:
        conv = await store.create_conversation(user_id, sid, body.title.strip())
    except ConversationExistsError:
        raise HTTPException(
            status_code=409,
            detail={"code": "session_exists", "message": f"会话 {sid} 已存在"},
        )
    return {"session_id": sid, "title": body.title.strip(),
            "created_at": conv.get("created_at"), "updated_at": conv.get("updated_at")}


@router.get("/api/sessions")
async def list_sessions(user_id: str = Depends(_user_id)) -> dict:
    """列出当前用户全部会话（按最近更新倒序，含消息数）。"""
    sessions = await get_session_store().list_conversations(user_id)
    return {"sessions": sessions}


@router.get("/api/sessions/{session_id}/messages")
async def session_messages(
    session_id: str = Path(max_length=SESSION_ID_MAX),
    limit: int = Query(default=200, ge=1, le=1000),
    user_id: str = Depends(_user_id),
) -> dict:
    """查看某会话的消息历史（时间正序）。"""
    msgs = await get_session_store().list_messages(user_id, session_id, limit)
    return {"session_id": session_id, "messages": msgs}


@router.patch("/api/sessions/{session_id}")
async def rename_session(
    body: SessionRename,
    session_id: str = Path(max_length=SESSION_ID_MAX),
    user_id: str = Depends(_user_id),
) -> dict:
    """重命名会话（更新标题）。"""
    ok = await get_session_store().update_title(user_id, session_id, body.title.strip())
    if not ok:
        raise HTTPException(status_code=404, detail={
            "code": "session_not_found", "message": f"会话 {session_id} 不存在"})
    return {"session_id": session_id, "title": body.title.strip()}


@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str = Path(max_length=SESSION_ID_MAX),
    user_id: str = Depends(_user_id),
) -> dict:
    """删除会话（级联删除其消息）。"""
    ok = await get_session_store().delete_conversation(user_id, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail={
            "code": "session_not_found", "message": f"会话 {session_id} 不存在"})
    return {"deleted": True, "session_id": session_id}
