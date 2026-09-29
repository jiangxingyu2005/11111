"""记忆编排入口：检索主链只调这两个函数（load_context 读 / commit_turn 写）。

- load_context：读侧，返回 {summary, window, memories}，任一失败返回空（fail-closed）。
- commit_turn：写侧，append 本轮 -> maybe_compress，由调用方 fire-and-forget 后台跑。
"""
from __future__ import annotations

import asyncio
import logging

from ..config import effective_cfg, get_settings
from .backend import get_memory_backend
from .compressor import estimate_tokens, maybe_compress
from .session_store import get_session_store

log = logging.getLogger("retrieval.memory.service")

# 强引用后台任务，防被 GC；任务结束自动移除
_background: set[asyncio.Task] = set()


def spawn_background(coro) -> None:
    """fire-and-forget：不阻塞 SSE 响应，失败由任务自身 fail-closed。"""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def _enabled() -> bool:
    return get_settings().memory_enabled


def _cfg() -> dict:
    return (effective_cfg().get("memory") or {}) or {}


async def load_context(user_id: str, session_id: str, query: str) -> dict:
    """读侧：装载三层上下文（摘要 + 窗口 + 长期事实）。任一失败记 warning、该层留空。"""
    ctx: dict = {"summary": "", "window": [], "memories": []}
    if not _enabled() or not user_id:
        return ctx
    cfg = _cfg()
    store = get_session_store()
    sid = session_id or "default"
    try:
        conv = await store.get_or_create_conversation(user_id, sid)
        ctx["summary"] = conv.get("summary", "")
        ctx["window"] = await store.load_window(conv["id"], int(cfg.get("window_size", 10)))
    except Exception as e:
        log.warning("load session context failed (窗口/摘要留空): %s", e)
    try:
        ctx["memories"] = await get_memory_backend().search(query, user_id, k=3)
    except Exception as e:
        log.warning("load long-term memory failed (memories 留空): %s", e)
    return ctx


async def commit_turn(user_id: str, session_id: str, query: str, answer: str) -> None:
    """写侧：append 本轮（user+assistant）-> maybe_compress。由 spawn_background 调用。"""
    if not _enabled() or not user_id:
        return
    cfg = _cfg()
    store = get_session_store()
    sid = session_id or "default"
    try:
        conv = await store.get_or_create_conversation(user_id, sid)
        await store.append_message(conv["id"], "user", query, estimate_tokens(query))
        if answer:
            await store.append_message(conv["id"], "assistant", answer, estimate_tokens(answer))
        await maybe_compress(store, get_memory_backend(), conv, cfg)
    except Exception as e:
        log.warning("commit turn failed (fail-closed): %s", e)
