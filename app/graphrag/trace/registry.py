"""GP-09:溯源数据查询(Plan §3.11,async 移植)。

文件列表(含文件级角标数)、详情(召回历史时间线)、源文件定位。
文件级召回计数口径:source_file_registry.recall_count(按查询去重,见 trace/hooks)。
"""
from __future__ import annotations


from sqlalchemy.ext.asyncio import AsyncSession

from ...models import RecallLog, SourceFileRegistry
from .. import db


def _iso(v) -> str | None:
    """datetime -> ISO 字符串(供 JSON 序列化)。"""
    return v.isoformat() if hasattr(v, "isoformat") else v


async def list_files(session: AsyncSession, kb_id=None) -> list[dict]:
    filters = {}
    if kb_id is not None:
        filters["kb_id"] = kb_id
    rows = await db.select_where(session, SourceFileRegistry, **filters)
    out = []
    for f in rows:
        out.append({
            "doc_id": f["doc_id"],
            "file_name": f["file_name"],
            "doc_type": f.get("doc_type"),
            "event_time": f.get("event_time"),
            "uploaded_at": _iso(f.get("uploaded_at")),
            "recall_count": int(f.get("recall_count") or 0),
        })
    return out


async def file_detail(session: AsyncSession, doc_id: str) -> dict:
    rows = await db.select_where(session, SourceFileRegistry, doc_id=doc_id)
    if not rows:
        return {"doc_id": doc_id, "found": False}
    f = rows[0]
    history = await db.select_where(session, RecallLog, doc_id=doc_id)
    return {
        "doc_id": doc_id,
        "file_name": f["file_name"],
        "found": True,
        "recall_count": int(f.get("recall_count") or 0),
        "history": [
            {"query": l["query"], "child_id": l["child_id"],
             "parent_id": l["parent_id"], "ts": _iso(l.get("ts"))}
            for l in history
        ],
    }


async def locate(session: AsyncSession, doc_id: str) -> dict:
    rows = await db.select_where(session, SourceFileRegistry, doc_id=doc_id)
    if not rows:
        return {"doc_id": doc_id, "found": False, "source_path": None}
    return {"doc_id": doc_id, "found": True, "source_path": rows[0].get("source_path")}
