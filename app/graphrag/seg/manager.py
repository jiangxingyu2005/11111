"""GP-10 SEG-02:分段级 + 文档级开关(ems seg/manager.py 的异步移植)。

幂等:enabled 状态不变时不产生冗余事件(文档级开关 enabled 相同时跳过该分段)。
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ChunkRelation, ParentDocument
from .. import db
from ..milvus_client import F_ENABLED, AsyncMilvusClient
from ..neo4j_client import Neo4jClient


async def _get_parent(session: AsyncSession, parent_id: str) -> Optional[dict]:
    rows = await db.select_where(session, ParentDocument, parent_id=parent_id)
    return rows[0] if rows else None


async def _apply_segment_enabled(session: AsyncSession, milvus: AsyncMilvusClient,
                                 neo4j: Neo4jClient, parent: dict, enabled: bool) -> dict:
    """把父块 enabled 状态传播到 PG 子块 / Milvus / Neo4j。"""
    children = await db.select_where(session, ChunkRelation, parent_id=parent["parent_id"])
    for c in children:
        await db.update(session, ChunkRelation,
                        {"child_id": c["child_id"], "enabled": enabled}, "child_id")
        await milvus.update_fields(c["child_id"], **{F_ENABLED: enabled})
    await neo4j.merge_segment({
        "parent_id": parent["parent_id"], "doc_id": parent["doc_id"],
        "kb_id": parent.get("kb_id"), "title": parent.get("title", ""),
        "enabled": enabled, "chapter_no": parent.get("chapter_no", ""),
    })
    return {"parent_id": parent["parent_id"], "enabled": enabled, "children": len(children)}


async def set_segment_enabled(session: AsyncSession, milvus: AsyncMilvusClient,
                              neo4j: Neo4jClient, parent_id: str, enabled: bool) -> dict:
    """SEG-02:单个分段开关(幂等)。"""
    parent = await _get_parent(session, parent_id)
    if parent is None:
        return {"ok": False, "reason": "parent not found"}
    if bool(parent.get("enabled")) == bool(enabled):
        return {"ok": True, "changed": False, "reason": "no-op"}   # 不产生冗余事件
    await db.update(session, ParentDocument,
                    {"parent_id": parent_id, "enabled": enabled, "segment_override": True},
                    "parent_id")
    parent["enabled"] = enabled
    r = await _apply_segment_enabled(session, milvus, neo4j, parent, enabled)
    return {"ok": True, "changed": True, **r}


async def set_document_enabled(session: AsyncSession, milvus: AsyncMilvusClient,
                               neo4j: Neo4jClient, doc_id: str, enabled: bool) -> dict:
    """SEG-02-D:文档级总开关(幂等,enabled 相同跳过分段,不产生冗余事件)。"""
    parents = await db.select_where(session, ParentDocument, doc_id=doc_id)
    if not parents:
        return {"ok": False, "reason": "document not found"}
    changed = 0
    for p in parents:
        if bool(p.get("enabled")) == bool(enabled):
            continue
        await db.update(session, ParentDocument,
                        {"parent_id": p["parent_id"], "enabled": enabled, "segment_override": True},
                        "parent_id")
        p["enabled"] = enabled
        await _apply_segment_enabled(session, milvus, neo4j, p, enabled)
        changed += 1
    return {"ok": True, "changed": changed, "total_segments": len(parents)}
