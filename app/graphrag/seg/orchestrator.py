"""GP-10 SEG-03:七步编辑编排(ems seg/orchestrator.py 的异步移植)。

七步:① PG 更新 ② 重切分 ③ Milvus 增删(重嵌)④ 增量重抽
      ⑤ 差量校正(Neo4j BINDS)⑥ 关键词/摘要 + bind_back 回写 ⑦ CDC 留痕
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ChunkRelation, ParentDocument
from .. import db, chunk, resolve
from ..binding import bind_back, build_binds
from ..config import get
from ..dict import load_entity_defs, load_relations
from ..embeddings import EmbeddingClient
from ..extract_entities import extract as extract_entities
from ..extract_relations import extract as extract_relations
from ..llm import LLMClient
from ..milvus_client import (
    F_CHILD_ID, F_CONTENT, F_DOC_ID, F_ENABLED, F_KB_ID, F_KG_ENTITY_IDS,
    F_PARENT_ENABLED, F_PARENT_ID, F_VECTOR, AsyncMilvusClient,
)
from ..neo4j_client import Neo4jClient


async def _get_parent(session: AsyncSession, parent_id: str) -> Optional[dict]:
    rows = await db.select_where(session, ParentDocument, parent_id=parent_id)
    return rows[0] if rows else None


async def edit_segment(session: AsyncSession, milvus: AsyncMilvusClient,
                       neo4j: Neo4jClient, parent_id: str, new_content: str,
                       edited_by: Optional[str] = None,
                       llm: Optional[LLMClient] = None,
                       embeddings: Optional[EmbeddingClient] = None) -> dict:
    llm = llm or LLMClient()
    embeddings = embeddings or EmbeddingClient()
    chunk_size = int(get("chunk.chunk_size", 512))
    overlap = float(get("chunk.overlap", 0.10))

    parent = await _get_parent(session, parent_id)
    if parent is None:
        return {"ok": False, "reason": "parent not found"}

    # ① PG 更新
    new_version = int(parent.get("content_version") or 1) + 1
    await db.update(session, ParentDocument, {
        "parent_id": parent_id, "content": new_content,
        "content_version": new_version, "edited_by": edited_by,
        "edited_at": datetime.now(timezone.utc),
    }, "parent_id")
    parent["content"] = new_content
    parent["content_version"] = new_version

    # ② 重切分
    pieces = chunk.chunk_text(new_content, chunk_size, overlap)
    old_children = await db.select_where(session, ChunkRelation, parent_id=parent_id)
    old_ids = {c["child_id"] for c in old_children}
    new_ids = {f"{parent_id}#{i}" for i in range(1, len(pieces) + 1)}

    # ③ Milvus 增删 + 重嵌
    removed = sorted(old_ids - new_ids)
    if removed:
        await milvus.delete(removed)
        for cid in removed:
            await db.delete(session, ChunkRelation, cid, "child_id")

    enabled = bool(parent.get("enabled", True))
    new_children: list[dict] = []
    for i, piece in enumerate(pieces, start=1):
        cid = f"{parent_id}#{i}"
        child = {"child_id": cid, "parent_id": parent_id,
                 "chunk_index": i, "content": piece, "enabled": enabled}
        await db.upsert(session, ChunkRelation, child, "child_id")
        new_children.append(child)
        vec = await asyncio.to_thread(embeddings.embed_one, piece)
        await milvus.upsert([{
            F_CHILD_ID: cid, F_CONTENT: piece, F_PARENT_ID: parent_id,
            F_DOC_ID: parent.get("doc_id", ""),
            F_KB_ID: str(parent["kb_id"]) if parent.get("kb_id") else "",
            F_ENABLED: enabled, F_PARENT_ENABLED: enabled,
            F_KG_ENTITY_IDS: [], F_VECTOR: vec,
        }])

    # ④ 增量重抽(仅该父块)
    entity_defs = await load_entity_defs(session)
    entity_types = [d["name"] for d in entity_defs]
    relation_defs = await load_relations(session)
    entities = await extract_entities(session, llm, [parent], entity_types, entity_defs)
    relations, _ = await extract_relations(session, entities, llm, [parent], relation_defs)
    # ④.5 跨文档消歧:对齐到全局注册表 + 写回(编辑重抽复用既有实体)
    entities, relations, _remap = await resolve.align(session, entities, relations,
                                                      milvus, embeddings, llm)
    await resolve.persist(session, entities, milvus, embeddings)

    # ⑤ 差量校正(Neo4j 节点 + BINDS)
    for e in entities:
        await neo4j.merge_entity(e.to_dict())
    for r in relations:
        await neo4j.merge_relation(r.to_dict())
    target_binds = {e.entity_id: e for e in entities if e.source_parent_id == parent_id}
    current_binds = set(await neo4j.entity_ids_of_parent(parent_id))
    for eid in current_binds - set(target_binds):
        await neo4j.delete_bind(eid, parent_id)
    for eid, e in target_binds.items():
        await neo4j.merge_binds(eid, parent_id, "primary", e.confidence)

    # ⑥ 回写:PG kg_*(bind_back)+ 关键词/摘要;无绑定则清空 kg_*
    binds = build_binds(entities)
    await bind_back(session, milvus, binds, [parent], new_children)
    await db.update(session, ParentDocument, {
        "parent_id": parent_id,
        "keywords": "、".join(e.name for e in entities),
        "summary": new_content[:200],
    }, "parent_id")
    if not target_binds:
        await db.update(session, ParentDocument, {
            "parent_id": parent_id, "kg_entity_id": "",
            "kg_role": None, "kg_confidence": None,
        }, "parent_id")

    # ⑦ CDC 留痕(图 Segment 保持同步)
    from ..cdc.consumer import handle_event
    await handle_event({"op": "u", "before": {}, "after": parent}, neo4j)

    return {
        "ok": True, "content_version": new_version,
        "removed_chunks": len(removed), "new_chunks": len(pieces),
        "entities": len(entities), "relations": len(relations),
        "primary_binds": len(target_binds),
    }
