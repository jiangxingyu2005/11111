"""GP-03 EMB:向量嵌入与 Milvus 写入(ems emb/embed_upsert.py 移植,async + 直对象)。

子块批量嵌入 -> Milvus upsert,初始值:kg_entity_ids=[]、enabled=true、parent_enabled 随父块。
upsert 幂等:重复执行不产生重复行。kb_id 随父块写进子块字段。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .config import get
from .embeddings import EmbeddingClient
from .milvus_client import (
    AsyncMilvusClient, F_CHILD_ID, F_CONTENT, F_DOC_ID, F_ENABLED, F_KB_ID,
    F_KG_ENTITY_IDS, F_PARENT_ENABLED, F_PARENT_ID, F_VECTOR,
)


async def embed_and_upsert(milvus: AsyncMilvusClient, embeddings: Optional[EmbeddingClient],
                           parents: list[dict], children: list[dict],
                           batch_size: Optional[int] = None) -> dict:
    embeddings = embeddings or EmbeddingClient()
    batch_size = batch_size or int(get("embedding.batch_size", 64))
    parents_map = {
        p["parent_id"]: {"doc_id": p["doc_id"], "kb_id": p.get("kb_id"), "enabled": p.get("enabled", True)}
        for p in parents
    }

    total = 0
    for i in range(0, len(children), batch_size):
        batch = children[i:i + batch_size]
        vectors = await asyncio.to_thread(embeddings.embed, [c["content"] for c in batch])
        rows = []
        for c, vec in zip(batch, vectors):
            parent = parents_map.get(c["parent_id"], {"doc_id": "", "kb_id": None, "enabled": True})
            rows.append({
                F_CHILD_ID: c["child_id"], F_CONTENT: c["content"],
                F_PARENT_ID: c["parent_id"], F_DOC_ID: parent["doc_id"],
                F_KB_ID: str(parent["kb_id"]) if parent.get("kb_id") else "",
                F_ENABLED: True, F_PARENT_ENABLED: bool(parent["enabled"]),
                F_KG_ENTITY_IDS: [], F_VECTOR: vec,
            })
        await milvus.upsert(rows)
        total += len(rows)
    return {"upserted": total, "chunk_count": len(children)}
