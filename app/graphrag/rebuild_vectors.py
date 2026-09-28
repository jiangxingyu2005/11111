"""重建 Milvus 向量:清空两个集合并用当前 embedding provider 重嵌入存量数据。

运行:
    cd qingxi && python -m app.graphrag.rebuild_vectors

场景:切换 GRAPHRAG_EMBEDDING_PROVIDER(如 mock -> bge)后,存量向量与查询向量不同源,
语义检索失效;且历史反复 ingest 残留孤儿向量、verify 脚本可能误删实体向量。
本脚本从 PG(唯一结构真源)重建 ems_chunks + ems_entities,统一用当前 provider,
清孤儿、补缺失。幂等:可重复执行。不改 PG/Neo4j 结构数据。
"""

from __future__ import annotations

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台无法编码中文
except Exception:
    pass

from ..models import ChunkRelation, EntityRegistryEntry, ParentDocument
from ..utils.database import SessionLocal, init_db
from . import db
from .config import get
from .dict import load_mergeable_types
from .embeddings import EmbeddingClient
from .milvus_client import (
    AsyncMilvusClient, F_CHILD_ID, F_CONTENT, F_DOC_ID, F_ENABLED, F_KB_ID,
    F_KG_ENTITY_IDS, F_PARENT_ENABLED, F_PARENT_ID, F_VECTOR,
)


async def main() -> int:
    await init_db()
    milvus = AsyncMilvusClient()
    embeddings = EmbeddingClient()  # provider 取 .env(当前 bge)

    # 1. 清空两个集合(drop 幂等,create_* 幂等重建 schema)
    await milvus.drop_collection(milvus.collection_name)
    await milvus.drop_collection(milvus.entity_collection_name)
    await milvus.create_collection()
    await milvus.create_entity_collection()

    async with SessionLocal() as session:
        # 2. 重建子块向量:chunk_relation + parent_document 按 parent_id join
        chunks = await db.select_all(session, ChunkRelation)
        parents = await db.select_all(session, ParentDocument)
        p_map = {p["parent_id"]: p for p in parents}
        batch_size = int(get("embedding.batch_size", 64))

        total_chunks = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = await asyncio.to_thread(embeddings.embed, [c["content"] for c in batch])
            rows = []
            for c, vec in zip(batch, vecs):
                p = p_map.get(c["parent_id"], {})
                kg_ids = [x for x in (p.get("kg_entity_id") or "").split(",") if x]
                rows.append({
                    F_CHILD_ID: c["child_id"], F_CONTENT: c["content"],
                    F_PARENT_ID: c["parent_id"], F_DOC_ID: p.get("doc_id", ""),
                    F_KB_ID: str(p.get("kb_id") or ""),
                    F_ENABLED: bool(c.get("enabled", True)),
                    F_PARENT_ENABLED: bool(p.get("enabled", True)),
                    F_KG_ENTITY_IDS: kg_ids, F_VECTOR: vec,
                })
            await milvus.upsert(rows)
            total_chunks += len(rows)
        await milvus.flush()  # 子块 upsert 不自动 flush,不 flush 则 search 看不到新向量

        # 3. 重建实体名向量:仅 active 且可向量化类型(与 resolve.persist 一致)
        mergeable = await load_mergeable_types(session)
        regs = await db.select_all(session, EntityRegistryEntry)
        total_entities = 0
        for r in regs:
            if r["status"] != "active" or r["entity_type"] not in mergeable:
                continue
            vec = await asyncio.to_thread(embeddings.embed_one, r["name"])
            await milvus.upsert_entity_vector(r["entity_id"], r["name"], r["entity_type"], vec)
            total_entities += 1

    print(f"重建完成:chunks={total_chunks}(父块 {len(p_map)}) entities={total_entities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
