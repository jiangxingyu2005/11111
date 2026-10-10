"""每文档编排器(ems tests/helpers.py build_env 的单文档、免 CSV 版)。

链路:chunk -> 源文件登记 -> embed(Milvus)-> 字典 -> 实体抽取 -> 关系抽取 -> 绑定 ->
Neo4j 导入 -> 回写 PG/Milvus -> 漏抽质检。从 _finalize 直通:ingest_document(kb_id, doc_id, clean_md, engine)。

任何一步失败向上抛,由 _finalize 捕获记日志,不影响 parse 主链路;重跑幂等
(确定性哈希 + PG upsert + Neo4j MERGE + Milvus upsert)。
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from ..config import settings
from ..models import IngestQuality, SourceFileRegistry
from ..utils.database import SessionLocal
from . import chunk, db, embed, extract_entities, extract_relations, neo4j_load, resolve
from .binding import bind_back
from .dict import load_entity_defs, load_relations
from .embeddings import EmbeddingClient
from .llm import LLMClient
from .mesh.apply import enhance_graph
from .milvus_client import AsyncMilvusClient
from .neo4j_client import Neo4jClient
from .parse_md import content_hash

logger = logging.getLogger(__name__)


async def ingest_document(kb_id: UUID, doc_id: str, clean_md: str, engine: str,
                          file_name: Optional[str] = None, graph: bool = True) -> dict:
    embeddings = EmbeddingClient()
    milvus = AsyncMilvusClient()

    async with SessionLocal() as session:
        # 1. 切分 + 写 PG(parent_document + chunk_relation)
        parents, children = await chunk.write_to_pg(session, doc_id, kb_id, clean_md)
        # 2. 源文件登记
        await db.upsert(session, SourceFileRegistry, {
            "doc_id": doc_id, "kb_id": kb_id,
            "file_name": file_name or f"{doc_id}.md",
            "content_hash": content_hash(clean_md), "doc_type": engine,
        }, "doc_id")
        # 3. 嵌入 + Milvus upsert(噪声文档也切块,问答还能检索原文)
        await embed.embed_and_upsert(milvus, embeddings, parents, children)
        if not graph:
            logger.info("噪声/低质隔离:文档 %s(%s)只切块不抽图谱", doc_id, file_name)
            await _record_quality(session, doc_id, kb_id, file_name,
                                  len(parents), 0, 0, 0, kg_status="low")
            await session.commit()
            return {
                "parents": len(parents), "children": len(children),
                "entities": 0, "relations": 0, "binds": 0, "graph": False,
            }

        llm = LLMClient()
        async with Neo4jClient() as neo4j:
            # 4. 抽取字典(实体类型完整定义 / 关系定义)
            entity_defs = await load_entity_defs(session)
            entity_types = [d["name"] for d in entity_defs]
            relation_defs = await load_relations(session)
            # 5. 实体抽取(完整定义供 LLM prompt 讲清类型含义)
            entities = await extract_entities.extract(session, llm, parents, entity_types, entity_defs)
            # 6. 关系抽取
            relations, _ = await extract_relations.extract(session, entities, llm, parents, relation_defs)
            # 6.5 跨文档消歧:对齐到全局注册表 + 写回(开关关闭时原样透传)
            entities, relations, _remap = await resolve.align(session, entities, relations,
                                                              milvus, embeddings, llm)
            await resolve.persist(session, entities, milvus, embeddings)
            # 7. mesh 图增强:归一消歧 + 派生边 + 保真绑定(mock 下空操作,不改变既有行为)
            entities, relations, binds = enhance_graph(entities, relations)
            # 8. Neo4j 导入(含表外 ID 校验)
            await neo4j_load.import_graph(neo4j, entities, relations, binds, parents)
            # 9. 回写 PG kg_* + Milvus kg_entity_ids
            await bind_back(session, milvus, binds, parents, children)
            # 9.5 漏抽质检门禁:写 ingest_quality,让「抽少/抽空」可见
            await _record_quality(session, doc_id, kb_id, file_name,
                                  len(parents), len(entities), len(relations), len(binds))
            await session.commit()

    return {
        "parents": len(parents), "children": len(children),
        "entities": len(entities), "relations": len(relations),
        "binds": len(binds), "graph": True,
    }


def _kg_status(chunk_count: int, entity_count: int) -> str:
    """按漏抽门禁判 kg_status:空图谱=empty,抽太少=low,否则 ok。"""
    if entity_count == 0:
        return "empty"
    if chunk_count and entity_count < chunk_count * settings.graphrag_quality_entity_chunk_ratio:
        return "low"
    return "ok"


async def _record_quality(session, doc_id: str, kb_id: UUID, file_name: Optional[str],
                          chunk_count: int, entity_count: int,
                          relation_count: int, bind_count: int,
                          kg_status: Optional[str] = None) -> None:
    """写入库质检行(幂等 upsert);empty 时告警。"""
    status = kg_status or _kg_status(chunk_count, entity_count)
    if status == "empty":
        logger.warning("漏抽告警:文档 %s(%s)抽取到 0 个实体,图谱为空", doc_id, file_name)
    await db.upsert(session, IngestQuality, {
        "doc_id": doc_id, "kb_id": kb_id, "file_name": file_name or f"{doc_id}.md",
        "chunk_count": chunk_count, "entity_count": entity_count,
        "relation_count": relation_count, "bind_count": bind_count,
        "kg_status": status,
    }, "doc_id")
