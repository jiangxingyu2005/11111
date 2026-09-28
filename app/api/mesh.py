"""GraphRAG 图增强接口(Phase 4):图质量四指标 + 孤儿兜底绑定。

全 async def;只新增,不动既有接口。Neo4j 不可用时返回 502,不影响主链路。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..graphrag import db as gdb
from ..graphrag.mesh.fallback import link_orphan_segments
from ..graphrag.mesh.fusion import compute_metrics, detect_orphans
from ..graphrag.mesh.registry import EntityRegistry
from ..graphrag.neo4j_client import Neo4jClient
from ..config import settings
from ..models import IngestQuality, ParentDocument, PendingExtraction
from ..utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph-mesh"])


@router.get("/metrics")
async def graph_metrics(db: AsyncSession = Depends(get_db)) -> dict:
    """图质量四指标(全局):孤立率 / BINDS 覆盖率 / 最大连通分量 / 平均度。"""
    try:
        async with Neo4jClient() as neo4j:
            entities = await neo4j.list_all_entities()
            binds = await neo4j.list_all_binds()
            relations = await neo4j.list_all_relations()
    except Exception as e:
        logger.exception("读取图指标失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"读取图指标失败: {e}")

    parents = await gdb.select_all(db, ParentDocument)
    metrics = compute_metrics(entities, binds, relations, parents)
    name_by_id = {e["entity_id"]: e["name"] for e in entities}
    metrics["totals"] = {
        "entities": len(entities), "relations": len(relations), "parents": len(parents),
    }
    metrics["orphan_entities"] = [
        {"entity_id": oid, "name": name_by_id.get(oid)}
        for oid in detect_orphans(entities, binds)
    ]
    return metrics


@router.get("/quality")
async def graph_quality(db: AsyncSession = Depends(get_db)) -> dict:
    """漏抽质检 + 图健康门禁:四指标 + healthy 布尔 + 漏抽文档清单(kg_status != ok)。"""
    try:
        async with Neo4jClient() as neo4j:
            entities = await neo4j.list_all_entities()
            binds = await neo4j.list_all_binds()
            relations = await neo4j.list_all_relations()
    except Exception as e:
        logger.exception("读取图指标失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"读取图指标失败: {e}")

    parents = await gdb.select_all(db, ParentDocument)
    metrics = compute_metrics(entities, binds, relations, parents)

    quality_rows = await gdb.select_all(db, IngestQuality)
    leak = [r for r in quality_rows if r.get("kg_status") != "ok"]
    empty_count = sum(1 for r in quality_rows if r.get("kg_status") == "empty")

    orphan_rate = metrics.get("orphan_rate", 0.0)
    coverage = metrics.get("binds_coverage", 0.0)
    empty_ratio = (empty_count / len(quality_rows)) if quality_rows else 0.0
    healthy = (
        orphan_rate <= settings.graphrag_quality_max_orphan_rate
        and coverage >= settings.graphrag_quality_min_binds_coverage
        and empty_ratio <= settings.graphrag_quality_max_empty_ratio
    )

    return {
        "metrics": metrics,
        "totals": {
            "entities": len(entities), "relations": len(relations),
            "parents": len(parents), "documents_checked": len(quality_rows),
            "empty_documents": empty_count,
        },
        "healthy": healthy,
        "gates": {
            "orphan_rate": {"value": orphan_rate, "max": settings.graphrag_quality_max_orphan_rate},
            "binds_coverage": {"value": coverage, "min": settings.graphrag_quality_min_binds_coverage},
            "empty_ratio": {"value": empty_ratio, "max": settings.graphrag_quality_max_empty_ratio},
        },
        "leak_documents": leak,
    }


@router.post("/mesh/link-orphans")
async def mesh_link_orphans(db: AsyncSession = Depends(get_db)) -> dict:
    """孤儿兜底绑定:对 kg_entity_id 为空的父块按别名/指纹链接现有实体(幂等)。"""
    try:
        async with Neo4jClient() as neo4j:
            registry = EntityRegistry(await neo4j.list_all_entities_full())
            parents = await gdb.select_all(db, ParentDocument)
            result = link_orphan_segments(parents, registry)
            for link in result["linked"]:
                await neo4j.merge_binds(link["entity_id"], link["parent_id"],
                                        "supplemental", 0.5)
        # 回写 PG kg_*(supplemental)+ 未命中留痕(幂等:已留痕不重复插入)
        for link in result["linked"]:
            await gdb.update(db, ParentDocument, {
                "parent_id": link["parent_id"], "kg_entity_id": link["entity_id"],
                "kg_role": "supplemental", "kg_confidence": 0.5,
            }, "parent_id")
        if result["reextract"]:
            existing = await gdb.select_where_in(db, PendingExtraction, "parent_id", result["reextract"])
            already = {r["parent_id"] for r in existing
                       if r.get("stage") == "entity" and r.get("reason") == "孤儿兜底无命中"}
            for pid in result["reextract"]:
                if pid in already:
                    continue
                await gdb.insert(db, PendingExtraction, {
                    "parent_id": pid, "stage": "entity",
                    "payload": {"note": "orphan fallback no match"},
                    "reason": "孤儿兜底无命中", "attempts": 1,
                })
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("孤儿兜底绑定失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"孤儿兜底绑定失败: {e}")
    return result


@router.post("/resolve")
async def graph_resolve(dry_run: bool = False,
                        db: AsyncSession = Depends(get_db)) -> dict:
    """全量实体对账:合并历史重复实体 + 四层回填(PG 注册表 / Neo4j / Milvus 双库)。

    - dry_run=true 只返回归并计划,不落库。
    - 幂等:重复调用 merged=[];Neo4j/Milvus 不可用返回 502。
    """
    from ..graphrag import resolve
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.llm import LLMClient
    from ..graphrag.milvus_client import AsyncMilvusClient

    try:
        async with Neo4jClient() as neo4j:
            result = await resolve.global_resolve(
                db, neo4j, AsyncMilvusClient(), EmbeddingClient(), LLMClient(),
                dry_run=dry_run)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("实体对账失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"实体对账失败: {e}")
    return result
