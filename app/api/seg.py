"""GraphRAG 分段管理接口(Phase 3):分段/文档级开关 + 七步编辑 + 分流 + 对账。

全部 async def;只新增,不动既有接口。Neo4j/Milvus 不可用时返回 502,不影响主链路。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..graphrag import db as gdb
from ..graphrag.milvus_client import AsyncMilvusClient
from ..graphrag.neo4j_client import Neo4jClient
from ..models import ParentDocument
from ..utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph-seg"])


class EnabledRequest(BaseModel):
    enabled: bool


class EditRequest(BaseModel):
    content: str
    edited_by: str | None = None


class FileSize(BaseModel):
    size: int


class RouteRequest(BaseModel):
    files: list[FileSize] | None = None
    sizes: list[int] | None = None


def _project(r: dict) -> dict:
    return {
        "parent_id": r["parent_id"], "doc_id": r["doc_id"],
        "kb_id": str(r["kb_id"]) if r.get("kb_id") else None,
        "title": r.get("title", ""), "enabled": r.get("enabled", True),
        "content_version": r.get("content_version", 1),
        "recall_count": r.get("recall_count", 0),
    }


@router.get("/segments")
async def list_segments(db: AsyncSession = Depends(get_db)) -> dict:
    """分段列表(全库,不含正文)。"""
    rows = await gdb.select_all(db, ParentDocument)
    return {"segments": [_project(r) for r in rows]}


@router.get("/segments/{parent_id}")
async def get_segment(parent_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """单分段详情(含正文)。不存在返回 found=false。"""
    rows = await gdb.select_where(db, ParentDocument, parent_id=parent_id)
    if not rows:
        return {"found": False}
    return {"found": True, **rows[0]}


@router.post("/segments/{parent_id}/enabled")
async def segment_enabled(parent_id: str, body: EnabledRequest,
                          db: AsyncSession = Depends(get_db)) -> dict:
    """分段级开关(幂等)。"""
    from ..graphrag.seg.manager import set_segment_enabled

    try:
        async with Neo4jClient() as neo4j:
            result = await set_segment_enabled(db, AsyncMilvusClient(), neo4j,
                                               parent_id, body.enabled)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("分段开关失败: %s", parent_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"分段开关失败: {e}")
    return result


@router.post("/documents/{doc_id}/enabled")
async def document_enabled(doc_id: str, body: EnabledRequest,
                           db: AsyncSession = Depends(get_db)) -> dict:
    """文档级总开关(幂等)。"""
    from ..graphrag.seg.manager import set_document_enabled

    try:
        async with Neo4jClient() as neo4j:
            result = await set_document_enabled(db, AsyncMilvusClient(), neo4j,
                                                doc_id, body.enabled)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("文档级开关失败: %s", doc_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"文档级开关失败: {e}")
    return result


@router.put("/segments/{parent_id}")
async def edit_segment(parent_id: str, body: EditRequest,
                       db: AsyncSession = Depends(get_db)) -> dict:
    """七步编辑:重切分 → 重嵌 → 增量重抽 → Neo4j 差量校正 → 回写 → CDC 留痕。"""
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.llm import LLMClient
    from ..graphrag.seg.orchestrator import edit_segment as _edit

    try:
        async with Neo4jClient() as neo4j:
            result = await _edit(db, AsyncMilvusClient(), neo4j, parent_id,
                                 body.content, body.edited_by,
                                 LLMClient(), EmbeddingClient())
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("分段编辑失败: %s", parent_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"分段编辑失败: {e}")
    return result


@router.post("/route")
async def route_pipeline(body: RouteRequest) -> dict:
    """分流判定(非破坏,不影响 /parse 行为;offline 仅提示)。"""
    from ..graphrag.route.router import route_pipeline as _route

    if body.sizes is not None:
        files = [{"size": s} for s in body.sizes]
    else:
        files = list(body.files or [])
    result = _route(files)
    return {"lane": result.lane, "rule": result.rule, "tip": result.tip}


@router.post("/reconcile")
async def reconcile_graph(db: AsyncSession = Depends(get_db)) -> dict:
    """三方对账:PG parent_document ↔ Neo4j :Segment。"""
    from ..graphrag.cdc.reconcile import reconcile

    try:
        async with Neo4jClient() as neo4j:
            return await reconcile(db, neo4j)
    except Exception as e:
        logger.exception("对账失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"对账失败: {e}")
