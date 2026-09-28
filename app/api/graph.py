"""GraphRAG 接口:抽取字典增删改查(Phase 1)。

字典 = 定义「要抽取哪些实体节点 / 关系」的唯一数据源,全局共享、不分知识库。
抽取侧每次现查,改字典即时生效。逻辑删除(置 deleted_at)后不参与抽取。
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ExtractionDict
from ..schemas import (
    ExtractionDictCreate,
    ExtractionDictRead,
    ExtractionDictUpdate,
    GraphQueryRequest,
)
from ..utils.database import get_db

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

_ALLOWED_CATEGORY = {"entity", "relation"}


def _validate_category(category: str) -> None:
    if category not in _ALLOWED_CATEGORY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"category 必须是 entity 或 relation,收到: {category!r}",
        )


async def _get_row(db: AsyncSession, dict_id: UUID) -> ExtractionDict | None:
    stmt = select(ExtractionDict).where(
        ExtractionDict.id == dict_id,
        ExtractionDict.deleted_at.is_(None),
    )
    return await db.scalar(stmt)


@router.post("/dict", response_model=ExtractionDictRead, status_code=status.HTTP_201_CREATED)
async def create_dict(payload: ExtractionDictCreate, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """新增字典条目(实体类型或关系定义)。"""
    _validate_category(payload.category)
    row = ExtractionDict(**payload.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/dict", response_model=list[ExtractionDictRead])
async def list_dict(
    db: AsyncSession = Depends(get_db),
    category: str | None = Query(None, description="按类别过滤:entity / relation"),
) -> list[ExtractionDictRead]:
    """列表(不含已逻辑删除),可按类别过滤。"""
    stmt = select(ExtractionDict).where(ExtractionDict.deleted_at.is_(None))
    if category is not None:
        _validate_category(category)
        stmt = stmt.where(ExtractionDict.category == category)
    stmt = stmt.order_by(ExtractionDict.created_at)
    return list(await db.scalars(stmt))


@router.get("/dict/{dict_id}", response_model=ExtractionDictRead)
async def get_dict(dict_id: UUID, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """查询单条字典(不含已逻辑删除)。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    return row


@router.put("/dict/{dict_id}", response_model=ExtractionDictRead)
async def update_dict(dict_id: UUID, payload: ExtractionDictUpdate, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """更新字典(仅传要改的字段)。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    if payload.category is not None:
        _validate_category(payload.category)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(row, k, v)
    row.updated_at = func.now()
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/dict/{dict_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dict(dict_id: UUID, db: AsyncSession = Depends(get_db)) -> None:
    """逻辑删除:置 deleted_at,不再参与抽取。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    row.deleted_at = func.now()
    row.updated_at = func.now()
    await db.commit()


@router.post("/query")
async def graph_query(payload: GraphQueryRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """在线查询:意图分流 → 实体匹配 → A/B 双路召回 → 重排 → (关系型)图扩展 → LLM 生成。

    命中结果计入溯源(parent/file recall_count + recall_log)。Neo4j/Milvus 不可用时
    返回 502,不影响其它接口与主链路。
    """
    from ..graphrag.dict import load_relations
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.llm import LLMClient
    from ..graphrag.milvus_client import AsyncMilvusClient
    from ..graphrag.neo4j_client import Neo4jClient
    from ..graphrag.query import router as query_router
    from ..graphrag.trace.hooks import make_trace_hook

    relation_defs = await load_relations(db)
    try:
        async with Neo4jClient() as neo4j:
            result = await query_router.search(
                query=payload.query,
                milvus=AsyncMilvusClient(),
                neo4j=neo4j,
                embeddings=EmbeddingClient(),
                llm=LLMClient(),
                relation_defs=relation_defs,
                kb_id=payload.kb_id,
                trace_hook=make_trace_hook(db),
            )
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("图谱查询失败: %s", payload.query)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图谱查询失败: {e}",
        )
    return result


@router.get("/trace/files")
async def trace_files(kb_id: UUID | None = Query(None, description="按知识库过滤,不传 = 全库"),
                      db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:文件列表 + 文件级召回次数。"""
    from ..graphrag.trace import registry

    return {"files": await registry.list_files(db, kb_id=kb_id)}


@router.get("/trace/files/{doc_id}")
async def trace_file_detail(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:文件详情(召回历史时间线)。"""
    from ..graphrag.trace import registry

    return await registry.file_detail(db, doc_id)


@router.get("/trace/files/{doc_id}/locate")
async def trace_file_locate(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:源文件定位(存储路径)。"""
    from ..graphrag.trace import registry

    return await registry.locate(db, doc_id)
