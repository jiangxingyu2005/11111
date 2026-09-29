"""知识库 CRUD 接口。"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ..schemas import (
    DocumentRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
)
from ..utils import document, kb, minio_client
from ..utils.database import get_db

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_kb(payload: KnowledgeBaseCreate, db: AsyncSession = Depends(get_db)) -> KnowledgeBaseRead:
    try:
        return await kb.create_kb(db, payload.name, payload.description)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"知识库名称已存在: {payload.name}")


@router.get("")
async def list_kbs(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1, description="页码,从 1 开始"),
    page_size: int | None = Query(None, ge=1, le=100, description="每页条数;不传则返回全量列表"),
):
    """知识库列表:不传 page_size 返回全量(兼容旧接口);传了则分页。"""
    if page_size is None:
        rows = await kb.list_kbs(db)
        return [KnowledgeBaseRead.model_validate(r).model_dump(mode="json") for r in rows]

    items, total = await kb.list_kbs_page(db, page, page_size)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [KnowledgeBaseRead.model_validate(r).model_dump(mode="json") for r in items],
    }


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
async def get_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)) -> KnowledgeBaseRead:
    row = await kb.get_kb(db, kb_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    return row


@router.get("/{kb_id}/documents", response_model=list[DocumentRead])
async def list_kb_documents(kb_id: UUID, db: AsyncSession = Depends(get_db)) -> list[DocumentRead]:
    """知识库文件表:文件名 + 源文件大小(MinIO) + 登记时间等(不含逻辑删除)。"""
    if await kb.get_kb(db, kb_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")

    docs = await document.list_documents(db, kb_id)
    sizes = await run_in_threadpool(minio_client.list_source_sizes, kb_id)

    return [
        DocumentRead(
            doc_id=str(d.id),
            filename=d.filename,
            size=sizes.get(str(d.id)),
            score=d.score,
            in_kb=d.in_kb,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in docs
    ]


@router.put("/{kb_id}", response_model=KnowledgeBaseRead)
async def update_kb(kb_id: UUID, payload: KnowledgeBaseUpdate, db: AsyncSession = Depends(get_db)) -> KnowledgeBaseRead:
    row = await kb.update_kb(db, kb_id, name=payload.name, description=payload.description)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    return row


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)) -> None:
    if not await kb.delete_kb(db, kb_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
