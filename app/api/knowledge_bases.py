"""知识库 CRUD 接口。"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import KnowledgeBaseCreate, KnowledgeBaseRead, KnowledgeBaseUpdate
from ..utils import kb
from ..utils.database import get_db

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_kb(payload: KnowledgeBaseCreate, db: AsyncSession = Depends(get_db)) -> KnowledgeBaseRead:
    try:
        return await kb.create_kb(db, payload.name, payload.description)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"知识库名称已存在: {payload.name}")


@router.get("", response_model=list[KnowledgeBaseRead])
async def list_kbs(db: AsyncSession = Depends(get_db)) -> list[KnowledgeBaseRead]:
    return await kb.list_kbs(db)


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
async def get_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)) -> KnowledgeBaseRead:
    row = await kb.get_kb(db, kb_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    return row


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
