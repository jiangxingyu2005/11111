"""知识库数据访问层(多知识库根表 CRUD,逻辑删除,SQLAlchemy 异步)。

所有函数接受一个 AsyncSession(由接口层 Depends(get_db) 注入),
事务由函数内部 commit;删除为逻辑删除(置 deleted_at),查询均过滤 deleted_at IS NULL。
"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import KnowledgeBase


def _coerce_id(kb_id) -> UUID:
    """统一把 kb_id 转成 uuid.UUID(接受 str 或 UUID)。"""
    return kb_id if isinstance(kb_id, UUID) else UUID(str(kb_id))


async def create_kb(session: AsyncSession, name: str, description: str = "") -> KnowledgeBase:
    """创建知识库,返回完整记录。name 冲突时 sqlalchemy.exc.IntegrityError 冒泡。"""
    kb = KnowledgeBase(name=name, description=description)
    session.add(kb)
    await session.commit()
    await session.refresh(kb)
    return kb


async def list_kbs(session: AsyncSession) -> list[KnowledgeBase]:
    """未删除的知识库列表,按创建时间升序。"""
    stmt = (
        select(KnowledgeBase)
        .where(KnowledgeBase.deleted_at.is_(None))
        .order_by(KnowledgeBase.created_at)
    )
    return list(await session.scalars(stmt))


async def get_kb(session: AsyncSession, kb_id) -> KnowledgeBase | None:
    """按 id 查未删除的记录,不存在或已删除返回 None。"""
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == _coerce_id(kb_id),
        KnowledgeBase.deleted_at.is_(None),
    )
    return await session.scalar(stmt)


async def update_kb(session: AsyncSession, kb_id, name=None, description=None) -> KnowledgeBase | None:
    """更新未删除记录的非 None 字段,updated_at 置 now()。不存在/已删除返回 None。"""
    kb = await get_kb(session, kb_id)
    if kb is None:
        return None
    if name is not None:
        kb.name = name
    if description is not None:
        kb.description = description
    kb.updated_at = func.now()
    await session.commit()
    await session.refresh(kb)
    return kb


async def delete_kb(session: AsyncSession, kb_id) -> bool:
    """逻辑删除:置 deleted_at = now()。返回是否删到(仅未删除记录)。"""
    kb = await get_kb(session, kb_id)
    if kb is None:
        return False
    kb.deleted_at = func.now()
    kb.updated_at = func.now()
    await session.commit()
    return True
