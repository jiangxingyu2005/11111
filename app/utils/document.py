"""文件(文档)数据访问层:登记解析清洗产物,归属知识库,逻辑删除。"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Document


def _coerce_id(file_id) -> UUID:
    """统一把 file_id 转成 uuid.UUID(接受 str 或 UUID)。"""
    return file_id if isinstance(file_id, UUID) else UUID(str(file_id))


async def create_document(
    session: AsyncSession,
    kb_id: UUID,
    filename: str,
    meta_path: str,
    clean_md_path: str,
    score: float | None = None,
    in_kb: bool = False,
    id: UUID | None = None,
) -> Document:
    """登记一条文件记录。id 必须与落盘目录名 / MinIO 对象键里的文件 uuid 一致。"""
    doc = Document(
        id=id,
        kb_id=kb_id,
        filename=filename,
        meta_path=meta_path,
        clean_md_path=clean_md_path,
        score=score,
        in_kb=in_kb,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def get_document(session: AsyncSession, file_id) -> Document | None:
    """按 id 查未删除的文件记录,不存在或已删除返回 None。"""
    stmt = select(Document).where(
        Document.id == _coerce_id(file_id),
        Document.deleted_at.is_(None),
    )
    return await session.scalar(stmt)


async def list_documents(session: AsyncSession, kb_id=None) -> list[Document]:
    """未删除的文件列表(可按知识库过滤),按创建时间升序。"""
    stmt = select(Document).where(Document.deleted_at.is_(None))
    if kb_id is not None:
        stmt = stmt.where(Document.kb_id == _coerce_id(kb_id))
    stmt = stmt.order_by(Document.created_at)
    return list(await session.scalars(stmt))


async def delete_document(session: AsyncSession, file_id) -> bool:
    """逻辑删除:置 deleted_at = now()。返回是否删到(仅未删除记录)。"""
    doc = await get_document(session, file_id)
    if doc is None:
        return False
    doc.deleted_at = func.now()
    doc.updated_at = func.now()
    await session.commit()
    return True


async def mark_document_in_kb(session: AsyncSession, file_id) -> bool:
    """置 in_kb=True(文档已入图谱/知识库)。返回是否更新到。"""
    doc = await get_document(session, file_id)
    if doc is None:
        return False
    doc.in_kb = True
    doc.updated_at = func.now()
    await session.commit()
    return True
