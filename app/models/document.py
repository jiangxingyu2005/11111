"""文件(文档)ORM 模型:解析清洗产物登记,归属某个知识库,逻辑删除。"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_kb_id", "kb_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)  # 文件 uuid
    kb_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("knowledge_bases.id"), nullable=False
    )  # 所属知识库 id
    filename: Mapped[str] = mapped_column(String, nullable=False)  # 原始文件名
    meta_path: Mapped[str] = mapped_column(Text, nullable=False, default="")  # 元数据路径(meta.json)
    clean_md_path: Mapped[str] = mapped_column(Text, nullable=False, default="")  # 清洗后 md 路径(clean.md)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 文件质量分(打分接入后填充)
    in_kb: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # 是否已入知识库(向量化)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
