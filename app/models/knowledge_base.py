"""知识库 ORM 模型(多知识库根表,逻辑删除)。"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    # 仅未删除记录 name 唯一(逻辑删除后可重建同名)
    __table_args__ = (
        Index("uq_knowledge_bases_name_active", "name", unique=True,
              postgresql_where=text("deleted_at IS NULL")),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
