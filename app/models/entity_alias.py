"""别名库 ORM:代号/别名 → 类别字典实体类型条目(一对多:一个类型可对应多个代号)。

独立于 entity_registry.aliases(那是 LLM 自动抽取产出的实体别名),本表是人工维护的
「代号 → 字典条目」规则:实体抽取时作为动态提示词注入,让模型认出代号属于哪个类别。
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class EntityAlias(Base):
    __tablename__ = "entity_alias"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    alias: Mapped[str] = mapped_column(String, nullable=False, unique=True)  # 代号/别名(如 a1)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)          # 所属类别字典实体类型名(如 辐射源)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
