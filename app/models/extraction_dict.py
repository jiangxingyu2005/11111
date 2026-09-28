"""抽取字典 ORM 模型:定义「要抽取哪些实体节点 / 关系」的唯一数据源。

字段:名称 / 类别(entity 节点 | relation 关系)/ 描述 / JSON Schema(LLM function calling)
      / 触发词(仅关系)/ 最小置信度(仅关系)/ 逻辑删除。
全局共享,不分知识库;改字典即时生效(抽取侧每次现查)。
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, String, Text, false, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class ExtractionDict(Base):
    __tablename__ = "extraction_dict"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)          # 名称(实体类型 / 关系中文名)
    en: Mapped[str | None] = mapped_column(String, nullable=True)      # 英文名(关系用,如 BELONGS_TO)
    category: Mapped[str] = mapped_column(String, nullable=False)      # entity(节点) / relation(关系)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    json_schema: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # LLM function-calling 的 JSON Schema
    trigger_words: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # 关系触发词(仅 relation)
    min_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # 关系最低置信度(默认 0.75)
    # 关系两端实体类型约束(domain/range):head_types/tail_types 存中文实体类型名集合,
    # 约束该关系能连哪两类节点(仅 relation;空 = 不约束),防止「辐射源-装备于-辐射源」这类荒谬边。
    head_types: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tail_types: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 是否允许向量语义层做同义异名归并(消歧匹配器②):参数值/枚举类(频率/调制方式/信号等)为 false,
    # 只走确定性匹配;名词性类(装备/辐射源/平台/任务)为 true 才进 ems_entities 向量比较。
    semantic_mergeable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
