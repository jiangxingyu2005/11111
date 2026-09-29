"""预设推理规则 ORM 模型（需求(7)「触发预设推理规则」的唯一数据源）。

规则 = 一条用户自定义的「条件 → 动作」评判项：检索完成后由确定性规则引擎求值，
命中的规则作为结构化事实注入 LLM 提示词，LLM 只负责翻译/解释/下结论，不负责计算。

字段说明：
  - id        语义主键（String(50)，形如 "R_001"，由调用方显式指定，非 UUID）
  - condition JSONB 谓词（见 docs/api/10-rules.md 的 DSL：all/any + field/op/value）
  - action    JSONB 动作（{type:"mark", label, severity, hint}，只做标记不做副作用）
  - priority  int    优先级，越大越靠前输出
  - kb_id     UUID   作用域知识库（空 = 全局规则，对所有知识库生效）
  - enabled   bool   是否启用
  - deleted_at 逻辑删除（沿用全项目 deleted_at 约定，不用 is_deleted 布尔）
"""
from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class Rule(Base):
    __tablename__ = "rules"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)          # 语义主键 R_XXX
    name: Mapped[str] = mapped_column(String(200), nullable=False)         # 规则名（中文可读）
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    condition: Mapped[dict] = mapped_column(JSONB, nullable=False)         # 谓词（all/any 子句）
    action: Mapped[dict] = mapped_column(JSONB, nullable=False)            # 命中动作（label/severity/hint）
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    kb_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)  # 空 = 全局
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
