"""GraphRAG(ems_graphrag 融合)PG 表 ORM 模型。

对应 ems `sql/01_schema.sql` 的 6 张表,格式与现有 Document/KnowledgeBase 一致。
融合后新增 kb_id 贯穿:parent_document / source_file_registry 带知识库 id,查询按库隔离。
doc_id = documents.id(uuid),parent_id = {doc_id}_{章节序号},child_id = {parent_id}#{子块序号}。
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..utils.database import Base


class ParentDocument(Base):
    """父块(parent_document 表)。"""
    __tablename__ = "parent_document"
    __table_args__ = (
        Index("idx_parent_doc_id", "doc_id"),
    )

    parent_id: Mapped[str] = mapped_column(String, primary_key=True)      # {doc_id}_{章节序号}
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    kb_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=True)  # 知识库 id(融合新增)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)    # H1=1 / H2=2 / H3=3
    chapter_no: Mapped[str] = mapped_column(String, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_time: Mapped[str | None] = mapped_column(String, nullable=True)
    azimuth_range: Mapped[str | None] = mapped_column(String, nullable=True)
    freq_band: Mapped[str | None] = mapped_column(String, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    segment_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    content_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    edited_by: Mapped[str | None] = mapped_column(String, nullable=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recall_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kg_entity_id: Mapped[str | None] = mapped_column(String, nullable=True)  # primary 实体 ID(逗号分隔)
    kg_role: Mapped[str | None] = mapped_column(String, nullable=True)       # primary | supplemental | None
    kg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChunkRelation(Base):
    """子块(chunk_relation 表)。"""
    __tablename__ = "chunk_relation"
    __table_args__ = (
        Index("idx_chunk_parent", "parent_id"),
    )

    child_id: Mapped[str] = mapped_column(String, primary_key=True)          # {parent_id}#{子块序号}
    parent_id: Mapped[str] = mapped_column(
        String, ForeignKey("parent_document.parent_id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class PendingExtraction(Base):
    """抽取失败隔离(pending_extraction 表)。"""
    __tablename__ = "pending_extraction"
    __table_args__ = (
        Index("idx_pending_parent", "parent_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    parent_id: Mapped[str] = mapped_column(String, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)               # entity | relation
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False, default="")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")  # pending | resolved | abandoned
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SourceFileRegistry(Base):
    """源文件登记(source_file_registry 表)。"""
    __tablename__ = "source_file_registry"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    kb_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=True)  # 知识库 id(融合新增)
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    doc_type: Mapped[str | None] = mapped_column(String, nullable=True)
    event_time: Mapped[str | None] = mapped_column(String, nullable=True)
    azimuth_range: Mapped[str | None] = mapped_column(String, nullable=True)
    freq_band: Mapped[str | None] = mapped_column(String, nullable=True)
    recall_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_recalled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ParseError(Base):
    """解析失败(parse_error 表)。"""
    __tablename__ = "parse_error"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_name: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RecallLog(Base):
    """召回日志(recall_log 表)。"""
    __tablename__ = "recall_log"
    __table_args__ = (
        Index("idx_recall_doc", "doc_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String, nullable=False)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EntityRegistryEntry(Base):
    """实体消歧注册表(entity_registry 表):全局 canonical 实体的名字/别名/指纹对照。

    跨文档消歧的持久化确定性索引;status=merged 作为全量对账的账本(merged_into 指向 canonical)。
    """
    __tablename__ = "entity_registry"
    __table_args__ = (
        Index("idx_entity_registry_name", "entity_type", "name"),
        Index("idx_entity_registry_fp", "entity_type", "fingerprint"),
    )

    entity_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)              # 规范化名
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    aliases: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)   # 规范化别名(含 name)
    fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    properties: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")  # active | merged
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IngestQuality(Base):
    """入库质检(ingest_quality 表):每文档抽取结果的漏抽质检状态。

    让「漏抽/空图谱」从静默失败变为可见的 kg_status(ok|low|empty)+ 清单(/graph/quality)。
    """
    __tablename__ = "ingest_quality"
    __table_args__ = (
        Index("idx_ingest_quality_kb", "kb_id"),
    )

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    kb_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=True)  # 知识库 id(贯穿)
    file_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 父块数
    entity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bind_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kg_status: Mapped[str] = mapped_column(String, nullable=False, default="ok")  # ok | low | empty
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
