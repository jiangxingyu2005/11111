from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CleanStats(BaseModel):
    blocks_before: int = 0
    blocks_after: int = 0
    removed_images: int = 0
    removed_page_numbers: int = 0
    removed_headers: int = 0
    removed_short_blocks: int = 0
    removed_duplicates: int = 0


class FileArtifacts(BaseModel):
    raw_md: str
    raw_json: str
    clean_md: str
    clean_json: str
    meta: str


class FileResult(BaseModel):
    filename: str
    doc_id: str = ""
    success: bool = False
    page_count: Optional[int] = None
    clean_md: Optional[str] = None
    files: Optional[FileArtifacts] = None
    stats: Optional[CleanStats] = None
    error: Optional[str] = None


class ParseResponse(BaseModel):
    results: list[FileResult]


class IqResult(FileResult):
    """IQ 信号解析结果:在 FileResult 基础上附 IQ 处理中间结果(字段结构照搬 dc see_parse)。"""
    snr: Optional[dict] = None
    attenuation_db: Optional[float] = None
    charts: Optional[dict] = None
    features: Optional[dict] = None
    prediction: Optional[dict] = None
    param_guess: Optional[dict] = None
    iq_raw: Optional[list] = None
    iq_clean: Optional[list] = None
    iq_masked: Optional[list] = None
    iq_recovered: Optional[list] = None
    iq_meta: Optional[dict] = None


class IqParseResponse(BaseModel):
    results: list[IqResult]


class IqIngestRequest(BaseModel):
    """IQ 入库打标请求:文档 uuid + 人工类别标签(雷达型号/类型)。"""
    doc_id: UUID
    category: str


class IqIngestResponse(BaseModel):
    """IQ 入库结果:向量维度 + 图谱入库计数。"""
    doc_id: str
    category: str
    vector_dim: int
    graph: Optional[dict] = None


class IqSearchRequest(BaseModel):
    """IQ 判断类型请求:文档 uuid。"""
    doc_id: UUID


class IqSearchResponse(BaseModel):
    """IQ 判断类型结果:类型 + 概率(向量库为空则 status=unknown)。"""
    status: str
    type: Optional[str] = None
    probability: Optional[float] = None


class MinerUHealth(BaseModel):
    ok: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    mineru: MinerUHealth


class KnowledgeBaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DocumentRead(BaseModel):
    """知识库文件表条目:文件名 + 源文件大小(MinIO) + 登记时间。"""
    doc_id: str
    filename: str
    size: Optional[int] = None       # 源文件大小(字节),MinIO 缺失时 null
    score: Optional[float] = None    # 文件质量分
    in_kb: bool = False              # 是否已入图谱/知识库
    created_at: datetime
    updated_at: datetime


class ExtractionDictCreate(BaseModel):
    """新建抽取字典条目(category=entity 节点 / relation 关系 / derived 派生关系)。"""
    name: str
    en: Optional[str] = None
    category: str              # entity / relation / derived
    description: str = ""
    json_schema: Optional[dict] = None
    trigger_words: Optional[list[str]] = None
    min_confidence: Optional[float] = None
    head_types: Optional[list[str]] = None   # 关系 head 允许的实体类型(domain/range)
    tail_types: Optional[list[str]] = None   # 关系 tail 允许的实体类型
    enabled: bool = True
    semantic_mergeable: bool = False  # 是否允许向量语义层归并(消歧匹配器②;参数值类=false)


class ExtractionDictUpdate(BaseModel):
    """更新抽取字典条目(仅传要改的字段)。"""
    name: Optional[str] = None
    en: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    json_schema: Optional[dict] = None
    trigger_words: Optional[list[str]] = None
    min_confidence: Optional[float] = None
    head_types: Optional[list[str]] = None
    tail_types: Optional[list[str]] = None
    enabled: Optional[bool] = None
    semantic_mergeable: Optional[bool] = None


class ExtractionDictRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    en: Optional[str] = None
    category: str
    description: str
    json_schema: Optional[dict] = None
    trigger_words: Optional[list] = None
    min_confidence: Optional[float] = None
    head_types: Optional[list] = None
    tail_types: Optional[list] = None
    enabled: bool
    semantic_mergeable: bool
    created_at: datetime
    updated_at: datetime


class RuleCreate(BaseModel):
    """新建预设推理规则（需求(7)：检索后触发预设推理规则）。"""
    id: str                     # 语义主键，形如 "R_001"（调用方显式指定，非 UUID）
    name: str
    description: str = ""
    condition: dict             # 谓词 DSL：{"all": [...], "any": [...]}
    action: dict                # 命中动作：{"type":"mark","label","severity","hint"}
    priority: int = 0
    enabled: bool = True
    kb_id: Optional[UUID] = None    # 空 = 全局规则


class RuleUpdate(BaseModel):
    """更新预设推理规则（仅传要改的字段；None = 不修改）。"""
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[dict] = None
    action: Optional[dict] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    kb_id: Optional[UUID] = None


class RuleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    condition: dict
    action: dict
    priority: int
    enabled: bool
    kb_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime


class GraphQueryRequest(BaseModel):
    """在线查询请求:自然语言问题 + 知识库 id(可选,不传 = 全库)。"""
    query: str
    kb_id: Optional[UUID] = None
