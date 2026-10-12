"""IQ 专属检索库接口:入库打标 + 判断类型(相似度检索)。

独立于现有知识库/图谱集合的 IQ 检索库(ems_iq,256 维),专做 IQ 信号相似度检索:
- POST /iq/ingest: uuid + 类别 -> 向量化写 ems_iq(带标量)+ md 追加类别后正常入图谱
- POST /iq/search: uuid -> 向量化 -> 最近邻,返回类型 + 概率;空库返回未知
"""

import logging
import math
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..graphrag.milvus_client import IQ_FEATURE_FIELDS, AsyncMilvusClient
from ..schemas import (
    IqIngestRequest,
    IqIngestResponse,
    IqSearchRequest,
    IqSearchResponse,
)
from ..utils.database import SessionLocal
from ..utils.document import get_document, mark_document_in_kb
from ..utils.iq import process_iq_h5
from ..utils.iq_embedding import embed_document, embed_segments
from ..utils.minio_client import download_source

router = APIRouter(prefix="/iq", tags=["iq"])

logger = logging.getLogger(__name__)


def _build_scalars(result: dict, category: str) -> dict:
    """从 process_iq_h5 结果组装 ems_iq 标量字段(不含 iq_id 与向量)。"""
    flat = result.get("flat") or {}
    raw = result.get("raw_entry") or {}
    snr = (result.get("snr") or {}).get("mean_snr", 0.0)

    def _f(v, default=0.0) -> float:
        try:
            f = float(v)
            return f if f == f and f not in (float("inf"), float("-inf")) else default
        except (TypeError, ValueError):
            return default

    scalars = {
        "category": category,
        "modulation": flat.get("modulation") or raw.get("modulation") or "Unknown",
        "mod_id": int(flat.get("mod_id") or raw.get("mod_id") or 14),
        "snr": _f(snr),
        "sample_rate_hz": _f(raw.get("sample_rate_hz")),
        "symbol_rate": _f(raw.get("symbol_rate_hz")),
        "attenuation_db": _f(result.get("attenuation_db")),
    }
    for f in IQ_FEATURE_FIELDS:
        scalars[f] = _f(flat.get(f))
    return scalars


def _fmt_hz(v):
    """Hz -> 人类可读字符串(带单位);None/非数值/inf/nan 返回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if abs(f) >= 1e6:
        return f"{f / 1e6:.4f} MHz"
    if abs(f) >= 1e3:
        return f"{f / 1e3:.4f} kHz"
    return f"{f:.4f} Hz"


def _fmt_db(v):
    """dB 值 -> 字符串;None/非数值/inf/nan 返回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f"{f:.2f} dB"


def _build_param_lines(result: dict) -> list[str]:
    """从 param_guess + raw_entry 组装中文信号参数行,供实体抽取填进 properties。

    IQ 的实体属性 schema 要 频率/频段/调制方式/脉宽/重频/带宽(+采样率);而 describe_signal_features
    只输出 26 个归一化统计特征(频率均值/光谱带宽等),名称与量纲都对不上,导致抽取节点属性为空。
    这里把 analyze_iq 算出的真实参数(载频/带宽/符号速率)与 H5 元数据(采样率/符号速率)翻成
    与 schema 同名的中文行,LLM 才能按 _PARAM_RULE 填进所属实体。
    """
    params = (result.get("param_guess") or {}).get("parameters") or {}
    raw = result.get("raw_entry") or {}
    snr = (result.get("snr") or {}).get("mean_snr")

    lines: list[str] = []
    mod = params.get("modulation") or raw.get("modulation")
    if mod:
        lines.append(f"  - 调制方式: {mod}")
    carrier = _fmt_hz(params.get("carrier"))
    if carrier:
        lines.append(f"  - 载频: {carrier}")
    bw = _fmt_hz(params.get("bandwidth"))
    if bw:
        lines.append(f"  - 带宽: {bw}")
    # 符号速率优先取 H5 元数据(raw_entry),缺失退回 analyze_iq 估计值
    symbol_rate = _fmt_hz(raw.get("symbol_rate_hz") or params.get("symbol_rate"))
    if symbol_rate:
        lines.append(f"  - 符号速率: {symbol_rate}")
    sample_rate = _fmt_hz(raw.get("sample_rate_hz"))
    if sample_rate:
        lines.append(f"  - 采样率: {sample_rate}")
    snr_s = _fmt_db(snr)
    if snr_s:
        lines.append(f"  - 信噪比: {snr_s}")
    att = _fmt_db(result.get("attenuation_db"))
    if att:
        lines.append(f"  - 衰减: {att}")
    return lines


async def _load_doc_h5(doc_id: UUID):
    """查文档 + 取回原始 H5 字节;缺失抛 404。"""
    async with SessionLocal() as db:
        doc = await get_document(db, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    try:
        content = await run_in_threadpool(download_source, doc.kb_id, doc.id, doc.filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"源文件不存在: {e}"
        ) from e
    return doc, content


@router.post("/ingest", response_model=IqIngestResponse)
async def ingest_iq(payload: IqIngestRequest) -> IqIngestResponse:
    """IQ 入库打标:向量 + 标量写 ems_iq,并把类别追加到 md 后正常入知识库/知识图谱。"""
    doc_id = payload.doc_id
    category = (payload.category or "").strip()
    if not category:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="类别不能为空")

    doc, content = await _load_doc_h5(doc_id)

    # 1. 重跑 IQ 管道一次,同时拿到归一化片段(嵌入)+ 26 特征/md(标量 + 图谱)
    result = await run_in_threadpool(process_iq_h5, content)

    # 2. 向量化(文档级 mean-pool)
    vector = await run_in_threadpool(embed_segments, result.get("iq_norm") or [])
    vector = [float(x) for x in vector]

    # 3. 写 ems_iq(标量 + 类别 + 向量)
    row = _build_scalars(result, category)
    row["iq_id"] = str(doc_id)
    row["embedding"] = vector
    await AsyncMilvusClient().upsert_iq([row])

    # 4. md 追加类别 -> 正常入知识库/知识图谱(与其他文本一致)
    graph_summary = None
    if settings.graphrag_enabled:
        param_lines = _build_param_lines(result)
        param_block = ("\n\n【信号参数】\n" + "\n".join(param_lines)) if param_lines else ""
        md_with_category = f"{result['md']}{param_block}\n\n类别:{category}"
        try:
            from ..graphrag.ingest import ingest_document

            graph_summary = await ingest_document(
                doc.kb_id, str(doc_id), md_with_category, "iq", doc.filename
            )
            async with SessionLocal() as db:
                await mark_document_in_kb(db, doc_id)
        except Exception as e:
            logger.warning("IQ 图谱入库失败(doc_id=%s): %s", doc_id, e)

    return IqIngestResponse(
        doc_id=str(doc_id),
        category=category,
        vector_dim=settings.iq_dim,
        graph=graph_summary,
    )


@router.post("/search", response_model=IqSearchResponse)
async def search_iq(payload: IqSearchRequest) -> IqSearchResponse:
    """IQ 判断类型:向量化查询文档 -> 最近邻,返回类型 + 概率(空库返回未知)。"""
    _, content = await _load_doc_h5(payload.doc_id)

    milvus = AsyncMilvusClient()
    if await milvus.iq_count() == 0:
        return IqSearchResponse(status="unknown", type="未知", probability=None)

    vector = await run_in_threadpool(embed_document, content)
    vector = [float(x) for x in vector]

    hits = await milvus.search_iq(
        vector, topk=3, expr=f"iq_id != '{payload.doc_id}'"
    )
    if not hits:
        return IqSearchResponse(status="unknown", type="未知", probability=None)

    top = hits[0]
    return IqSearchResponse(
        status="ok",
        type=top.get("category"),
        probability=float(top.get("_score", 0.0)),
    )
