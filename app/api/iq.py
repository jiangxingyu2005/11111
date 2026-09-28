"""IQ 专属检索库接口:入库打标 + 判断类型(相似度检索)。

独立于现有知识库/图谱集合的 IQ 检索库(ems_iq,256 维),专做 IQ 信号相似度检索:
- POST /iq/ingest: uuid + 类别 -> 向量化写 ems_iq(带标量)+ md 追加类别后正常入图谱
- POST /iq/search: uuid -> 向量化 -> 最近邻,返回类型 + 概率;空库返回未知
"""

import logging
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
        md_with_category = f"{result['md']}\n\n类别:{category}"
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
