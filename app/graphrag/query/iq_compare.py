"""IQ 对比意图:有待测 IQ 才查 ems_iq,产出相似信号表 / 一致度 / 时频图。

意图只决定走不走这条路;查询向量必须来自一份 IQ(WavesFM 256 维),
不会把问句文本向量丢进 ems_iq(维数、空间都不同)。
"""
from __future__ import annotations

import re
from typing import Optional
from uuid import UUID

_IQ_WORDS = ("相似信号", "波形比对", "iq比对", "iq 比对", "h5", "待测信号", "信号相似")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CHART_KEYS = ("time_domain", "spectrogram", "psd")   # 时域 / 时频 / 频谱
_TOPK = 4


def resolve_iq_doc_id(query: str, iq_doc_id=None) -> Optional[str]:
    """显式 iq_doc_id 优先;否则问句同时有 IQ 意图词和文档 UUID 才走 IQ。"""
    if iq_doc_id:
        return str(iq_doc_id)
    q = (query or "").lower()
    if not any(w in q for w in _IQ_WORDS):
        return None
    m = _UUID_RE.search(query or "")
    return m.group(0) if m else None


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and x not in (float("inf"), float("-inf")) else None


def _row(h: dict) -> dict:
    freq = _f(h.get("frequency_mean"))
    if freq is None:
        freq = _f(h.get("psd_peak_frequency"))
    return {
        "id": h.get("iq_id"),
        "name": h.get("category") or h.get("iq_id"),
        "category": h.get("category") or "",
        "frequency_hz": freq,
        "pri_hz": _f(h.get("pri_hz")),
        "pulse_width_us": _f(h.get("pulse_width_us")),
        "modulation": h.get("modulation") or "",
        "snr_db": _f(h.get("snr")),
        "source": "ems_iq",
        "score": round(float(h.get("_score") or 0.0), 4),
    }


def _known_mod(m: str) -> str:
    m = str(m or "").strip()
    return "" if m.lower() in ("", "unknown", "未知", "未知调制") else m


def _agree(query_flat: dict, query_mod: str, top: dict) -> tuple[Optional[float], list[str]]:
    """待测 vs Top-1 参数一致度:可比项里一致的比例。

    调制为 Unknown 不参与比较(两边都未知不能算一致);主频 ±5% 内算一致。
    返回 (分数, 参与比较的项)。没有可比项时分数为 None,前端灰掉。
    """
    same, used = 0, []
    qm, tm = _known_mod(query_mod), _known_mod(top.get("modulation"))
    if qm and tm:
        used.append("modulation")
        same += int(qm.lower() == tm.lower())
    qf, tf = _f(query_flat.get("frequency_mean")), top.get("frequency_hz")
    if qf is not None and tf is not None:
        used.append("frequency")
        same += int(abs(qf - tf) <= 0.05 * max(abs(qf), abs(tf), 1e-9))
    return (round(same / len(used), 4) if used else None), used


async def compare_iq(milvus, doc_id: str, with_charts: bool = True) -> dict:
    """待测 IQ → WavesFM → ems_iq 近邻。任何一步失败都返回空结果 + status,不让问答崩。

    with_charts=True 时额外处理 Top-1 样例的 H5,返回待测+样例双线对比图数据。
    图表数据全部来自 process_iq_h5 真实计算,不是假数据。
    """
    from starlette.concurrency import run_in_threadpool

    from ...utils.database import SessionLocal
    from ...utils.document import get_document
    from ...utils.iq import process_iq_h5
    from ...utils.iq_embedding import embed_segments
    from ...utils.minio_client import download_source

    out = {"status": "ok", "doc_id": doc_id, "similar_signals": [],
           "signal_agree": None, "agree_fields": [], "iq_query": None,
           "iq_charts": None, "sample_charts": None}
    try:
        if await milvus.iq_count() == 0:
            return {**out, "status": "empty_library"}
        async with SessionLocal() as db:
            doc = await get_document(db, UUID(doc_id))
        if doc is None:
            return {**out, "status": "doc_not_found"}
        content = await run_in_threadpool(download_source, doc.kb_id, doc.id, doc.filename)
        result = await run_in_threadpool(process_iq_h5, content)
        vector = await run_in_threadpool(embed_segments, result.get("iq_norm") or [])
        hits = await milvus.search_iq([float(x) for x in vector], topk=_TOPK,
                                      expr=f"iq_id != '{doc_id}'")
    except Exception as e:
        return {**out, "status": f"failed: {e}"[:200]}

    rows = [_row(h) for h in hits]
    flat = result.get("flat") or {}
    query_mod = (result.get("param_guess") or {}).get("modulation") or flat.get("modulation") or ""
    out["similar_signals"] = rows
    out["iq_query"] = {"doc_id": doc_id, "filename": doc.filename, "modulation": query_mod,
                       "frequency_hz": _f(flat.get("frequency_mean")),
                       "snr_db": _f((result.get("snr") or {}).get("mean_snr"))}
    if rows:
        out["signal_agree"], out["agree_fields"] = _agree(flat, query_mod, rows[0])

    if with_charts:
        # 待测信号图表(真实计算)
        q_charts = result.get("charts") or {}
        out["iq_charts"] = {k: q_charts.get(k) for k in _CHART_KEYS if k in q_charts}

        # Top-1 样例图表(从 MinIO 取 H5 → process_iq_h5 真实计算)
        if rows:
            try:
                sample_id = rows[0].get("id")
                async with SessionLocal() as db:
                    sdoc = await get_document(db, UUID(sample_id))
                if sdoc is not None:
                    s_content = await run_in_threadpool(
                        download_source, sdoc.kb_id, sdoc.id, sdoc.filename)
                    s_result = await run_in_threadpool(process_iq_h5, s_content)
                    s_charts = s_result.get("charts") or {}
                    out["sample_charts"] = {
                        "doc_id": sample_id,
                        "name": rows[0].get("name") or sample_id,
                        **{k: s_charts.get(k) for k in _CHART_KEYS if k in s_charts}
                    }
            except Exception:
                out["sample_charts"] = None   # 样例处理失败不影响待测图表
    return out
