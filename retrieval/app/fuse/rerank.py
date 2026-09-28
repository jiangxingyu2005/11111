"""Rerank（M-Q5）：top50 -> top5，threshold 0.35。

real：bge-reranker-v2-m3（TEI/vLLM），POST {query, texts} -> {scores}
fake：本地确定性打分（哈希嵌入余弦映射到 0~1）——只为跑通链路，不得用于评估排序质量

超时/失败 -> RRF 序直出 + rerank_timeout（降级矩阵 #6），排序语义不崩（只丢精排，不丢召回）。
"""
from __future__ import annotations

import logging

from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from .rrf import Hit

log = logging.getLogger("retrieval.fuse.rerank")


def _local_scores(query: str, texts: list[str]) -> list[float]:
    """本地打分（fake 通道）：余弦 [-1,1] -> [0,1]，保证 threshold 语义可用。"""
    from ..store.hash_embed import cosine, hash_embedding
    qv = hash_embedding(query)
    out = []
    for t in texts:
        c = cosine(qv, hash_embedding(t))
        out.append(round(max(0.0, (c + 1.0) / 2.0) ** 3, 6))  # 立方拉开区分度，模拟 reranker 的陡峭分布
    return out


async def rerank(query: str, hits: list[Hit], collector: DegradedCollector) -> list[Hit]:
    cfg = effective_cfg().get("reranker", {})
    top_out = int(cfg.get("top_k_out", 5))
    threshold = float(cfg.get("threshold", 0.35))
    url = get_settings().retrieval_rerank_service_url
    if not hits:
        return []
    try:
        from ..store.fake import guard
        guard("rerank")
        if effective_cfg().get("_backend") == "fake" or not url:
            scores = _local_scores(query, [h.content for h in hits])
        else:
            import httpx
            timeout_ms = effective_cfg().get("timeout", {}).get("reranker_ms", 900)
            async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as cli:
                r = await cli.post(url, json={
                    "query": query,
                    "texts": [h.content for h in hits],
                })
                r.raise_for_status()
                scores = [float(x) for x in r.json()["scores"]]  # 与 texts 同序
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        out = []
        for h, sc in ranked:
            if sc < threshold:
                continue
            h.meta["rerank_score"] = float(sc)
            out.append(h)
        return out[:top_out]
    except Exception as e:
        log.warning("rerank failed: %s", e)
        collector.record(DegradedCode.RERANK_TIMEOUT, str(e)[:200], "rerank")
        return hits[:top_out]  # RRF 序直出
