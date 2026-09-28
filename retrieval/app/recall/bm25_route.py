"""② 路外置 BM25（M-Q3 备选路）——开关默认关（消融 D2 裁决前）。

若 F2 最终砍 sparse 股，则本路提前扶正（D2 决策提前）。
real = httpx 调 RETRIEVAL_BM25_URL；fake = FakeStore.bm25_search（稀疏内积等价）。
异常 -> 空列表 + bm25_unavailable（降级矩阵 #18）。
"""
from __future__ import annotations

import logging

from ..config import get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from ..fuse.rrf import Hit

log = logging.getLogger("retrieval.recall.bm25")


async def recall(query: str, *, scope: list[str], collector: DegradedCollector) -> list[Hit]:
    s = get_settings()
    if not s.retrieval_bm25_enabled:
        return []  # 未启用不算降级（正常缺席，1 source 1 vote 自然不投票）
    try:
        if s.retrieval_backend == "fake":
            from ..store.fake import get_fake_store
            rows = get_fake_store().bm25_search(query, top_k=100, scope=scope)
            return [
                Hit(child_id=str(r["child_id"]), parent_id=str(r.get("parent_id", "")),
                    rank_score=float(r.get("score", 0.0)), source="bm25",
                    content=r.get("content", ""), kg_entity_ids=r.get("kg_entity_ids", []),
                    meta={"kb_id": r.get("kb_id", ""), "span": r.get("span")})
                for r in rows
            ]
        import httpx
        async with httpx.AsyncClient(timeout=1.0) as cli:
            r = await cli.post(s.retrieval_bm25_url, json={"query": query, "scope": scope, "top_k": 100})
            r.raise_for_status()
            return [
                Hit(child_id=str(x["child_id"]), parent_id=str(x.get("parent_id", "")),
                    rank_score=float(x["score"]), source="bm25",
                    content=x.get("content", ""))
                for x in r.json().get("hits", [])
            ]
    except Exception as e:
        log.warning("bm25 route failed: %s", e)
        collector.record(DegradedCode.BM25_UNAVAILABLE, str(e)[:200], "recall_bm25")
        return []
