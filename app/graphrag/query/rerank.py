"""GP-12 QRY-04:双路合并重排序(D10:A路加实体命中权重分,不做硬过滤)。

A路(实体锚定,kg_entity_ids 非空)命中查询实体 → 加权重分;B路(纯向量兜底)不加。
合并按得分降序、child_id 去重,截取 topk。纯逻辑,无 I/O,保持同步。
"""
from __future__ import annotations

from typing import Optional

from ..config import get


def rerank(hits_a: list[dict], hits_b: list[dict],
           matched_ids: Optional[list] = None,
           topk: Optional[int] = None,
           entity_boost_weight: Optional[float] = None) -> list[dict]:
    matched = set(matched_ids or [])
    boost = (entity_boost_weight if entity_boost_weight is not None
             else float(get("query.entity_boost_weight", 0.15)))
    topk = topk or int(get("query.rerank_topk", 5))

    merged: dict[str, dict] = {}
    for h in hits_a + hits_b:
        cid = h.get("child_id")
        if not cid:
            continue
        h = dict(h)
        score = float(h.get("_score", 0.0))
        # A路实体锚定:命中查询实体 → 加权重分(D10)
        if matched and any(eid in matched for eid in (h.get("kg_entity_ids") or [])):
            score += boost
        h["_score"] = score
        if cid not in merged or score > float(merged[cid].get("_score", 0.0)):
            merged[cid] = h

    ranked = sorted(merged.values(), key=lambda x: float(x.get("_score", 0.0)), reverse=True)
    return ranked[:topk]
