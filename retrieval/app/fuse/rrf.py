"""外层 RRF 融合（M-Q4）——纯算法，绿档模块，完整实现。

纪律：
  - 1 source 1 vote（G-QC-02）：同一来源多路命中先组内合并，跨来源才进 RRF
  - RRF score = Σ 1 / (k + rank)，rank 从 1 起
  - 单路被剥离不影响其他路相对序（这是选 RRF 而非加权求和的原因）
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    """一路召回的单个命中。source 取值：vector_hybrid / bm25 / graph_local / graph_global。"""

    child_id: str
    parent_id: str = ""
    rank_score: float = 0.0      # 该来源内部原始分（仅展示用，不参与融合）
    source: str = ""
    content: str = ""            # Milvus 行内冗余（PG 熔断兜底用）
    kg_entity_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def merge_within_source(hits: list[Hit]) -> list[Hit]:
    """同一来源内部按 child_id 去重：保留该来源内最高 rank（rank_score 最大）。"""
    best: dict[str, Hit] = {}
    for h in hits:
        cur = best.get(h.child_id)
        if cur is None or h.rank_score > cur.rank_score:
            best[h.child_id] = h
    return sorted(best.values(), key=lambda h: h.rank_score, reverse=True)


def rrf_fuse(
    sources: dict[str, list[Hit]],
    k: int = 60,
    top_k: int = 50,
) -> list[Hit]:
    """跨来源 RRF 融合。

    sources: {source_name: [Hit, ...]（按该来源优劣排序）}
    返回：融合后 Hit 列表，meta 里带 rrf_score 与 voted_by（命中来源清单）。
    """
    k = int(k)
    agg: dict[str, dict] = {}
    for src_name, hits in sources.items():
        if not hits:
            continue  # 被剥离的路传空表/缺席即可，天然不投票
        merged = merge_within_source(hits)
        for rank, h in enumerate(merged, start=1):
            slot = agg.setdefault(
                h.child_id,
                {"hit": h, "score": 0.0, "voted_by": []},
            )
            slot["score"] += 1.0 / (k + rank)  # 1 source 1 vote
            slot["voted_by"].append(src_name)
    out: list[Hit] = []
    for slot in agg.values():
        h = slot["hit"]
        h.meta["rrf_score"] = slot["score"]
        h.meta["voted_by"] = slot["voted_by"]
        out.append(h)
    out.sort(key=lambda h: h.meta["rrf_score"], reverse=True)
    return out[:top_k]


def assert_one_source_one_vote(sources: dict[str, list[Hit]]) -> None:
    """G-QC-02 断言辅助：每来源内部不得有重复 child_id（融合前调用）。"""
    for name, hits in sources.items():
        ids = [h.child_id for h in hits]
        dup = {i for i in ids if ids.count(i) > 1}
        assert not dup, f"source {name} has duplicated child_id before merge: {dup}"
