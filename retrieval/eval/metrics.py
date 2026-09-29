"""召回率 + 响应时间指标。

单一事实源：从 scripts/eval_date.py 原样迁移。
"""
from __future__ import annotations

import math

BUDGET_S = 5.2  # 检索总预算（degradation.budget_s）


def metrics(ranked: list[str], gold: dict[str, int], ks: list[int]) -> dict:
    """单条 query 的召回类指标。ranked = 按优劣排序的 child_id 列表。"""
    gold_ids = set(gold)
    core_ids = {c for c, g in gold.items() if g >= 2}
    out = {}
    for k in ks:
        top = ranked[:k]
        hit = set(top) & gold_ids
        out[f"recall@{k}"] = len(hit) / max(1, len(gold_ids))
        out[f"crecall@{k}"] = len(set(top) & core_ids) / max(1, len(core_ids)) if core_ids else 0.0
        out[f"cprec@{k}"] = len(set(top) & core_ids) / k
    mrr = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in gold_ids:
            mrr = 1.0 / i
            break
    out["mrr"] = mrr
    dcg = lambda rs: sum((2 ** gold.get(c, 0) - 1) / math.log2(i + 1)  # noqa: E731
                         for i, c in enumerate(rs, start=1))
    ideal = dcg(sorted(gold, key=lambda x: -gold[x])[:10])
    out["ndcg@10"] = (dcg(ranked[:10]) / ideal) if ideal > 0 else 0.0
    return out


def agg(rows: list[dict]) -> dict:
    """跨 query 取均值（rows 非空且键一致）。"""
    if not rows:
        return {}
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0].keys()}


def latency_stats(ms: list[float], budget_s: float = BUDGET_S) -> dict:
    """响应时间统计：mean/p50/p90/p95/p99/max + 超预算条数。"""
    if not ms:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0,
                "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "over_budget": 0}
    s = sorted(ms)
    n = len(s)
    pct = lambda p: s[min(n - 1, int(n * p))]  # noqa: E731
    over = sum(1 for x in ms if x > budget_s * 1000)
    return {"n": n, "mean_ms": sum(s) / n, "p50_ms": pct(0.50), "p90_ms": pct(0.90),
            "p95_ms": pct(0.95), "p99_ms": pct(0.99), "max_ms": s[-1],
            "over_budget": over}
