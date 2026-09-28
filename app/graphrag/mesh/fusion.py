"""GP-07 MESH-02 / MESH-04：兜底绑定 + retired 机制 + 度量门禁（Plan §3.10）。

- 孤立节点检测
- retired 引用计数（摘除最后引用标记、重新引用恢复）
- 看板四指标：孤立率 / BINDS 覆盖率 / 最大连通分量 / 平均度
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


class RetiredRegistry:
    """retired 引用计数（MESH-02）。"""

    def __init__(self):
        self.ref_counts: dict[str, int] = defaultdict(int)
        self.retired: set = set()

    def add_ref(self, entity_id: str) -> None:
        self.ref_counts[entity_id] += 1
        self.retired.discard(entity_id)

    def remove_ref(self, entity_id: str) -> None:
        self.ref_counts[entity_id] -= 1
        if self.ref_counts[entity_id] <= 0:
            self.retired.add(entity_id)

    def is_retired(self, entity_id: str) -> bool:
        return entity_id in self.retired


def detect_orphans(entities: list[dict], binds: list[dict]) -> list[str]:
    """孤立节点：无任何 BINDS 的实体。"""
    bound = {b["entity_id"] for b in binds}
    return [e["entity_id"] for e in entities if e["entity_id"] not in bound]


def _connected_components(relations: list[dict]) -> tuple[int, float]:
    """实体关系图的最大连通分量大小 + 平均度。"""
    adj: dict[str, set] = defaultdict(set)
    nodes: set = set()
    for r in relations:
        h, t = r["head_entity_id"], r["tail_entity_id"]
        adj[h].add(t)
        adj[t].add(h)
        nodes.add(h)
        nodes.add(t)
    if not nodes:
        return 0, 0.0
    visited: set = set()
    max_comp = 0
    for n in nodes:
        if n in visited:
            continue
        stack = [n]
        visited.add(n)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        max_comp = max(max_comp, size)
    avg_degree = sum(len(v) for v in adj.values()) / len(nodes)
    return max_comp, round(avg_degree, 3)


def compute_metrics(entities: list[dict], binds: list[dict],
                    relations: list[dict], parents: list[dict]) -> dict:
    """MESH-04 四指标。"""
    orphan_rate = round(len(detect_orphans(entities, binds)) / len(entities), 4) if entities else 0.0
    bound_parents = {b["parent_id"] for b in binds}
    binds_coverage = round(len(bound_parents) / len(parents), 4) if parents else 0.0
    max_comp, avg_degree = _connected_components(relations)
    return {
        "orphan_rate": orphan_rate,
        "binds_coverage": binds_coverage,
        "max_connected_component": max_comp,
        "avg_degree": avg_degree,
    }
