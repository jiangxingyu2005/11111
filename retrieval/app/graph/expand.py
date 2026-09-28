"""G4 图扩展 + graph_ref（M-R4）骨架。

focus = 重排 Top-5 命中实体（kg_entity_ids）；1 跳默认；0.6s 预算；超时 -> degraded 不阻塞。
产出喂 /api/graph/subgraph（独立图谱页，M-R5）。
"""
from __future__ import annotations

from ..fuse.rrf import Hit

EXPAND_BUDGET_S = 0.6


def focus_entities(top_hits: list[Hit], top_n: int = 5) -> list[str]:
    """从重排后 Top-N 命中的 kg_entity_ids 提取 focus 实体（保序去重）。"""
    seen: list[str] = []
    for h in top_hits[:top_n]:
        for eid in h.kg_entity_ids:
            if eid and eid not in seen:
                seen.append(eid)
    return seen
