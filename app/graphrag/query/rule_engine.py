"""GP-12 QRY-03:规则引擎(条款(7):预置规则评估)。

规则外置可扩展:每条规则 = (id, 谓词函数)。
预置规则:R_50KM_HIGH_PRIORITY —— 候选分段方位角区间覆盖锚点方位角(50km 演示口径)→ 标记高优先级。
纯逻辑,无 I/O,保持同步(ems 里仅测试引用,可选接入)。
"""
from __future__ import annotations

from typing import Optional


def _azimuth_in_range(azimuth_range, anchor_azimuth: float) -> bool:
    """方位角区间 "045-075" 是否覆盖 anchor_azimuth。"""
    rng = str(azimuth_range or "")
    if "-" not in rng:
        return False
    lo_s, hi_s = rng.split("-", 1)
    try:
        return float(lo_s) <= float(anchor_azimuth) <= float(hi_s)
    except ValueError:
        return False


def rule_50km(candidate: dict, anchor: dict) -> bool:
    """50km 规则:候选方位角区间覆盖锚点方位角 → 视为 50km 内高优先级。"""
    az = anchor.get("azimuth")
    if az is None:
        return False
    return _azimuth_in_range(candidate.get("azimuth_range"), float(az))


_RULES = [
    ("R_50KM_HIGH_PRIORITY", rule_50km),
]


def eval_rules(candidates: list[dict], anchor: dict,
               rules: Optional[list] = None) -> list[str]:
    """对候选集做规则评估,返回命中标记(候选 segment_id/parent_id/child_id 列表)。"""
    rules = rules or _RULES
    marked: list[str] = []
    for c in candidates:
        if any(pred(c, anchor) for _, pred in rules):
            key = c.get("segment_id") or c.get("parent_id") or c.get("child_id")
            if key:
                marked.append(key)
    return marked
