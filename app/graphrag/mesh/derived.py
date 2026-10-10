"""GP-12 MESH-03：派生关系织网（第三道防线，文本边稀疏时保底连通，条款(7)(15)）。

从实体属性派生结构边（不依赖文本叙述，不进 LLM 抽取）：
  SAME_BAND  频段重合（条款7「信号频段与知识库记录重合」规则的输入）
  CO_TIME    时间共现（条款15 时空关联）
  CO_LOCATE  活动区域相同（条款15 时空关联）
派生边带 derived=true 标记，与抽取边区分，可整体重算。
DERIVED_RELATIONS 是派生关系的唯一定义，seed 把它登记进字典（category=derived）。
"""
from __future__ import annotations

import re

from ..ids import canonical_relation_id

_SIGNAL_TYPES = ["辐射源", "原始信号", "信号样本", "信号簇", "平台"]
_SPATIOTEMPORAL_TYPES = ["辐射源", "平台", "事件", "目标", "组织", "武器"]

DERIVED_RELATIONS = [
    {"name": "同频段", "en": "SAME_BAND",
     "description": "规则派生:两实体的频率/频段属性区间重合(条款7 频段重合规则输入)",
     "head_types": _SIGNAL_TYPES, "tail_types": _SIGNAL_TYPES},
    {"name": "同时段", "en": "CO_TIME",
     "description": "规则派生:两实体的时间属性相同(条款15 时空关联)",
     "head_types": _SPATIOTEMPORAL_TYPES, "tail_types": _SPATIOTEMPORAL_TYPES},
    {"name": "同区域", "en": "CO_LOCATE",
     "description": "规则派生:两实体的活动区域属性相同(条款15 时空关联)",
     "head_types": _SPATIOTEMPORAL_TYPES, "tail_types": _SPATIOTEMPORAL_TYPES},
]

_TYPES = {d["en"]: set(d["head_types"]) for d in DERIVED_RELATIONS}

_UNIT_HZ = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}
# 单值或区间:2.4GHz / 2.4 GHz / 8-12GHz / 8~12 GHz / 850MHz
_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-~～至到]\s*(\d+(?:\.\d+)?))?\s*(ghz|mhz|khz|hz)",
                      re.IGNORECASE)


def _freq_interval(freq) -> tuple[float, float] | None:
    """频率/频段字符串 -> (下限Hz, 上限Hz);解析不出返回 None。"""
    m = _FREQ_RE.search(str(freq or ""))
    if not m:
        return None
    unit = _UNIT_HZ[m.group(3).lower()]
    lo = float(m.group(1)) * unit
    hi = float(m.group(2)) * unit if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def _freq_value(properties: dict):
    if not properties:
        return None
    return (properties.get("频段") or properties.get("频率")
            or properties.get("freq_band") or properties.get("frequency"))


def _derived(a: dict, b: dict, rel_type: str) -> dict:
    rid = canonical_relation_id(a["entity_id"], b["entity_id"], rel_type, "derived")
    return {
        "relation_id": rid,
        "head_entity_id": a["entity_id"],
        "tail_entity_id": b["entity_id"],
        "rel_type": rel_type,
        "confidence": 1.0,
        "evidence": "规则派生",
        "source_parent_id": None,
        "derived": True,
    }


def _pairwise(members: list, rel_type: str, out: list) -> None:
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            out.append(_derived(a, b, rel_type))


def derive_relations(entities: list[dict]) -> list[dict]:
    """派生 SAME_BAND / CO_TIME / CO_LOCATE 边(只在该派生关系允许的实体类型之间)。"""
    out: list[dict] = []

    banded: list[tuple[dict, tuple[float, float]]] = []
    time_groups: dict[str, list] = {}
    area_groups: dict[str, list] = {}
    for e in entities:
        etype = e.get("entity_type")
        props = e.get("properties") or {}
        if etype in _TYPES["SAME_BAND"]:
            iv = _freq_interval(_freq_value(props))
            if iv:
                banded.append((e, iv))
        if etype in _TYPES["CO_TIME"]:
            t = e.get("event_time") or props.get("时间")
            if t:
                time_groups.setdefault(str(t), []).append(e)
        if etype in _TYPES["CO_LOCATE"]:
            area = e.get("area") or props.get("area") or props.get("区域")
            if area:
                area_groups.setdefault(str(area), []).append(e)

    # 频段区间重合(含端点相接):2.4GHz 与 2-3GHz 重合,与 850MHz 不重合
    for i, (a, (alo, ahi)) in enumerate(banded):
        for b, (blo, bhi) in banded[i + 1:]:
            if alo <= bhi and blo <= ahi:
                out.append(_derived(a, b, "SAME_BAND"))
    for members in time_groups.values():
        _pairwise(members, "CO_TIME", out)
    for members in area_groups.values():
        _pairwise(members, "CO_LOCATE", out)
    return out
