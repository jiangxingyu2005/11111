"""GP-12 MESH-03：派生关系织网（第三道防线，文本边稀疏时保底连通，条款(7)(15)）。

从元数据派生结构边（不依赖文本叙述）：
  SAME_BAND  频段重叠的辐射源互联
  CO_TIME    时间窗共现（条款15 时空关联）
  CO_LOCATE  活动区域相交
派生边带 derived=true 标记，与抽取边区分，可整体重算。
"""
from __future__ import annotations

from ..ids import canonical_relation_id


def _band(freq) -> str | None:
    """频段归一：2.4GHz/2.45GHz → 2.4ghz/2.45ghz（小写、去空白）；非频段字符串返回 None。"""
    s = str(freq or "").strip().lower().replace(" ", "")
    if not s:
        return None
    for unit in ("ghz", "mhz", "khz"):
        if unit in s:
            return s
    return None


def _freq_value(properties: dict):
    if not properties:
        return None
    return (properties.get("freq_band") or properties.get("frequency")
            or properties.get("value") or properties.get("频率"))


def _derived(a: dict, b: dict, rel_type: str) -> dict:
    rid = canonical_relation_id(a["entity_id"], b["entity_id"], rel_type, "derived")
    return {
        "relation_id": rid,
        "head_entity_id": a["entity_id"],
        "tail_entity_id": b["entity_id"],
        "rel_type": rel_type,
        "confidence": 1.0,
        "evidence": "元数据派生",
        "source_parent_id": None,
        "derived": True,
    }


def _pairwise(members: list, rel_type: str, out: list) -> None:
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            out.append(_derived(a, b, rel_type))


def derive_relations(entities: list[dict]) -> list[dict]:
    """派生 SAME_BAND / CO_TIME / CO_LOCATE 边。"""
    out: list[dict] = []

    band_groups: dict[str, list] = {}
    time_groups: dict[str, list] = {}
    area_groups: dict[str, list] = {}
    for e in entities:
        props = e.get("properties") or {}
        band = _band(_freq_value(props))
        if band:
            band_groups.setdefault(band, []).append(e)
        t = e.get("event_time")
        if t:
            time_groups.setdefault(str(t), []).append(e)
        area = e.get("area") or props.get("area") or props.get("区域")
        if area:
            area_groups.setdefault(str(area), []).append(e)

    for members in band_groups.values():
        _pairwise(members, "SAME_BAND", out)
    for members in time_groups.values():
        _pairwise(members, "CO_TIME", out)
    for members in area_groups.values():
        _pairwise(members, "CO_LOCATE", out)
    return out
