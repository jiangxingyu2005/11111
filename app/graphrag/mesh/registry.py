"""GP-07 MESH-01：实体注册表（别名 + 属性指纹双路归一化，Plan §3.10）。

- 别名路：同一辐射源两种写法（别名）归一为同一 entity_id
- 指纹路：属性（频段+脉宽+重频，容差内）命中现有实体不新发号
"""
from __future__ import annotations

from typing import Optional

from ..ids import normalize_name


def _round_tol(value: float, tol: float) -> str:
    return f"{round(float(value) / tol) * tol:.3f}"


def _norm_freq(freq: str) -> str:
    return str(freq).strip().lower().replace(" ", "")


def fingerprint(properties: dict) -> Optional[str]:
    """属性指纹：频段 + 脉宽 + 重频（容差内取整）。缺字段则为 None。"""
    if not properties:
        return None
    parts: list[str] = []
    freq = properties.get("freq_band") or properties.get("频率") or properties.get("value")
    pw = properties.get("pulse_width") or properties.get("脉宽")
    prf = properties.get("prf") or properties.get("重频")
    if freq:
        parts.append("f:" + _norm_freq(str(freq)))
    if pw is not None:
        parts.append("pw:" + _round_tol(float(pw), 0.5))
    if prf is not None:
        parts.append("prf:" + _round_tol(float(prf), 100.0))
    return "|".join(parts) if parts else None


class EntityRegistry:
    """实体注册表：别名索引 + 指纹索引双路归一。"""

    def __init__(self, entities: Optional[list[dict]] = None):
        self.by_id: dict[str, dict] = {}
        self.alias_index: dict[str, str] = {}       # 归一别名 -> entity_id
        self.fingerprint_index: dict[str, str] = {}  # 指纹 -> entity_id
        for e in (entities or []):
            self.register(e)

    def register(self, entity: dict) -> None:
        eid = entity["entity_id"]
        self.by_id[eid] = entity
        for n in [entity.get("name")] + list(entity.get("aliases") or []):
            key = normalize_name(n)
            self.alias_index.setdefault(key, eid)   # 首次注册者为准
        fp = fingerprint(entity.get("properties") or {})
        if fp:
            self.fingerprint_index.setdefault(fp, eid)

    def resolve(self, name: Optional[str] = None, properties: Optional[dict] = None) -> Optional[str]:
        """双路归一：先别名、后指纹。返回 canonical entity_id 或 None。"""
        if name and normalize_name(name) in self.alias_index:
            return self.alias_index[normalize_name(name)]
        if properties:
            fp = fingerprint(properties)
            if fp and fp in self.fingerprint_index:
                return self.fingerprint_index[fp]
        return None


def normalize_entities(entities: list[dict]) -> list[dict]:
    """把别名/指纹重复的实体归一到 canonical entity_id。

    同一辐射源两种写法 -> 同一 entity_id（不新发号）。
    """
    reg = EntityRegistry(entities)
    canonical: dict[str, str] = {}
    for e in entities:
        canonical[e["entity_id"]] = reg.resolve(name=e["name"],
                                                properties=e.get("properties")) or e["entity_id"]
    merged: dict[str, dict] = {}
    order: list[str] = []
    for e in entities:
        cid = canonical[e["entity_id"]]
        if cid not in merged:
            merged[cid] = dict(e)
            merged[cid]["entity_id"] = cid
            merged[cid]["aliases"] = list(e.get("aliases") or [])
            order.append(cid)
        else:
            for a in list(e.get("aliases") or []) + [e["name"]]:
                if a not in merged[cid]["aliases"]:
                    merged[cid]["aliases"].append(a)
    return [merged[cid] for cid in order]
