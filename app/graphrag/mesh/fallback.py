"""GP-12 MESH-02：CDC 通道实体兜底绑定（第二道防线，防孤立星，条款(9)(11)）。

kg_entity_id 为空时不放弃绑定：
  ① 轻量规则链接（注册表别名/指纹匹配，写死规则，不违反"CDC 不上大模型"）
  ② 仍无命中 → 写入 reextract_queue，由离线 GRAPH 定期回炉抽取
"""
from __future__ import annotations

from typing import Optional

from .registry import EntityRegistry


def fallback_entity_link(parent: dict, registry: EntityRegistry) -> Optional[str]:
    """规则链接：别名/指纹双路归一命中现有实体 → 返回 canonical entity_id；否则 None。"""
    name = parent.get("title") or parent.get("keywords")
    props = {}
    for k in ("freq_band", "pulse_width", "prf"):
        if parent.get(k) is not None:
            props[k] = parent[k]
    return registry.resolve(name=name, properties=props or None)


def link_orphan_segments(parents: list[dict], registry: EntityRegistry) -> dict:
    """对 kg_entity_id 为空的父块做兜底绑定。

    返回 {linked: [{parent_id, entity_id}], reextract: [parent_id, ...]}。
    """
    linked: list[dict] = []
    reextract: list[str] = []
    for p in parents:
        if p.get("kg_entity_id"):
            continue
        eid = fallback_entity_link(p, registry)
        if eid:
            linked.append({"parent_id": p["parent_id"], "entity_id": eid})
        else:
            reextract.append(p["parent_id"])
    return {"linked": linked, "reextract": reextract}
