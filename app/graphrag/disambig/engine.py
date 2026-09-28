"""三阶段匹配器 + 入库对齐 + 全图批量对账(纯逻辑,依赖 stores.py 的内存替身)。

匹配管道(命中即返回,不往后走):
  ① 确定性规则:名称+类型精确 / 别名 / 属性指纹(类型作用域)
  ② 向量相似度:实体名向量 cosine ≥ sim_high 归并(类型作用域)
  ③ 大模型灰区:sim_low ≤ score < sim_high 时交 judge(默认 NeverJudge,不并)

集成时只换 stores / embedder / judge,本模块逻辑不变。
"""

from __future__ import annotations

from typing import Callable, Optional

from ..ids import normalize_name
from ..mesh.registry import fingerprint

from .dsu import DSU
from .stores import (
    DocIndexStore, EntityRecord, EntityRegistryStore, EntityVectorIndex, GraphStore,
)


class ResolverConfig:
    def __init__(self, *, fingerprint_enabled: bool = True, vector_enabled: bool = True,
                 sim_high: float = 0.95, sim_low: float = 0.85, topk: int = 5,
                 llm_gate: bool = False) -> None:
        self.fingerprint_enabled = fingerprint_enabled
        self.vector_enabled = vector_enabled
        self.sim_high = sim_high
        self.sim_low = sim_low
        self.topk = topk
        self.llm_gate = llm_gate


class NeverJudge:
    """mock/默认裁判:灰区一律不并(确定性优先)。"""

    def judge_same(self, name_a: str, name_b: str, entity_type: str) -> bool:
        return False


def resolve_one(cfg: ResolverConfig, registry: EntityRegistryStore,
                vector_index: Optional[EntityVectorIndex],
                embedder: Optional[Callable[[str], list]],
                judge, name: str, entity_type: str, aliases: list,
                properties: dict) -> Optional[str]:
    # ① 名称/别名(类型作用域):name 与每个 alias 都做 lookup
    keys: list[str] = []
    for a in [name] + list(aliases):
        k = normalize_name(a)
        if k and k not in keys:
            keys.append(k)
    for key in keys:
        hit = registry.lookup_name_or_alias(entity_type, key)
        if hit:
            return hit
    # ① 属性指纹(类型作用域)
    if cfg.fingerprint_enabled:
        fp = fingerprint(properties)
        if fp:
            hit = registry.lookup_fingerprint(entity_type, fp)
            if hit:
                return hit
    # ② 向量相似度
    if cfg.vector_enabled and vector_index is not None and embedder is not None:
        vec = embedder(name)
        top = vector_index.search(vec, entity_type, cfg.topk)
        if top:
            eid, cname, score = top[0]
            if score >= cfg.sim_high:
                return eid
            # ③ 大模型灰区
            if score >= cfg.sim_low and cfg.llm_gate and judge is not None:
                if judge.judge_same(name, cname, entity_type):
                    return eid
    return None


def align(cfg, registry, vector_index, embedder, judge, entities, relations):
    """把新实体对齐到全局 canonical:返回 (merged_records, remapped_relations, remap)。

    entities: list[dict],键 entity_id/name/entity_type/aliases/properties。
    relations: list[(head, tail, rel_type)]。
    """
    remap: dict[str, str] = {}
    for e in entities:
        resolved = resolve_one(cfg, registry, vector_index, embedder, judge,
                               e["name"], e["entity_type"],
                               e.get("aliases") or [], e.get("properties") or {})
        remap[e["entity_id"]] = resolved or e["entity_id"]

    merged_records = _merge(entities, remap)
    remapped_relations = _remap_relations(relations, remap)
    return merged_records, remapped_relations, remap


def persist(registry, vector_index, embedder, merged_records) -> None:
    """canonical 实体写注册表 + 写实体向量索引(幂等)。"""
    for r in merged_records:
        registry.upsert(r)
        if vector_index is not None and embedder is not None:
            vector_index.upsert(r.entity_id, r.name, r.entity_type, embedder(r.name))


def global_resolve(cfg, registry, vector_index, embedder, judge,
                   graph: GraphStore, doc_index: DocIndexStore,
                   dry_run: bool = False) -> dict:
    """全图批量对账:DSU 求 canonical + 四层回填,顺序无关、幂等。"""
    entries = sorted(registry.all_entries(), key=lambda r: (r.created_order, r.entity_id))
    order_map = {r.entity_id: r.created_order for r in entries}

    dsu = DSU()
    for r in entries:
        dsu.add(r.entity_id)

    # ① 确定性键 union(类型作用域)
    by_name: dict[tuple, list] = {}
    by_fp: dict[tuple, list] = {}
    for r in entries:
        if r.status != "active":
            continue
        for a in [r.name] + list(r.aliases):
            by_name.setdefault((r.entity_type, a), []).append(r.entity_id)
        if r.fingerprint:
            by_fp.setdefault((r.entity_type, r.fingerprint), []).append(r.entity_id)
    for group in by_name.values():
        _union_group(dsu, group)
    for group in by_fp.values():
        _union_group(dsu, group)

    # ② 向量命中 union(≥sim_high 并;灰区走 judge,默认关)
    if cfg.vector_enabled and vector_index is not None and embedder is not None:
        for r in entries:
            if r.status != "active":
                continue
            got = vector_index.get(r.entity_id)
            if got is None:
                continue
            vec = got[2]
            for (eid, cname, score) in vector_index.search(vec, r.entity_type, cfg.topk):
                if eid == r.entity_id:
                    continue
                if score >= cfg.sim_high:
                    dsu.union(r.entity_id, eid)
                elif score >= cfg.sim_low and cfg.llm_gate and judge is not None:
                    if judge.judge_same(r.name, cname, r.entity_type):
                        dsu.union(r.entity_id, eid)

    # canonical = 组内最早(created_order, entity_id)
    remap: dict[str, str] = {}
    for members in dsu.components().values():
        members.sort(key=lambda eid: (order_map.get(eid, 0), eid))
        canon = members[0]
        for m in members[1:]:
            remap[m] = canon

    if dry_run:
        return {
            "dry_run": True,
            "merged": [
                {"old_id": o, "new_id": n, "name": registry.entries[o].name}
                for o, n in remap.items()
            ],
        }

    report = {"merged": [], "binds_moved": 0, "relations_moved": 0,
              "parents_updated": 0, "chunks_updated": 0}
    for old, new in remap.items():
        old_r = registry.entries.get(old)
        name = old_r.name if old_r else ""
        registry.mark_merged(old, new)
        _absorb(registry, old, new)
        report["binds_moved"] += graph.move_binds(old, new)
        report["relations_moved"] += graph.move_relations(old, new)
        graph.delete_entity(old)
        p, c = doc_index.remap(old, new)
        report["parents_updated"] += p
        report["chunks_updated"] += c
        if vector_index is not None:
            vector_index.delete(old)
        report["merged"].append({"old_id": old, "new_id": new, "name": name})
    return report


# ---- 内部纯函数 ----

def _merge(entities, remap) -> list[EntityRecord]:
    merged: dict[str, EntityRecord] = {}
    order: list[str] = []
    for e in entities:
        cid = remap.get(e["entity_id"], e["entity_id"])
        if cid not in merged:
            merged[cid] = EntityRecord(
                entity_id=cid, name=normalize_name(e["name"]),
                entity_type=e["entity_type"], aliases=[],
                properties=dict(e.get("properties") or {}),
                fingerprint=fingerprint(e.get("properties") or {}),
            )
            order.append(cid)
        m = merged[cid]
        for a in [e["name"]] + list(e.get("aliases") or []):
            ak = normalize_name(a)
            if ak and ak != m.name and ak not in m.aliases:
                m.aliases.append(ak)
        for k, v in (e.get("properties") or {}).items():
            m.properties.setdefault(k, v)
        if m.fingerprint is None:
            m.fingerprint = fingerprint(e.get("properties") or {})
    return [merged[c] for c in order]


def _remap_relations(relations, remap) -> list[tuple]:
    out = []
    for (h, t, rt) in relations:
        hh = remap.get(h, h)
        tt = remap.get(t, t)
        if hh == tt:
            continue
        out.append((hh, tt, rt))
    return out


def _union_group(dsu: DSU, group: list) -> None:
    for i in range(1, len(group)):
        dsu.union(group[0], group[i])


def _absorb(registry: EntityRegistryStore, old: str, new: str) -> None:
    old_r = registry.entries.get(old)
    new_r = registry.entries.get(new)
    if not old_r or not new_r:
        return
    for a in [old_r.name] + list(old_r.aliases):
        ak = normalize_name(a)
        if ak and ak != new_r.name and ak not in new_r.aliases:
            new_r.aliases.append(ak)
    if new_r.fingerprint is None:
        new_r.fingerprint = old_r.fingerprint
    for k, v in old_r.properties.items():
        new_r.properties.setdefault(k, v)
