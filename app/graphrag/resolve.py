"""实体消歧:接到真实 PG/Milvus/Neo4j。

- resolve_deterministic:名称/别名/指纹查 entity_registry(类型作用域)。
- resolve_vector:向量只找候选;属性冲突否决;LLM 双向判同且置信度够才合并,否则记疑似。
- align:抽取结果对齐到全局 canonical(原地 remap entity_id + 关系 head/tail,自环丢弃)。
- persist:canonical 实体写回 entity_registry + ems_entities(幂等,name 首现者赢、aliases 并集)。
- global_resolve:全图批量对账(DSU 求 canonical + 回填 PG/Neo4j/Milvus)。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import EntityRegistryEntry, ParentDocument, PendingExtraction
from . import db
from . import dict as dict_store
from .disambig.dsu import DSU
from .countries import canon as _country_canon
from .ids import normalize_name
from .mesh.registry import fingerprint
from .models import Entity, Relation


def _dedup(items: list) -> list:
    out: list = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


async def _load_mergeable(session: AsyncSession, milvus, embeddings) -> Optional[set]:
    """取可向量化的实体类型集合(字典 semantic_mergeable=true)。不可用/未启用返回 None=不限制。"""
    if not settings.graphrag_resolve_vector_enabled or milvus is None or embeddings is None:
        return None
    try:
        return await dict_store.load_mergeable_types(session)
    except Exception:
        return None  # 字典查询失败降级为不限制,消歧链路不因字典故障崩


async def resolve_deterministic(session: AsyncSession, name: str, entity_type: str,
                                aliases: list, properties: dict) -> Optional[str]:
    """匹配器①:名称/别名/指纹(类型作用域),命中返回 canonical entity_id。"""
    keys: list[str] = []
    for a in [name] + list(aliases or []):
        k = normalize_name(a)
        if k and k not in keys:
            keys.append(k)
    # 名称精确(取最早)
    for key in keys:
        rows = await db.select_where(session, EntityRegistryEntry,
                                     entity_type=entity_type, name=key, status="active")
        if rows:
            return _earliest(rows)["entity_id"]
    # 别名(JSONB @> [key])
    for key in keys:
        from sqlalchemy import select
        stmt = (
            select(EntityRegistryEntry)
            .where(EntityRegistryEntry.entity_type == entity_type,
                   EntityRegistryEntry.status == "active",
                   EntityRegistryEntry.aliases.contains([key]))
            .order_by(EntityRegistryEntry.created_at.asc(), EntityRegistryEntry.entity_id.asc())
        )
        res = await session.execute(stmt)
        hit = res.scalars().first()
        if hit is not None:
            return hit.entity_id
    # 指纹(取最早)
    if settings.graphrag_resolve_fingerprint:
        fp = fingerprint(properties)
        if fp:
            rows = await db.select_where(session, EntityRegistryEntry,
                                         entity_type=entity_type, fingerprint=fp, status="active")
            if rows:
                return _earliest(rows)["entity_id"]
    return None


_judge_cache: dict[tuple, dict] = {}


def _veto(props_a: dict, codes_a: set, props_b: dict, codes_b: set) -> Optional[str]:
    """第 ② 层:属性冲突否决。两边都有值且能比较时才否决;返回冲突原因或 None。

    不比粒度:LLM 填的粒度经常不准,当硬否决会挡住真同一。
    """
    from .mesh.derived import _freq_interval
    ca = _country_canon((props_a or {}).get("国别", ""))
    cb = _country_canon((props_b or {}).get("国别", ""))
    if ca and cb and ca != cb:
        return f"国别不同({ca}/{cb})"
    if codes_a and codes_b and not (codes_a & codes_b):
        return f"型号编码不同({'/'.join(sorted(codes_a))} vs {'/'.join(sorted(codes_b))})"
    fa = _freq_interval((props_a or {}).get("频率") or (props_a or {}).get("频段"))
    fb = _freq_interval((props_b or {}).get("频率") or (props_b or {}).get("频段"))
    if fa and fb and not (fa[0] <= fb[1] and fb[0] <= fa[1]):
        return "频段不重合"
    return None


async def _judge(llm, a: tuple, b: tuple, entity_type: str) -> dict:
    """第 ③ 层:LLM 判断是否同一(带属性),按名称对缓存,同一对只判一次方向。"""
    key = (entity_type, a[0], b[0])
    if key not in _judge_cache:
        _judge_cache[key] = await asyncio.to_thread(llm.judge_entity_detail, a[0], a[1], b[0], b[1], entity_type)
    return _judge_cache[key]


async def resolve_vector(session, milvus, embeddings, llm, name: str, entity_type: str,
                         mergeable: Optional[set] = None, properties: Optional[dict] = None,
                         parent_id: Optional[str] = None, self_id: Optional[str] = None,
                         record_suspect: bool = True) -> Optional[str]:
    """第 ②③④ 层:向量只找候选 → 属性冲突否决 → LLM 判断(交换顺序问两次,都判同一且置信度够才合并)
    → 判不了的不合并,记一条「疑似同一」待审(仅审计,不改图)。

    名称向量分不清「名字像但不是同一个」:BGE-M3 下 对空搜索雷达/对海搜索雷达 0.909,
    高于真同一的 Iskander-M/伊斯坎德尔-M 0.592,所以分数只用来找候选。llm_gate 关闭时不经向量合并。
    mergeable=None 表示不限制;集合外的类型(参数值/枚举类)只走确定性。
    """
    if not settings.graphrag_resolve_vector_enabled or milvus is None or embeddings is None:
        return None
    if not settings.graphrag_resolve_llm_gate or llm is None:
        return None
    if mergeable is not None and entity_type not in mergeable:
        return None
    from .alias_evidence import entity_codes
    try:
        vec = await asyncio.to_thread(embeddings.embed_one, name)
        top = await milvus.search_entities(vec, entity_type, settings.graphrag_resolve_topk)
    except Exception:
        return None  # 向量层不可用 -> 静默降级为确定性
    props = dict(properties or {})
    codes = entity_codes(name, entity_type, props, [])
    cands = [t for t in top if t[2] >= settings.graphrag_resolve_sim_low and t[0] != self_id]
    for eid, cname, score in cands[:settings.graphrag_resolve_judge_topk]:
        rows = await db.select_where(session, EntityRegistryEntry, entity_id=eid)
        crow = rows[0] if rows else {}
        cprops = dict(crow.get("properties") or {})
        ccodes = entity_codes(cname, entity_type, cprops, list(crow.get("aliases") or []))
        if _veto(props, codes, cprops, ccodes):
            continue
        r1 = await _judge(llm, (name, props), (cname, cprops), entity_type)
        r2 = await _judge(llm, (cname, cprops), (name, props), entity_type)
        conf = settings.graphrag_resolve_judge_confidence
        ok = [bool(r.get("same")) and float(r.get("confidence") or 0) >= conf for r in (r1, r2)]
        if all(ok):
            return eid
        if record_suspect and parent_id and (r1.get("same") or r2.get("same")):
            await db.insert(session, PendingExtraction, {
                "parent_id": parent_id, "stage": "resolve",
                "payload": {"name": name, "candidate": cname, "candidate_id": eid,
                            "entity_type": entity_type, "score": round(float(score), 3),
                            "judge": [r1, r2]},
                "reason": f"疑似同一:「{name}」与「{cname}」两次判断不一致或置信度不足,未合并",
                "attempts": 1,
            })
    return None


def _earliest(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda r: (r.get("created_at") or datetime.max,
                                       r.get("entity_id") or ""))[0]


async def align(session: AsyncSession, entities: list[Entity],
                relations: list[Relation], milvus=None, embeddings=None, llm=None
                ) -> tuple[list[Entity], list[Relation], dict]:
    """跨文档对齐:实体/关系原地 remap 到全局 canonical。开关关闭时原样透传。"""
    if not settings.graphrag_resolve_enabled or not entities:
        return entities, relations, {}
    mergeable = await _load_mergeable(session, milvus, embeddings)
    remap: dict[str, str] = {}
    for e in entities:
        resolved = await resolve_deterministic(session, e.name, e.entity_type,
                                               e.aliases, e.properties)
        if not resolved:
            resolved = await resolve_vector(session, milvus, embeddings, llm, e.name, e.entity_type,
                                            mergeable, properties=e.properties,
                                            parent_id=e.source_parent_id, self_id=e.entity_id)
        if resolved and resolved != e.entity_id:
            remap[e.entity_id] = resolved
    if remap:
        for e in entities:
            e.entity_id = remap.get(e.entity_id, e.entity_id)
        kept: list[Relation] = []
        for r in relations:
            r.head_entity_id = remap.get(r.head_entity_id, r.head_entity_id)
            r.tail_entity_id = remap.get(r.tail_entity_id, r.tail_entity_id)
            if r.head_entity_id == r.tail_entity_id:
                continue
            kept.append(r)
        relations = kept
    return entities, relations, remap


async def persist(session: AsyncSession, entities: list[Entity],
                  milvus=None, embeddings=None) -> None:
    """写回注册表 + 实体向量索引(幂等):name 首现者赢,aliases 并集,fingerprint 非空保留。"""
    mergeable = await _load_mergeable(session, milvus, embeddings)
    for e in entities:
        key = normalize_name(e.name)
        aliases = _dedup([key] + [normalize_name(a) for a in (e.aliases or [])])
        fp = fingerprint(e.properties)
        cur = await db.select_where(session, EntityRegistryEntry, entity_id=e.entity_id)
        if cur:
            prev = cur[0]
            name = prev.get("name") or key
            merged_aliases = _dedup(list(prev.get("aliases") or []) + aliases + [name])
            if not fp:
                fp = prev.get("fingerprint")
            props = dict(prev.get("properties") or {})
            for k, v in (e.properties or {}).items():
                props.setdefault(k, v)
            data = {"entity_id": e.entity_id, "name": name, "entity_type": e.entity_type,
                    "aliases": merged_aliases, "fingerprint": fp,
                    "properties": props, "status": "active"}
        else:
            data = {"entity_id": e.entity_id, "name": key, "entity_type": e.entity_type,
                    "aliases": _dedup(aliases + [key]), "fingerprint": fp,
                    "properties": dict(e.properties or {}), "status": "active"}
        await db.upsert(session, EntityRegistryEntry, data, "entity_id")
        # 实体名向量写回(消歧语义匹配层 ②,失败不影响确定性);参数值/枚举类型不进向量库
        if (settings.graphrag_resolve_vector_enabled and milvus is not None
                and embeddings is not None
                and (mergeable is None or e.entity_type in mergeable)):
            try:
                vec = await asyncio.to_thread(embeddings.embed_one, e.name)
                await milvus.upsert_entity_vector(e.entity_id, e.name, e.entity_type, vec)
            except Exception:
                pass


# ---- 全图批量对账 ----

def _union_group(dsu: DSU, group: list) -> None:
    for i in range(1, len(group)):
        dsu.union(group[0], group[i])


async def _absorb_registry(session: AsyncSession, old: str, new: str,
                           all_by_id: dict) -> None:
    """canonical 吸收 old 的 aliases/fingerprint/properties;old 标记 merged。"""
    old_r = all_by_id.get(old)
    new_rows = await db.select_where(session, EntityRegistryEntry, entity_id=new)
    new_r = new_rows[0] if new_rows else None
    if not old_r or not new_r:
        return
    new_name = normalize_name(new_r.get("name") or "")
    new_aliases = _dedup(list(new_r.get("aliases") or []))
    for a in [old_r.get("name")] + list(old_r.get("aliases") or []):
        k = normalize_name(a)
        if k and k != new_name and k not in new_aliases:
            new_aliases.append(k)
    fp = new_r.get("fingerprint") or old_r.get("fingerprint")
    props = dict(new_r.get("properties") or {})
    for k, v in (old_r.get("properties") or {}).items():
        props.setdefault(k, v)
    await db.upsert(session, EntityRegistryEntry, {
        "entity_id": new, "name": new_r.get("name"), "entity_type": new_r.get("entity_type"),
        "aliases": new_aliases, "fingerprint": fp, "properties": props, "status": "active",
    }, "entity_id")
    await db.update(session, EntityRegistryEntry, {
        "entity_id": old, "status": "merged", "merged_into": new,
    }, "entity_id")


async def _remap_parents(session: AsyncSession, old: str, new: str) -> int:
    """PG parent_document.kg_entity_id(逗号串)token 级 old -> new,返更新行数。"""
    rows = await db.select_all(session, ParentDocument)
    count = 0
    for r in rows:
        kg = r.get("kg_entity_id") or ""
        tokens = [t.strip() for t in kg.split(",") if t.strip()]
        if old not in tokens:
            continue
        new_tokens = _dedup([new if t == old else t for t in tokens])
        await db.update(session, ParentDocument, {
            "parent_id": r["parent_id"], "kg_entity_id": ",".join(new_tokens),
        }, "parent_id")
        count += 1
    return count


async def global_resolve(session: AsyncSession, neo4j, milvus, embeddings, llm,
                         dry_run: bool = False) -> dict:
    """全图批量对账:DSU 求 canonical + 四层回填,顺序无关、幂等。"""
    rows = await db.select_all(session, EntityRegistryEntry)
    all_by_id = {r["entity_id"]: r for r in rows}
    sorted_rows = sorted(rows, key=lambda r: (r.get("created_at") or datetime.max,
                                              r.get("entity_id") or ""))
    order_map = {r["entity_id"]: i for i, r in enumerate(sorted_rows)}

    dsu = DSU()
    for r in rows:
        dsu.add(r["entity_id"])

    # ① 确定性键 union(类型作用域)
    by_name: dict[tuple, list] = {}
    by_fp: dict[tuple, list] = {}
    for r in rows:
        if r.get("status") != "active":
            continue
        for a in [r.get("name")] + list(r.get("aliases") or []):
            k = normalize_name(a)
            if k:
                by_name.setdefault((r["entity_type"], k), []).append(r["entity_id"])
        if r.get("fingerprint"):
            by_fp.setdefault((r["entity_type"], r["fingerprint"]), []).append(r["entity_id"])
    for group in by_name.values():
        _union_group(dsu, group)
    for group in by_fp.values():
        _union_group(dsu, group)

    # ② 向量只找候选,LLM 判断同一才 union(llm_gate 关闭时不经向量合并);只对可向量化类型(参数值类跳过)
    mergeable = await _load_mergeable(session, milvus, embeddings)
    if settings.graphrag_resolve_vector_enabled and milvus is not None and embeddings is not None:
        for r in rows:
            if r.get("status") != "active":
                continue
            if mergeable is not None and r["entity_type"] not in mergeable:
                continue
            eid = await resolve_vector(session, milvus, embeddings, llm, r["name"], r["entity_type"],
                                       None, properties=r.get("properties"), self_id=r["entity_id"],
                                       record_suspect=False)
            if eid:
                dsu.union(r["entity_id"], eid)

    # canonical = 组内最早(created_at, entity_id)
    remap: dict[str, str] = {}
    for members in dsu.components().values():
        members.sort(key=lambda eid: (order_map.get(eid, 1 << 30), eid))
        canon = members[0]
        for m in members[1:]:
            remap[m] = canon

    if dry_run:
        return {
            "dry_run": True,
            "entities_before": len(rows),
            "entities_after": len(rows) - len(remap),
            "merged": [
                {"old_id": o, "new_id": n, "name": all_by_id.get(o, {}).get("name", "")}
                for o, n in remap.items()
            ],
        }

    report = {"dry_run": False, "merged": [], "binds_moved": 0, "relations_moved": 0,
              "parents_updated": 0, "chunks_updated": 0,
              "entities_before": len(rows), "entities_after": len(rows) - len(remap)}
    for old, new in remap.items():
        old_r = all_by_id.get(old, {})
        name = old_r.get("name", "")
        await _absorb_registry(session, old, new, all_by_id)          # ① PG 注册表
        report["binds_moved"] += await neo4j.move_binds(old, new)     # ② Neo4j BINDS
        report["relations_moved"] += await neo4j.move_relations(old, new)  # ② Neo4j 关系
        await neo4j.delete_entity(old)
        report["parents_updated"] += await _remap_parents(session, old, new)  # ③ PG 父块
        if milvus is not None:                                         # ④ Milvus 双库
            report["chunks_updated"] += await milvus.remap_kg_entity_ids(old, new)
            await milvus.delete_entity_vector(old)
        report["merged"].append({"old_id": old, "new_id": new, "name": name})
    return report
