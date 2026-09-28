"""实体消歧解析引擎:三阶段匹配(①确定性 ②向量 ③LLM 灰区)接到真实 PG/Milvus/Neo4j。

- resolve_deterministic:名称/别名/指纹查 entity_registry(类型作用域)。
- resolve_vector:实体名向量查 Milvus ems_entities(类型作用域,COSINE)。
- align:抽取结果对齐到全局 canonical(原地 remap entity_id + 关系 head/tail,自环丢弃)。
- persist:canonical 实体写回 entity_registry + ems_entities(幂等,name 首现者赢、aliases 并集)。
- global_resolve:全图批量对账(DSU 求 canonical + 四层回填 PG/Neo4j/Milvus)。

纯逻辑已由 disambig/engine.py 离线验证(16/16),本模块是其生产实现;向量层/灰区/对账
为其生产化落地,与 disambig 三阶段管道一一对应。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models import EntityRegistryEntry, ParentDocument
from . import db
from . import dict as dict_store
from .disambig.dsu import DSU
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


async def resolve_vector(milvus, embeddings, llm, name: str, entity_type: str,
                         mergeable: Optional[set] = None) -> Optional[str]:
    """匹配器②③:实体名向量相似度(类型作用域)。≥sim_high 自动并,灰区交 LLM。

    mergeable=None 表示不限制(向后兼容);传入可向量化类型集合时,集合外的类型直接跳过
    (参数值/枚举类实体不走向量层,只走确定性)。
    """
    if not settings.graphrag_resolve_vector_enabled or milvus is None or embeddings is None:
        return None
    if mergeable is not None and entity_type not in mergeable:
        return None
    try:
        vec = await asyncio.to_thread(embeddings.embed_one, name)
        top = await milvus.search_entities(vec, entity_type, settings.graphrag_resolve_topk)
    except Exception:
        return None  # 向量层不可用 -> 静默降级为确定性
    if not top:
        return None
    eid, cname, score = top[0]
    if score >= settings.graphrag_resolve_sim_high:
        return eid
    if (score >= settings.graphrag_resolve_sim_low
            and settings.graphrag_resolve_llm_gate and llm is not None):
        same = await asyncio.to_thread(llm.judge_same_entity, name, cname, entity_type)
        if same:
            return eid
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
            resolved = await resolve_vector(milvus, embeddings, llm, e.name, e.entity_type,
                                            mergeable)
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

    # ② 向量命中 union(≥sim_high 并;灰区交 LLM,默认关);只对可向量化类型(参数值类跳过)
    mergeable = await _load_mergeable(session, milvus, embeddings)
    if settings.graphrag_resolve_vector_enabled and milvus is not None and embeddings is not None:
        for r in rows:
            if r.get("status") != "active":
                continue
            if mergeable is not None and r["entity_type"] not in mergeable:
                continue
            try:
                vec = await asyncio.to_thread(embeddings.embed_one, r["name"])
                top = await milvus.search_entities(vec, r["entity_type"],
                                                   settings.graphrag_resolve_topk)
            except Exception:
                continue
            for (eid, cname, score) in top:
                if eid == r["entity_id"]:
                    continue
                if score >= settings.graphrag_resolve_sim_high:
                    dsu.union(r["entity_id"], eid)
                elif (score >= settings.graphrag_resolve_sim_low
                      and settings.graphrag_resolve_llm_gate and llm is not None):
                    same = await asyncio.to_thread(llm.judge_same_entity,
                                                   r["name"], cname, r["entity_type"])
                    if same:
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
