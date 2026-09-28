"""GP-04 阶段二:关系抽取(ems graph/extract_relations.py 移植,I/O 改 async)。

关系类型/触发词从 extraction_dict 表读(经 relation_defs 传入),不再读 yaml。
低置信或校验不过 -> pending_extraction。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import PendingExtraction
from . import db
from .config import load_few_shot
from .ids import canonical_relation_id
from .llm import LLMClient
from .models import Entity, Relation, Support


def _relation_types(relation_defs: dict) -> dict[str, str]:
    """en 名称 -> 中文名称。"""
    return {r["en"]: r["name"] for r in relation_defs["relation_types"]}


def _validate(raw: dict, entity_ids: set, rel_types: set,
              id_to_type: dict, rel_domains: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ["关系必须是 JSON 对象"]
    for f in ("head", "tail", "rel_type"):
        if not raw.get(f):
            errors.append(f"缺少字段 {f}")
    if raw.get("head") not in entity_ids:
        errors.append(f"head 实体不在实体集中: {raw.get('head')!r}")
    if raw.get("tail") not in entity_ids:
        errors.append(f"tail 实体不在实体集中: {raw.get('tail')!r}")
    rt = raw.get("rel_type")
    if rt not in rel_types:
        errors.append(f"rel_type 非法: {rt!r}")
    else:
        # 两端实体类型约束(domain/range):head/tail 的实体类型必须在关系定义允许的集合内,
        # 否则就是「辐射源—装备于→辐射源」这类荒谬边。
        head_types, tail_types = rel_domains.get(rt, (set(), set()))
        ht = id_to_type.get(raw.get("head"))
        tt = id_to_type.get(raw.get("tail"))
        if head_types and ht is not None and ht not in head_types:
            errors.append(f"head 实体类型 {ht!r} 不允许作为 {rt} 的起点(允许 {sorted(head_types)})")
        if tail_types and tt is not None and tt not in tail_types:
            errors.append(f"tail 实体类型 {tt!r} 不允许作为 {rt} 的终点(允许 {sorted(tail_types)})")
    conf = raw.get("confidence")
    if conf is None or not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errors.append(f"confidence 非法: {conf!r}")
    return errors


async def extract(session: AsyncSession, entities: list[Entity], llm: Optional[LLMClient],
                  parents: list[dict], relation_defs: dict) -> tuple[list[Relation], list[Support]]:
    llm = llm or LLMClient()
    rel_types = set(_relation_types(relation_defs).keys())
    # 低置信阈值取类型外置 min_confidence(默认 0.75),而非全局低阈值
    min_conf = {r["en"]: float(r.get("min_confidence", 0.75))
                for r in relation_defs["relation_types"]}
    fewshot = "\n\n".join(load_few_shot(r["en"]) for r in relation_defs["relation_types"]
                          if load_few_shot(r["en"]))

    name_to_id = {e.name: e.entity_id for e in entities}
    id_to_type = {e.entity_id: e.entity_type for e in entities}
    rel_domains = {
        r["en"]: (set(r.get("head_types") or []), set(r.get("tail_types") or []))
        for r in relation_defs["relation_types"]
    }
    relations: dict[str, Relation] = {}
    supports: list[Support] = []

    for p in parents:
        # 该父块覆盖到的实体(按其名称出现在正文中)
        parent_entities = [e for e in entities if e.name in p.get("content", "")]
        if not parent_entities:
            continue
        raw_list = await asyncio.to_thread(
            llm.extract_relations, p.get("content", ""),
            [e.to_dict() for e in parent_entities],
            relation_defs["relation_types"], fewshot,
        )
        for raw in raw_list:
            # mock/openai 返回 name -> 映射为 id
            raw["head"] = name_to_id.get(raw.get("head"), raw.get("head"))
            raw["tail"] = name_to_id.get(raw.get("tail"), raw.get("tail"))
            errors = _validate(raw, set(name_to_id.values()), rel_types,
                               id_to_type, rel_domains)
            conf = raw.get("confidence", 0.0) if isinstance(raw, dict) else 0.0
            thresh = min_conf.get(raw.get("rel_type"), 0.75)
            if errors or conf < thresh:
                await db.insert(session, PendingExtraction, {
                    "parent_id": p["parent_id"], "stage": "relation", "payload": raw,
                    "reason": "；".join(errors) if errors else f"置信度 {conf} 低于阈值 {thresh}",
                    "attempts": 1,
                })
                continue
            rid = canonical_relation_id(raw["head"], raw["tail"], raw["rel_type"], p["parent_id"])
            if rid not in relations:
                relations[rid] = Relation(
                    relation_id=rid, head_entity_id=raw["head"], tail_entity_id=raw["tail"],
                    rel_type=raw["rel_type"], confidence=float(conf),
                    evidence=raw.get("evidence", ""), source_parent_id=p["parent_id"],
                )
                supports.append(Support(
                    relation_id=rid, entity_id=raw["head"], position="head",
                    evidence_sentence=raw.get("evidence", ""), parent_id=p["parent_id"],
                ))
                supports.append(Support(
                    relation_id=rid, entity_id=raw["tail"], position="tail",
                    evidence_sentence=raw.get("evidence", ""), parent_id=p["parent_id"],
                ))
    return list(relations.values()), supports
