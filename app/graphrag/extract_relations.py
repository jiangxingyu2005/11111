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
from .ids import canonical_relation_id, normalize_name
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
    # 中文名/触发词 -> en 归一化映射:小模型(如 ollama qwen2.5:3b)常输出中文 rel_type
    # (如 "装备于"/"隶属"/"归属")而非英文 en,这里在 _validate 前统一映射回 en。
    # 覆盖 name(关系名)与 trigger_words(同义触发词,如 "归属" 是 BELONGS_TO 的触发词)。
    rel_label_to_en: dict[str, str] = {}
    for r in relation_defs["relation_types"]:
        en = r.get("en", "")
        if not en:
            continue
        rel_label_to_en[en] = en
        name = r.get("name", "")
        if name:
            rel_label_to_en[name] = en
        for tw in (r.get("trigger_words") or []):
            if tw:
                rel_label_to_en[tw] = en
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
        # 该父块覆盖到的实体(按其归一化名称出现在归一化正文中)。
        # 实体名已被 normalize_name 小写化(如 AN/TPQ-50 -> an/tpq-50),正文仍是原始大小写,
        # 若直接 e.name in content 做大小写敏感匹配会全部落空、关系抽取被跳过(LLM 从未调用)。
        # 这里把正文也归一化后再做子串匹配,保证英文大写实体/含空格中文名能命中原文。
        norm_content = normalize_name(p.get("content", ""))
        parent_entities = [e for e in entities if normalize_name(e.name) in norm_content]
        if not parent_entities:
            continue
        raw_list = await asyncio.to_thread(
            llm.extract_relations, p.get("content", ""),
            [e.to_dict() for e in parent_entities],
            relation_defs["relation_types"], fewshot,
        )
        for raw in raw_list:
            # mock/openai 返回实体名 -> 映射为 id。实体名已 normalize_name(小写+去空白),
            # 而小模型常回原大小写(如 AN/TPQ-50 vs 实体名 an/tpq-50),这里先归一化再查表。
            raw["head"] = name_to_id.get(normalize_name(str(raw.get("head", ""))), raw.get("head"))
            raw["tail"] = name_to_id.get(normalize_name(str(raw.get("tail", ""))), raw.get("tail"))
            # 中文 rel_type -> en 归一化(小模型偶发输出中文而非英文标识)
            rt = raw.get("rel_type")
            if isinstance(rt, str) and rt.strip() not in rel_types:
                mapped = rel_label_to_en.get(rt.strip())
                if mapped:
                    raw["rel_type"] = mapped
            # 小模型偶发漏 confidence 字段,默认中档置信,不因此丢关系
            if isinstance(raw, dict) and raw.get("confidence") is None:
                raw["confidence"] = 0.8
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
