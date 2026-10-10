"""GP-04 阶段一:实体抽取(ems graph/extract_entities.py 移植,I/O 改 async)。

六道防线体现在 _validate + 三档置信度分流:high 直接采用;[low,high) 采用;<low 或校验不过
-> pending_extraction。实体 ID 确定性哈希,同父块重跑不翻倍。
融合后:实体类型从 extraction_dict 表经参数传入(不再读 yaml config.entity_types)。
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import PendingExtraction
from . import db
from .config import get
from .alias_evidence import add_alias_evidence
from .coref import merge_mentions
from .ids import canonical_entity_id, normalize_name
from .llm import LLMClient
from .models import Entity

# 需求(10):频率/脉宽/重频/调制等是「属性」不是实体。名字只是参数值的,拒收进 pending。
_PARAM_VALUE_RE = re.compile(
    r"^[\d.\s~～\-至到]*\d[\d.\s~～\-至到]*(ghz|mhz|khz|hz|μs|us|ms|ns|db|dbm|w|kw|km|m|%)?$",
    re.IGNORECASE,
)
_MODULATION_NAMES = {"qpsk", "bpsk", "8psk", "16qam", "64qam", "fsk", "msk", "gmsk",
                     "fm", "am", "lfm", "nlfm", "ofdm", "dsss", "跳频", "线性调频"}


def _is_param_value(name: str) -> bool:
    s = str(name).strip().lower().replace(" ", "")
    return bool(_PARAM_VALUE_RE.match(s)) or s in _MODULATION_NAMES


def _validate(raw: dict, allowed_types: list) -> list[str]:
    """实体 JSON Schema 约束(防线)。返回错误列表,空 = 通过。"""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ["实体必须是 JSON 对象"]
    name = raw.get("name")
    etype = raw.get("entity_type")
    conf = raw.get("confidence")
    if not name or not str(name).strip():
        errors.append("name 为空")
    elif _is_param_value(name):
        errors.append(f"参数值不是实体: {name!r}(应写入所属实体 properties)")
    if not etype or str(etype).strip() not in allowed_types:
        errors.append(f"entity_type 非法: {etype!r}")
    if conf is None or not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errors.append(f"confidence 非法: {conf!r}")
    return errors


def _confidence_gate() -> tuple[float, float]:
    return float(get("graph.confidence_gate.high", 0.9)), float(get("graph.confidence_gate.low", 0.6))


# 同名多类型裁决:逐类抽取时,同一名字常被多个类型的调用同时返回(如「干扰机」既进辐射源又进信号样本)。
# 不裁决的话下游按名字合并会随机取类型。置信度高者胜,平手按类型优先级(装备/名词类优先于信号数据类)。
_TYPE_PRIORITY = ["辐射源", "武器", "平台", "组织", "目标", "事件", "情报产品",
                  "工作状态", "信号簇", "原始信号", "信号样本"]


def _rank(e: Entity) -> tuple:
    prio = (_TYPE_PRIORITY.index(e.entity_type) if e.entity_type in _TYPE_PRIORITY
            else len(_TYPE_PRIORITY))
    return (-e.confidence, prio)


async def _resolve_type_conflicts(session: AsyncSession, entities: list[Entity]) -> list[Entity]:
    """同名实体只留一个类型;落选者的别名/属性并入胜者,本身进 pending 留痕。"""
    by_name: dict[str, list[Entity]] = {}
    for e in entities:
        by_name.setdefault(e.name, []).append(e)
    kept: list[Entity] = []
    for group in by_name.values():
        group.sort(key=_rank)
        winner = group[0]
        for loser in group[1:]:
            for a in loser.aliases:
                if a not in winner.aliases:
                    winner.aliases.append(a)
            for k, v in loser.properties.items():
                winner.properties.setdefault(k, v)
            await db.insert(session, PendingExtraction, {
                "parent_id": loser.source_parent_id, "stage": "entity",
                "payload": {"name": loser.name, "entity_type": loser.entity_type,
                            "confidence": loser.confidence},
                "reason": f"类型冲突:同名实体已归为「{winner.entity_type}」",
                "attempts": 1,
            })
        kept.append(winner)
    return kept


async def extract(session: AsyncSession, llm: Optional[LLMClient], parents: list[dict],
                  entity_types: list[str],
                  entity_defs: Optional[list] = None) -> list[Entity]:
    llm = llm or LLMClient()
    allowed_types = list(entity_types)
    _, low = _confidence_gate()

    registry: dict[str, Entity] = {}          # entity_id -> Entity(归一化注册表)
    for p in parents:
        raw_list = await asyncio.to_thread(llm.extract_entities, p.get("content", ""),
                                           allowed_types, entity_defs)
        for raw in raw_list:
            # 小模型(如 ollama qwen2.5:3b)偶发漏 confidence 字段,默认中档置信,不因此丢实体
            if isinstance(raw, dict) and raw.get("confidence") is None:
                raw["confidence"] = 0.8
            errors = _validate(raw, allowed_types)
            conf = raw.get("confidence", 0.0) if isinstance(raw, dict) else 0.0
            if errors or conf < low:
                await db.insert(session, PendingExtraction, {
                    "parent_id": p["parent_id"], "stage": "entity", "payload": raw,
                    "reason": "；".join(errors) if errors else f"置信度 {conf} 低于阈值 {low}",
                    "attempts": 1,
                })
                continue
            name = normalize_name(raw["name"])
            eid = canonical_entity_id(name, raw["entity_type"])
            if eid not in registry:
                registry[eid] = Entity(
                    entity_id=eid, name=name, entity_type=raw["entity_type"],
                    aliases=list(raw.get("aliases", []) or []),
                    properties=dict(raw.get("properties", {}) or {}),
                    confidence=float(conf), source_parent_id=p["parent_id"],
                )
            else:
                # 同名实体在多父块出现:补别名,不新发号
                for a in (raw.get("aliases", []) or []):
                    if a not in registry[eid].aliases:
                        registry[eid].aliases.append(a)
    entities = await _resolve_type_conflicts(session, list(registry.values()))
    # 指代消解(实体侧):「雷达」这类代称并回前文全名实体,不单独成节点
    entities, notes = merge_mentions(entities, parents)
    for note in notes:
        await db.insert(session, PendingExtraction, note)
    # 消歧第 ① 层:原文同指写法(括号对照/又称/即)与型号编码写成别名,供跨文档确定性合并
    add_alias_evidence(entities, parents)
    return entities
