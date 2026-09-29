"""GP-04 阶段一:实体抽取(ems graph/extract_entities.py 移植,I/O 改 async)。

六道防线体现在 _validate + 三档置信度分流:high 直接采用;[low,high) 采用;<low 或校验不过
-> pending_extraction。实体 ID 确定性哈希,同父块重跑不翻倍。
融合后:实体类型从 extraction_dict 表经参数传入(不再读 yaml config.entity_types)。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import PendingExtraction
from . import db
from .config import get
from .ids import canonical_entity_id, normalize_name
from .llm import LLMClient
from .models import Entity


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
    if not etype or str(etype).strip() not in allowed_types:
        errors.append(f"entity_type 非法: {etype!r}")
    if conf is None or not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errors.append(f"confidence 非法: {conf!r}")
    return errors


def _confidence_gate() -> tuple[float, float]:
    return float(get("graph.confidence_gate.high", 0.9)), float(get("graph.confidence_gate.low", 0.6))


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
    return list(registry.values())
