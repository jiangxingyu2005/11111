"""抽取字典读取:从 extraction_dict 表现查实体类型 / 关系定义(异步,不用缓存)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EntityAlias, ExtractionDict


async def load_entity_defs(session: AsyncSession) -> list[dict]:
    """要抽取的实体类型完整定义(category='entity' 且未逻辑删除、启用)。

    返回 name/en/description/json_schema,供 LLM prompt 讲清每类含义——改 /graph/dict 的
    description/json_schema 即可影响抽取,无需改代码(字典重定义框架)。
    """
    stmt = (
        select(ExtractionDict)
        .where(
            ExtractionDict.category == "entity",
            ExtractionDict.deleted_at.is_(None),
            ExtractionDict.enabled.is_(True),
        )
        .order_by(ExtractionDict.created_at)
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()
    return [
        {
            "name": r.name,
            "en": r.en or "",
            "description": r.description or "",
            "json_schema": dict(r.json_schema) if r.json_schema else None,
        }
        for r in rows
    ]


async def load_mergeable_types(session: AsyncSession) -> set[str]:
    """可走向量语义层归并的实体类型集合(category='entity' 且 semantic_mergeable=true)。

    消歧匹配器②(向量)只对这些类型生效;参数值/枚举类(频率/调制方式/信号等)被排除,
    避免「2.4GHz ↔ 850MHz」这类语义相近但非同一实体的窄带误并。
    """
    stmt = (
        select(ExtractionDict.name)
        .where(
            ExtractionDict.category == "entity",
            ExtractionDict.deleted_at.is_(None),
            ExtractionDict.enabled.is_(True),
            ExtractionDict.semantic_mergeable.is_(True),
        )
        .order_by(ExtractionDict.created_at)
    )
    res = await session.execute(stmt)
    return set(res.scalars())


async def load_relations(session: AsyncSession) -> dict:
    """关系定义(category='relation'),返回 ems 兼容的 {"relation_types": [...]}。"""
    stmt = (
        select(ExtractionDict)
        .where(
            ExtractionDict.category == "relation",
            ExtractionDict.deleted_at.is_(None),
            ExtractionDict.enabled.is_(True),
        )
        .order_by(ExtractionDict.created_at)
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()
    relation_types = [
        {
            "name": r.name,
            "en": r.en or "",
            "description": r.description or "",
            "trigger_words": list(r.trigger_words or []),
            "min_confidence": float(r.min_confidence) if r.min_confidence is not None else 0.75,
            "head_types": list(r.head_types or []),
            "tail_types": list(r.tail_types or []),
            "json_schema": dict(r.json_schema) if r.json_schema else None,
            "symmetric": False,
        }
        for r in rows
    ]
    return {"relation_types": relation_types}


async def load_alias_library(session: AsyncSession) -> list[dict]:
    """别名库:代号 → 类别字典实体类型(实体抽取动态提示词用,每次现查、无缓存)。

    返回 [{"alias", "entity_type"}, ...],按创建顺序排序;抽取侧只注入 entity_type
    在当前实体类型集内的代号,避免提示 schema 里没有的类型。
    """
    stmt = select(EntityAlias).order_by(EntityAlias.created_at)
    res = await session.execute(stmt)
    rows = res.scalars().all()
    return [{"alias": r.alias, "entity_type": r.entity_type} for r in rows]
