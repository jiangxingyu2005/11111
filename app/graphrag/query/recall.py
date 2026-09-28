"""GP-08 QRY-01:A/B 双路召回(Plan §3.9,async 移植)。

A 路:绑定子块,ARRAY_CONTAINS(kg_entity_ids, entity_id) 过滤命中;
B 路:未绑定子块,ARRAY_LENGTH(kg_entity_ids) == 0。
两路均带 enabled == true 过滤(禁用联动),并加 kb_id 过滤(知识库隔离)。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from ..config import get
from ..embeddings import EmbeddingClient
from ..ids import normalize_name
from ..milvus_client import F_ENABLED, F_KB_ID, F_KG_ENTITY_IDS, AsyncMilvusClient


def match_entities(query: str, entities: list[dict]) -> list[str]:
    """把查询词匹配到实体(名称/别名命中,大小写/全角不敏感)。返回 entity_id 列表。

    入库时实体 name 已经 normalize_name 小写化,查询侧同样归一化后再做子串匹配,
    否则「2.4GHz」永远匹配不到实体「2.4ghz」。
    """
    q = normalize_name(query)
    hits = []
    for e in entities:
        names = [e.get("name", "")] + list(e.get("aliases") or [])
        if any(n and normalize_name(n) in q for n in names):
            hits.append(e["entity_id"])
    return hits


def and_expr(*parts: str) -> str:
    """拼接 Milvus 过滤表达式(自动跳过空串)。"""
    return " && ".join(p for p in parts if p)


def kb_expr(kb_id) -> str:
    """kb_id 过滤子句(未传则空 = 全库)。"""
    return f"{F_KB_ID} == '{kb_id}'" if kb_id else ""


def _enabled_expr() -> str:
    return f"{F_ENABLED} == true"


async def recall_a(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                   entity_ids: list, kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """A 路:绑定子块,按实体 ARRAY_CONTAINS 过滤。"""
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    hits: list[dict] = []
    seen: set = set()
    for eid in entity_ids:
        expr = and_expr(_enabled_expr(), kb_expr(kb_id),
                        f"ARRAY_CONTAINS({F_KG_ENTITY_IDS}, '{eid}')")
        for h in await milvus.search(vec, topk, expr=expr):
            if h.get("child_id") not in seen:
                seen.add(h["child_id"])
                hits.append(h)
    return hits


async def recall_b(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                   kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """B 路:未绑定子块(ARRAY_LENGTH==0)。"""
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    expr = and_expr(_enabled_expr(), kb_expr(kb_id), f"ARRAY_LENGTH({F_KG_ENTITY_IDS}) == 0")
    return await milvus.search(vec, topk, expr=expr)


async def dual_recall(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                      entity_ids: list, kb_id=None,
                      topk: Optional[int] = None) -> dict:
    """A/B 双路召回,返回 {a, b}。"""
    return {
        "a": await recall_a(query, milvus, embeddings, entity_ids, kb_id, topk),
        "b": await recall_b(query, milvus, embeddings, kb_id, topk),
    }
