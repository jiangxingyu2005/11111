"""GP-08 QRY-01:A/B 双路召回(Plan §3.9,async 移植)。

A 路:绑定子块,ARRAY_CONTAINS(kg_entity_ids, entity_id) 过滤命中;
B 路:未绑定子块,ARRAY_LENGTH(kg_entity_ids) == 0。
两路均带 enabled == true 过滤(禁用联动),并加 kb_id 过滤(知识库隔离)。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from ..config import get
from ..embeddings import EmbeddingClient
from ..ids import normalize_name
from ..milvus_client import F_ENABLED, F_KB_ID, F_KG_ENTITY_IDS, AsyncMilvusClient

# 型号常写成「编号 + 名称」(MGM-140 ATACMS / 9K720 Iskander-M),用户只说其中一段
_SPLIT_RE = re.compile(r"[\s/、,，;；()（）]+")
_MIN_KEY_LEN = 3   # 太短的片段(如「M」)不单独当匹配键,避免误命中


def _match_keys(names: list[str]) -> set[str]:
    """实体名/别名 -> 匹配键:全名 + 按空格/斜杠等切出的片段(归一化后)。"""
    keys: set[str] = set()
    for n in names:
        if not n:
            continue
        keys.add(normalize_name(n))
        for part in _SPLIT_RE.split(str(n)):
            p = normalize_name(part)
            if len(p) >= _MIN_KEY_LEN:
                keys.add(p)
    return {k for k in keys if k}


def match_entities(query: str, entities: list[dict]) -> list[str]:
    """把查询词匹配到实体(名称/别名/型号片段命中,大小写/全角不敏感)。返回 entity_id 列表。

    入库时实体 name 已经 normalize_name 小写化,查询侧同样归一化后再做子串匹配,
    否则「2.4GHz」永远匹配不到实体「2.4ghz」。别名里的原始型号写法会切成片段,
    让「ATACMS导弹」能命中实体「MGM-140 ATACMS」。
    """
    q = normalize_name(query)
    hits = []
    for e in entities:
        names = [e.get("name", "")] + list(e.get("aliases") or [])
        model = _props(e.get("properties")).get("型号")
        if model:
            names.append(str(model))
        if any(k in q for k in _match_keys(names)):
            hits.append(e["entity_id"])
    return hits


def _props(v) -> dict:
    """Neo4j 里 properties 存成 JSON 字符串,这里解析回 dict。"""
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v) if v else {}
    except (TypeError, ValueError):
        return {}


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
    """B 路:纯向量兜底,覆盖全部启用子块。

    不再限定「未绑定子块」:入库抽取后大部分子块都绑定了实体,若 B 路只搜未绑定的,
    实体名没匹配上时这些子块就彻底召回不到。两路重叠由 rerank 按 child_id 去重。
    """
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    expr = and_expr(_enabled_expr(), kb_expr(kb_id))
    return await milvus.search(vec, topk, expr=expr)


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")   # 拼进 Milvus 表达式的实体 ID 只允许安全字符


async def recall_expanded(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                          entity_ids: list, kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """图扩展补召回:绑定到扩展实体的子块,一次检索(ARRAY_CONTAINS_ANY)。

    原先每个扩展实体各搜一次,扩展出 N 个实体就是 N 次检索;现在固定 1 次、最多 topk 条,
    结果交给 rerank 与 A/B 两路统一排序截断。
    """
    ids = [e for e in entity_ids if e and _SAFE_ID_RE.match(str(e))]
    if not ids:
        return []
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    id_list = ", ".join(f"'{e}'" for e in ids)
    expr = and_expr(_enabled_expr(), kb_expr(kb_id), f"ARRAY_CONTAINS_ANY({F_KG_ENTITY_IDS}, [{id_list}])")
    return await milvus.search(vec, topk, expr=expr)


async def dual_recall(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                      entity_ids: list, kb_id=None,
                      topk: Optional[int] = None) -> dict:
    """A/B 双路召回,返回 {a, b}。"""
    return {
        "a": await recall_a(query, milvus, embeddings, entity_ids, kb_id, topk),
        "b": await recall_b(query, milvus, embeddings, kb_id, topk),
    }
