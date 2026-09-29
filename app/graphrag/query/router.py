"""GP-08 QRY-01:意图分流 + 在线查询编排(Plan §3.9,async 移植)。

检索型走 A/B 双路召回;关系型在图扩展后追加绑定子块召回。
最终 generate_answer 带引用;命中经 trace_hook 计入 TRACE(§3.11 联动)。

关系触发词不再 import 时从 yaml 读:由端点现查 extraction_dict(dict.load_relations)
后经 relation_defs 传入,改字典即时生效。
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

from ..config import get
from ..embeddings import EmbeddingClient
from ..llm import LLMClient
from ..milvus_client import F_ENABLED, F_KG_ENTITY_IDS, AsyncMilvusClient
from ..neo4j_client import Neo4jClient
from .answer import generate_answer
from .expand import expand
from .nl2query import nl2query
from .recall import and_expr, dual_recall, kb_expr, match_entities
from .rerank import rerank


def route(query: str, relation_defs: dict) -> str:
    """意图分流:命中关系触发词 -> graph,否则 retrieval。"""
    rel_words = {w for r in relation_defs.get("relation_types", []) for w in r.get("trigger_words", [])}
    if any(w in query for w in rel_words):
        return "graph"
    return "retrieval"


async def search(query: str, milvus: AsyncMilvusClient, neo4j: Neo4jClient,
                 embeddings: Optional[EmbeddingClient] = None,
                 llm: Optional[LLMClient] = None,
                 relation_defs: Optional[dict] = None,
                 kb_id=None,
                 trace_hook: Optional[Callable] = None) -> dict:
    embeddings = embeddings or EmbeddingClient()
    llm = llm or LLMClient()
    relation_defs = relation_defs or {"relation_types": []}
    intent = route(query, relation_defs)
    cond = nl2query(query)

    entities = await neo4j.list_entities()
    matched_ids = match_entities(query, entities)

    rec = await dual_recall(query, milvus, embeddings, matched_ids, kb_id=kb_id)
    hits = rerank(rec["a"], rec["b"], matched_ids=matched_ids)

    # 关系型:图扩展并追加绑定子块召回
    expanded = []
    if intent == "graph" and matched_ids:
        expanded = await expand(neo4j, matched_ids)
        extra_ids = [e["entity_id"] for e in expanded]
        existing = {x.get("child_id") for x in hits}
        vec = await asyncio.to_thread(embeddings.embed_one, query)
        for eid in extra_ids:
            expr = and_expr(f"{F_ENABLED} == true", kb_expr(kb_id),
                            f"ARRAY_CONTAINS({F_KG_ENTITY_IDS}, '{eid}')")
            for h in await milvus.search(vec, int(get("query.topk", 10)), expr=expr):
                if h.get("child_id") not in existing:
                    existing.add(h["child_id"])
                    hits.append(h)

    answer = await generate_answer(query, hits, llm, trace_hook)
    return {"intent": intent, "cond": cond, "answer": answer, "hits": hits,
            "matched_entities": matched_ids, "expanded_entities": expanded}
