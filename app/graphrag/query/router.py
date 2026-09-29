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


# 关系型问句启发式:疑问词 + 关系动词,如「A国装备了哪些武器」。这类问法的答案
# 在图的边上(而非子块文本),须走图扩展分支拿结构化关系事实。字典触发词覆盖不到
# 「有哪些 Y」的关系问法时,靠这一层兜住。
_RELATION_QUESTION_MARKERS = ("哪些", "哪类", "何种", "什么", "多少", "谁")
_RELATION_VERBS = ("装备", "配备", "搭载", "安装", "部署", "拥有", "持有",
                   "属于", "隶属", "下辖", "配属", "归属", "编成", "辖",
                   "组成", "构成", "包含", "包括", "来源", "来自", "源自",
                   "产生", "生成", "输出", "发射", "匹配", "对应", "识别为")


def route(query: str, relation_defs: dict) -> str:
    """意图分流:命中关系触发词(字典)或关系型问句 -> graph,否则 retrieval。"""
    rel_words = {w for r in relation_defs.get("relation_types", []) for w in r.get("trigger_words", [])}
    if any(w in query for w in rel_words):
        return "graph"
    if (any(m in query for m in _RELATION_QUESTION_MARKERS)
            and any(v in query for v in _RELATION_VERBS)):
        return "graph"
    return "retrieval"


async def collect_graph_facts(neo4j: Neo4jClient, matched_ids: list[str],
                              id2name: dict) -> list[str]:
    """把命中实体的一跳关系边整理成结构化事实文本(看图答)。

    即便对端实体是孤儿(无 BINDS 子块内容),也能据边直接回答「X 有哪些 Y」。
    方向:out = 命中实体指向对端;in = 对端指向命中实体。
    """
    facts: list[str] = []
    for eid in matched_ids:
        for r in await neo4j.entity_relations(eid):
            head = id2name.get(eid, eid)
            tgt = r["target_name"]
            if r.get("target_type"):
                tgt = f"{tgt}（{r['target_type']}）"
            if r["dir"] == "out":
                facts.append(f"{head} --[{r['rel_type']}]--> {tgt}")
            else:
                facts.append(f"{tgt} --[{r['rel_type']}]--> {head}")
    return facts


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

    # 关系型:图扩展、收集一跳关系事实(看图答)、追加绑定子块召回
    expanded = []
    graph_facts: list[str] = []
    if intent == "graph" and matched_ids:
        expanded = await expand(neo4j, matched_ids)
        id2name = {e["entity_id"]: e["name"] for e in entities}
        graph_facts = await collect_graph_facts(neo4j, matched_ids, id2name)
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

    answer = await generate_answer(query, hits, llm, trace_hook, graph_facts=graph_facts)
    return {"intent": intent, "cond": cond, "answer": answer, "hits": hits,
            "matched_entities": matched_ids, "expanded_entities": expanded,
            "graph_facts": graph_facts}
