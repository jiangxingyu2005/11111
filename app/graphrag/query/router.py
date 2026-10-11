"""GP-08 QRY-01:意图分流 + 在线查询编排(Plan §3.9,async 移植)。

检索型走 A/B 双路召回;关系型在图扩展后补召回扩展实体的绑定子块,三路统一重排截断。
最终 generate_answer 带引用;命中经 trace_hook 计入 TRACE(§3.11 联动)。
上下文有界:图扩展最多 expand_limit 个实体(按 hop_quotas 分配每跳名额)、
每实体最多 facts_per_entity 条一跳事实、最多 path_facts 条多跳关系链、子块最多 rerank_topk 条。

关系触发词不再 import 时从 yaml 读:由端点现查 extraction_dict(dict.load_relations)
后经 relation_defs 传入,改字典即时生效。
"""
from __future__ import annotations

from typing import Callable, Optional

from ..config import get
from ..embeddings import EmbeddingClient
from ..llm import LLMClient
from ..milvus_client import AsyncMilvusClient
from ..neo4j_client import Neo4jClient
from .answer import generate_answer
from .expand import expand
from .nl2query import nl2query
from .recall import dual_recall, match_entities, recall_expanded
from .rerank import rerank


# 关系型问句启发式:疑问词 + 关系动词,如「A国装备了哪些武器」。这类问法的答案
# 在图的边上(而非子块文本),须走图扩展分支拿结构化关系事实。字典触发词覆盖不到
# 「有哪些 Y」的关系问法时,靠这一层兜住。
_RELATION_QUESTION_MARKERS = (
    "哪些", "哪类", "何种", "什么", "多少", "谁",
    "哪个", "哪部", "哪艘", "哪家", "哪国", "哪里", "哪儿",
)
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
                              id2name: dict, kb_id=None,
                              expanded: Optional[list] = None) -> list[str]:
    """把命中实体的关系整理成结构化事实文本(看图答)。

    一跳:命中实体直接相连的边,即便对端是孤儿(无 BINDS 子块)也能答「X 有哪些 Y」。
    多跳:expanded 里经关系链到达的实体(带 path),写成整条链,如
          「A --[隶属]--> B <--[装备于]-- C」,让 LLM 看到跨实体的推理路径。
    方向:箭头始终指向关系边的真实方向。
    """
    facts: list[str] = []
    per_entity = int(get("query.facts_per_entity", 15))
    for eid in matched_ids:
        for r in await neo4j.entity_relations(eid, limit=per_entity, kb_id=kb_id):
            head = id2name.get(eid, eid)
            tgt = r["target_name"]
            if r.get("target_type"):
                tgt = f"{tgt}（{r['target_type']}）"
            if r["dir"] == "out":
                facts.append(f"{head} --[{r['rel_type']}]--> {tgt}")
            else:
                facts.append(f"{tgt} --[{r['rel_type']}]--> {head}")
    facts += path_facts(expanded or [], int(get("query.path_facts", 10)))
    return facts


def path_facts(expanded: list, limit: int) -> list[str]:
    """多跳关系链 -> 事实文本。只保留最长链:链 A→B→C 已包含 A→B,不重复输出前缀。"""
    with_path = [e for e in expanded if e.get("path")]
    with_path.sort(key=lambda e: -len(e["path"]))
    covered: set = set()
    chains: list[tuple[int, str]] = []
    for e in with_path:
        if e["name"] in covered:
            continue
        steps = e["path"]
        text = steps[0]["from"]
        for s in steps:
            covered.add(s["to"])
            text += (f" --[{s['rel_type']}]--> {s['to']}" if s["dir"] == "out"
                     else f" <--[{s['rel_type']}]-- {s['to']}")
        type_tag = f"（{e['entity_type']}）" if e.get("entity_type") else ""
        chains.append((len(steps), f"{text}{type_tag}（{len(steps)}跳关系链）"))
        if len(chains) >= limit:
            break
    chains.sort(key=lambda c: c[0])
    return [t for _, t in chains]


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

    entities = await neo4j.list_entities(kb_id=kb_id)
    matched_ids = match_entities(query, entities)

    rec = await dual_recall(query, milvus, embeddings, matched_ids, kb_id=kb_id)

    # 关系型:图扩展、收集一跳关系事实(看图答)、补召回扩展实体的绑定子块。
    # 补召回结果与 A/B 两路一起进 rerank 统一排序截断,不再在重排之后直接追加(防上下文膨胀)。
    expanded = []
    graph_facts: list[str] = []
    extra_hits: list[dict] = []
    if intent == "graph" and matched_ids:
        expanded = await expand(neo4j, matched_ids, kb_id=kb_id)
        id2name = {e["entity_id"]: e["name"] for e in entities}
        graph_facts = await collect_graph_facts(neo4j, matched_ids, id2name, kb_id=kb_id,
                                                expanded=expanded)
        extra_hits = await recall_expanded(query, milvus, embeddings,
                                           [e["entity_id"] for e in expanded], kb_id=kb_id)

    hits = rerank(rec["a"] + extra_hits, rec["b"], matched_ids=matched_ids)

    answer = await generate_answer(query, hits, llm, trace_hook, graph_facts=graph_facts)
    return {"intent": intent, "cond": cond, "answer": answer, "hits": hits,
            "matched_entities": matched_ids, "expanded_entities": expanded,
            "graph_facts": graph_facts, "kb_ids": kb_id if isinstance(kb_id, list) else ([str(kb_id)] if kb_id else None)}
