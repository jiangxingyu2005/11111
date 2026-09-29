"""③-A Local 图遍历（M-R1）。

链路：实体链接（ems_entities 向量 KNN）-> 图 expand（度数截断 + 跳数上限）
      -> BINDS 回捞子块候选。
③-B 社区摘要路 deferred（IF-4，RETRIEVAL_COMMUNITY_ENABLED=false）。

双通道：real = Milvus KNN + Neo4j Cypher；fake = FakeStore（同接口、同配置）。
异常 -> 空列表 + graph_unavailable（降级矩阵 #7）。
无实体起点 -> 空列表且【不记降级】（正常缺席，不是故障）。
"""
from __future__ import annotations

import asyncio
import logging

from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from ..fuse.rrf import Hit

log = logging.getLogger("retrieval.recall.graph")


async def recall(query: str, *, scope: list[str], collector: DegradedCollector) -> list[Hit]:
    cfg = effective_cfg()
    gcfg = cfg.get("graph", {})
    if not get_settings().retrieval_graph_route_enabled:
        return []
    try:
        entity_ids = await link_entities(query, scope)
        if not entity_ids:
            return []  # 无实体起点 = 正常缺席（非降级）
        return await asyncio.wait_for(
            asyncio.to_thread(_expand_and_backfill, entity_ids, scope, gcfg),
            timeout=float(gcfg.get("budget_s", 0.6)),
        )
    except Exception as e:
        log.warning("graph route failed: %s", e)
        collector.record(DegradedCode.GRAPH_UNAVAILABLE, str(e)[:200], "recall_graph")
        return []


async def link_entities(query: str, scope: list[str], top_k: int | None = None) -> list[str]:
    """实体链接：query 向量 -> ems_entities KNN -> 卡阈值返回 entity_id。

    禁止全量拉实体（规模上去会打爆）；名称精确匹配作为 KNN 的补召回（可选，TODO）。
    """
    from .vector_route import _embed_query_full

    cfg = effective_cfg()
    gcfg = cfg.get("graph", {}).get("entity_link", {})
    k = int(top_k or gcfg.get("top_k", 10))
    tau = float(gcfg.get("threshold", 0.55))
    emb = await _embed_query_full(query)
    return _knn(emb["dense"], scope, k, tau)


def _knn(dense: list[float], scope: list[str], k: int, tau: float) -> list[str]:
    if effective_cfg().get("_backend") == "fake":
        from ..store.fake import get_fake_store
        rows = get_fake_store().knn_entities(dense, top_k=k, scope=scope, tau=tau)
        return [str(r["id"]) for r in rows]

    from pymilvus import MilvusClient  # 懒加载
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    scope_expr = "kb_id in [" + ",".join(f'"{x}"' for x in scope) + "]"
    res = client.search(
        collection_name=s.milvus_entity_collection,
        data=[dense],
        anns_field="embedding",
        search_params={"metric_type": "COSINE"},
        limit=k,
        filter=scope_expr,
        output_fields=["name", "type"],
    )
    return [str(r.get("entity_id") or r.get("id")) for r in res[0] if r["distance"] >= tau]


def _expand_and_backfill(entity_ids: list[str], scope: list[str], gcfg: dict) -> list[Hit]:
    """图遍历 + BINDS 回捞。"""
    cfg = effective_cfg()
    if cfg.get("_backend") == "fake":
        from ..store.fake import get_fake_store
        rows = get_fake_store().expand(
            entity_ids, scope,
            hops=int(gcfg.get("hops", 2)),
            max_degree=int(gcfg.get("max_degree", 50)),
            limit=int(gcfg.get("limit", 50)),
        )
        return [
            Hit(child_id=str(r["child_id"]), parent_id=str(r.get("parent_id", "")),
                rank_score=1.0, source="graph_local", content=r.get("content", ""),
                kg_entity_ids=r.get("kg_entity_ids", []),
                meta={"kb_id": r.get("kb_id", ""), "via_entity": r.get("via_entity", ""),
                      "span": r.get("span")})
            for r in rows
        ]

    from neo4j import GraphDatabase  # 懒加载
    s = get_settings()
    hops = int(gcfg.get("hops", 2))  # 变量长关系跳数不能参数化，须内联（配置受控整数）
    cypher = f"""
    MATCH (e:Entity)-[r*1..{hops}]-(n:Entity)<-[:BINDS]-(seg:Chunk)
    WHERE e.entity_id IN $eids AND seg.kb_id IN $scope
    WITH seg, size([(n)-[]-() | 1]) AS deg
    WHERE deg <= $max_deg
    RETURN DISTINCT seg.chunk_id AS child_id, seg.parent_id AS parent_id,
           seg.content AS content, seg.kb_id AS kb_id,
           seg.span_start AS span_start, seg.span_end AS span_end
    LIMIT $limit
    """
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        with driver.session() as sess:
            rows = sess.run(cypher, eids=entity_ids, scope=scope,
                            max_deg=int(gcfg.get("max_degree", 50)),
                            limit=int(gcfg.get("limit", 50)))
            return [
                Hit(child_id=str(r["child_id"]), parent_id=str(r["parent_id"] or ""),
                    rank_score=1.0, source="graph_local", content=r["content"] or "",
                    meta={"kb_id": r["kb_id"] or "", "span": _span_from_row(r)})
                for r in rows
            ]
    finally:
        driver.close()


def _span_from_row(r: dict) -> list[int] | None:
    """Neo4j 侧 Chunk 存 span_start/span_end，组合成证据锚点 [start, end]。"""
    if r.get("span_start") is not None and r.get("span_end") is not None:
        return [int(r["span_start"]), int(r["span_end"])]
    return None
