"""③-B 社区摘要路（M-R2）。

链路：Scene KNN（community_scene 集合）-> 命中社区取代表实体 -> 图 expand（度数截断 + 跳数上限）
      -> BINDS 回捞子块候选。
与 ③-A Local 图遍历并列并行（RETRIEVAL_COMMUNITY_ENABLED=true 时）；意图路由（GLOBAL/HYBRID）
留待 L2 落地后细化——当前 query_parser 尚未产出 GLOBAL/HYBRID 意图。

双通道：real = Milvus community_scene KNN + Neo4j Cypher；fake = FakeStore（同接口、同配置）。
降级五码（#8~12）：
  community_unavailable(#8)        整路异常 -> 空 + 记录
  community_no_start(#9)           Scene KNN 无过阈值社区 -> 空 + 记录
  community_stale(#10)             单社区过期 -> 跳过 + 记录，继续其余
  community_summary_partial(#11)   单社区摘要缺失 -> 跳过 + 记录，继续其余
  community_no_evidence(#12)       单社区无证据回捞 -> 跳过 + 记录，继续其余
"""
from __future__ import annotations

import asyncio
import logging

from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from ..fuse.rrf import Hit

log = logging.getLogger("retrieval.recall.community")


async def recall(query: str, *, scope: list[str], collector: DegradedCollector) -> list[Hit]:
    cfg = effective_cfg()
    ccfg = cfg.get("community", {})
    if not get_settings().retrieval_community_enabled:
        return []
    try:
        from .vector_route import _embed_query_full
        emb = await _embed_query_full(query)
        communities = await asyncio.wait_for(
            asyncio.to_thread(_scene_knn, emb["dense"], scope, ccfg),
            timeout=float(ccfg.get("budget_s", 0.8)),
        )
        if not communities:
            collector.record(DegradedCode.COMMUNITY_NO_START,
                             "Scene KNN 无过阈值社区", "recall_community")
            return []
        return _backfill(communities, scope, ccfg, collector)
    except Exception as e:
        log.warning("community route failed: %s", e)
        collector.record(DegradedCode.COMMUNITY_UNAVAILABLE, str(e)[:200], "recall_community")
        return []


def _scene_knn(dense: list[float], scope: list[str], ccfg: dict) -> list[dict]:
    """Scene KNN：query 向量 -> community_scene 最近邻（卡 τ_s）。"""
    cfg = effective_cfg()
    s_link = ccfg.get("scene_link", {})
    k = int(s_link.get("top_k", 5))
    tau = float(s_link.get("threshold", 0.55))
    if cfg.get("_backend") == "fake":
        from ..store.fake import get_fake_store
        rows = get_fake_store().community_knn(dense, top_k=k, scope=scope, tau=tau)
        return [{"community_id": str(r["id"]), "score": float(r["distance"]),
                 "summary": r["entity"].get("summary", ""),
                 "member_entity_ids": r["entity"].get("member_entity_ids", []),
                 "stale": bool(r["entity"].get("stale", False))}
                for r in rows]

    from pymilvus import MilvusClient  # 懒加载
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    scope_expr = "kb_id in [" + ",".join(f'"{x}"' for x in scope) + "]"
    res = client.search(
        collection_name=s.milvus_community_collection,
        data=[dense],
        anns_field="scene_vector",
        search_params={"metric_type": "COSINE"},
        limit=k,
        filter=scope_expr,
        output_fields=["summary", "member_entity_ids", "stale", "kb_id"],
    )
    return [{"community_id": str(r.get("community_id") or r.get("id")),
             "score": float(r["distance"]),
             "summary": (r["entity"].get("summary") or ""),
             "member_entity_ids": r["entity"].get("member_entity_ids") or [],
             "stale": bool(r["entity"].get("stale", False))}
            for r in res[0] if r["distance"] >= tau]


def _backfill(communities: list[dict], scope: list[str], ccfg: dict,
              collector: DegradedCollector) -> list[Hit]:
    """社区 -> 代表实体 -> 图 expand -> BINDS 回捞子块（source=graph_global）。"""
    hits: list[Hit] = []
    for comm in communities:
        cid = comm["community_id"]
        if comm.get("stale"):
            collector.record(DegradedCode.COMMUNITY_STALE, f"{cid} stale", "recall_community")
            continue
        if not comm.get("summary"):
            collector.record(DegradedCode.COMMUNITY_SUMMARY_PARTIAL,
                             f"{cid} summary empty", "recall_community")
            continue
        members = comm.get("member_entity_ids") or []
        if not members:
            collector.record(DegradedCode.COMMUNITY_NO_EVIDENCE,
                             f"{cid} no member entities", "recall_community")
            continue
        rows = _expand(members, scope, ccfg)
        if not rows:
            collector.record(DegradedCode.COMMUNITY_NO_EVIDENCE,
                             f"{cid} no evidence chunks", "recall_community")
            continue
        for r in rows:
            hits.append(Hit(child_id=str(r["child_id"]), parent_id=str(r.get("parent_id", "")),
                            rank_score=float(comm.get("score", 1.0)), source="graph_global",
                            content=r.get("content", ""),
                            kg_entity_ids=r.get("kg_entity_ids", []),
                            meta={"kb_id": r.get("kb_id", ""),
                                  "span": r.get("span"),
                                  "via_community": cid}))
    return hits


def _expand(entity_ids: list[str], scope: list[str], ccfg: dict) -> list[dict]:
    """图遍历 + BINDS 回捞（对齐 graph_route._expand_and_backfill 的语义）。"""
    cfg = effective_cfg()
    if cfg.get("_backend") == "fake":
        from ..store.fake import get_fake_store
        return get_fake_store().expand(
            entity_ids, scope,
            hops=int(ccfg.get("hops", 2)),
            max_degree=int(ccfg.get("max_degree", 50)),
            limit=int(ccfg.get("limit", 50)),
        )

    from neo4j import GraphDatabase  # 懒加载
    s = get_settings()
    hops = int(ccfg.get("hops", 2))  # 变量长关系跳数不能参数化，须内联（配置受控整数）
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
                            max_deg=int(ccfg.get("max_degree", 50)),
                            limit=int(ccfg.get("limit", 50)))
            return [
                {"child_id": str(r["child_id"]), "parent_id": str(r["parent_id"] or ""),
                 "content": r["content"] or "", "kb_id": r["kb_id"] or "",
                 "kg_entity_ids": [],
                 "span": [int(r["span_start"]), int(r["span_end"])]
                 if r.get("span_start") is not None and r.get("span_end") is not None
                 else None}
                for r in rows
            ]
    finally:
        driver.close()
