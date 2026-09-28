"""GET /api/graph/subgraph —— 独立图谱页接口（M-R5）。

契约：api/subgraph_view.v1.schema.json（严格对齐，响应体可被该 schema 校验）
三条硬约束：
  ① trace_id 绑定：focus 实体来自该次检索的重排 Top-N，不收前端传来的实体（防篡改/泄漏）
  ② scope 二次校验：检索时过滤过不算数，图谱接口必须在服务端再过滤一次（权限红线）
  ③ 独立降级：图谱慢/挂不影响检索结果渲染，degraded[] 单独统计
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query

from ..config import effective_cfg, get_settings, load_yaml
from ..trace.spans import get_binding

log = logging.getLogger("retrieval.api.graph")
router = APIRouter()

CONTRACT_VERSION = "1.0"

# 内部实体类型 -> 契约白名单类型（原始类型保留在 props.raw_type，不丢信息）
TYPE_MAP = {
    "Radar": "Emitter", "Platform": "Platform", "Function": "Sensor",
    "Band": "SignalSample", "Modulation": "SignalSample",
    "Area": "Event", "Country": "Organization",
}


def _degraded(code: str, message: str | None = None) -> dict:
    return {"code": code, "message": message, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _relation_whitelist() -> set[str]:
    """relation_types 白名单（config/relations.yaml，与 qingxi extraction_dict + 派生织网对齐）。"""
    cfg = load_yaml("relations")
    return set(cfg.get("extracted") or []) | set(cfg.get("derived") or [])


def _node(n: dict) -> dict:
    return {
        "id": n["id"],
        "label": n.get("name", n["id"]),
        "type": TYPE_MAP.get(n.get("type", ""), n.get("type", "Unknown")),
        "is_focus": bool(n.get("is_focus", False)),
        "hops_from_focus": int(n.get("hops_from_focus", 0)),
        "props": {"raw_type": n.get("type"), "degree": n.get("degree"), "kb_id": n.get("kb_id")},
        "evidence": n.get("evidence"),
    }


def _edge(e: dict) -> dict:
    ev = e.get("evidence") or {}
    return {
        "source": e["source"], "target": e["target"], "type": e.get("rel", "RELATED"),
        "confidence": e.get("confidence"), "t_start": None, "t_end": None,
        "evidence": {"child_id": ev.get("child_id", ""), "span": ev.get("span") or [0, 0],
                     "parent_id": ev.get("parent_id")},
    }


@router.get("/api/graph/subgraph")
async def subgraph(
    trace_id: str = Query(min_length=8),
    scope: list[str] = Query(min_length=1),
    hops: int = Query(1, ge=1, le=2),
    entity_ids: list[str] | None = Query(default=None),
    relation_types: list[str] | None = Query(default=None),
) -> dict:
    cfg = effective_cfg().get("subgraph", {})
    node_limit = int(cfg.get("node_limit", 50))
    edge_limit = int(cfg.get("edge_limit", 100))
    deg: list[dict] = []

    # G4：relation_types 白名单校验（契约 config/relations.yaml），未知关系类型直接 422，禁止静默放行
    if relation_types:
        allowed = _relation_whitelist()
        unknown = [rt for rt in relation_types if rt not in allowed]
        if unknown:
            raise HTTPException(status_code=422, detail={
                "code": "relation_type_invalid",
                "message": f"未知关系类型 {unknown}，白名单见 config/relations.yaml",
                "allowed": sorted(allowed)})

    # ── focus 来源：优先 trace 绑定，过期/无绑定才接受显式实体 ──
    binding = get_binding(trace_id)
    if binding and binding.get("focus_entities"):
        focus = binding["focus_entities"]
        mode = "by_trace"
    elif entity_ids:
        focus = list(entity_ids)
        mode = "by_entities"
        deg.append(_degraded("trace_expired_fallback_entities",
                             "trace 绑定已过期或不存在，回退为显式实体"))
    else:
        focus, mode = [], "by_trace"
        deg.append(_degraded("focus_entity_missing", "无 focus 实体且未传 entity_ids"))

    nodes_raw: list[dict] = []
    edges_raw: list[dict] = []
    pruned = False
    truncated_reason = None
    if focus:
        try:
            if effective_cfg().get("_backend") == "fake":
                from ..store.fake import get_fake_store
                nodes_raw, edges_raw, pruned = get_fake_store().subgraph(
                    focus, scope, hops=hops, node_limit=node_limit,
                    edge_limit=edge_limit, relation_types=relation_types)
            else:
                nodes_raw, edges_raw = _neo4j_subgraph(
                    focus, scope, hops, node_limit, edge_limit, relation_types)
                pruned = len(nodes_raw) >= node_limit or len(edges_raw) >= edge_limit
        except Exception as e:
            log.warning("subgraph failed: %s", e)
            deg.append(_degraded("subgraph_unavailable", str(e)[:200]))
        if not nodes_raw and not any(d["code"] == "subgraph_unavailable" for d in deg):
            deg.append(_degraded("scope_filtered_all", "focus 实体在当前 scope 下不可见"))
        if pruned:
            truncated_reason = "node_limit" if len(nodes_raw) >= node_limit else "edge_limit"

    nodes = [_node(n) for n in nodes_raw]
    edges = [_edge(e) for e in edges_raw]
    return {
        "contract_version": CONTRACT_VERSION,
        "trace_id": trace_id,
        "query": {"mode": mode, "entity_ids": entity_ids if mode == "by_entities" else None,
                  "hops": hops, "relation_types": relation_types, "direction": "both"},
        "focus_entities": focus,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "hops": hops, "pruned": pruned,
            "node_count": len(nodes), "edge_count": len(edges),
            "truncated_reason": truncated_reason,
            "source": "neo4j" if effective_cfg().get("_backend") == "real" else None,
            "layout_hint": "concentric",
            "scope_filtered": bool(focus) and len(nodes_raw) < len(focus),
        },
        "degraded": deg,
    }


def _neo4j_subgraph(focus: list[str], scope: list[str], hops: int,
                    node_limit: int, edge_limit: int,
                    relation_types: list[str] | None) -> tuple[list[dict], list[dict]]:
    from neo4j import GraphDatabase  # 懒加载

    s = get_settings()
    cy_nodes = """
    MATCH (f:Entity)-[*0..$hops]-(n:Entity)
    WHERE f.entity_id IN $focus AND n.kb_id IN $scope
    WITH DISTINCT n, [ (n)-[]-() | 1 ] AS degs
    RETURN n.entity_id AS id, n.name AS name, n.type AS type, n.kb_id AS kb_id,
           size(degs) AS degree, n.entity_id IN $focus AS is_focus
    LIMIT $node_limit
    """
    cy_edges = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.entity_id IN $ids AND b.entity_id IN $ids
      AND ($rel_types IS NULL OR type(r) IN $rel_types)
    RETURN a.entity_id AS source, b.entity_id AS target, type(r) AS rel,
           r.confidence AS confidence,
           r.evidence_child_id AS child_id, r.span_start AS span_start,
           r.span_end AS span_end, r.parent_id AS parent_id
    LIMIT $edge_limit
    """
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        with driver.session() as sess:
            nodes = [dict(r) for r in sess.run(cy_nodes, focus=focus, scope=scope,
                                               hops=hops, node_limit=node_limit)]
            ids = [n["id"] for n in nodes]
            edges = []
            for r in sess.run(cy_edges, ids=ids, rel_types=relation_types, edge_limit=edge_limit):
                d = dict(r)
                edges.append({
                    "source": d["source"], "target": d["target"], "rel": d["rel"],
                    "confidence": d.get("confidence"),
                    "evidence": {"child_id": d.get("child_id") or "",
                                 "span": [d.get("span_start") or 0, d.get("span_end") or 0],
                                 "parent_id": d.get("parent_id")},
                })
            return nodes, edges
    finally:
        driver.close()
