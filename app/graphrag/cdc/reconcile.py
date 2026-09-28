"""CDC-02:对账脚本(ems cdc/reconcile.py 的异步移植)。

比对 PG parent_document ↔ Neo4j Segment 集合;并核对绑定数。
任意操作后跑本脚本,图与 PG 应一致。
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ParentDocument
from .. import db
from ..neo4j_client import Neo4jClient


async def reconcile(session: AsyncSession, neo4j: Neo4jClient) -> dict:
    parents = await db.select_all(session, ParentDocument)
    pg_ids = {p["parent_id"] for p in parents}
    graph_ids = set(await neo4j.list_segment_ids())

    missing_in_graph = sorted(pg_ids - graph_ids)          # PG 有、图无
    orphan_in_graph = sorted(graph_ids - pg_ids)           # 图有、PG 无

    graph_binds = 0
    for pid in graph_ids:
        graph_binds += len(await neo4j.entity_ids_of_parent(pid))

    return {
        "consistent": not missing_in_graph and not orphan_in_graph,
        "pg_parents": len(pg_ids),
        "graph_segments": len(graph_ids),
        "missing_in_graph": missing_in_graph,
        "orphan_in_graph": orphan_in_graph,
        "graph_binds": graph_binds,
    }
