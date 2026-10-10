"""GP-08 NEO4J-02:图扩展(Plan §3.9,async 移植)。

从命中实体出发做多跳扩展。共享实体不误伤:实体节点不删,禁用分段仅令其子块在
召回层失效,不影响经其它分段可达性(故 ems 里按 disabled_segment_ids 过滤的
`parents_of_entity` 分支是死代码,融合时移除)。
"""
from __future__ import annotations

from typing import Optional

from ..config import get
from ..neo4j_client import Neo4jClient


async def expand(neo4j: Neo4jClient, entity_ids: list,
                 depth: Optional[int] = None, limit: Optional[int] = None,
                 kb_id=None) -> list[dict]:
    """图扩展。返回关联实体(含 hops;经关系链到达的带 path),最多 expand_limit 个,
    按 hop_quotas 分配每跳名额;传 kb_id 只在该库内扩展。"""
    depth = depth or int(get("query.expand_depth", 2))
    limit = limit or int(get("query.expand_limit", 30))
    quotas = get("query.hop_quotas", None) or None
    return await neo4j.expand(entity_ids, depth, limit, kb_id=kb_id, hop_quotas=quotas)
