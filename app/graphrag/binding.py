"""GP-05 META-02:绑定回写(ems meta/binding.py 移植,免 CSV 直对象 + async)。

导入 Neo4j 后:
  1. 回写 PG parent_document:kg_entity_id / kg_role / kg_confidence(仅 primary)
  2. Milvus 子块 kg_entity_ids = 父块 primary 实体集合(部分更新)
三方一致(PG 绑定行 ↔ Neo4j BINDS 边 ↔ Milvus 数组)。
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ParentDocument
from . import db
from .milvus_client import AsyncMilvusClient
from .models import Bind, Entity


def build_binds(entities: list[Entity]) -> list[Bind]:
    """生成实体 -> 父块绑定。主归属(source_parent_id)为 primary(仅 primary 回写)。"""
    binds = []
    for e in entities:
        if e.source_parent_id:
            binds.append(Bind(entity_id=e.entity_id, parent_id=e.source_parent_id,
                              kg_role="primary", kg_confidence=e.confidence, evidence=""))
    return binds


async def bind_back(session: AsyncSession, milvus: AsyncMilvusClient,
                    binds: list[Bind], parents: list[dict], children: list[dict]) -> dict:
    primary_by_parent: dict[str, list] = defaultdict(list)
    conf_by_parent: dict[str, float] = defaultdict(float)
    for b in binds:
        if b.kg_role == "primary":
            primary_by_parent[b.parent_id].append(b.entity_id)
            conf_by_parent[b.parent_id] = max(conf_by_parent[b.parent_id], b.kg_confidence)

    # 回写 PG(三字段)
    parent_ids = {p["parent_id"] for p in parents}
    bound = 0
    for parent_id, eids in primary_by_parent.items():
        if parent_id not in parent_ids:
            continue
        await db.update(session, ParentDocument, {
            "parent_id": parent_id, "kg_entity_id": ",".join(eids),
            "kg_role": "primary", "kg_confidence": conf_by_parent[parent_id],
        }, "parent_id")
        bound += 1

    # 回写 Milvus:子块 kg_entity_ids = 父块 primary 集合
    for c in children:
        await milvus.update_kg_entity_ids(c["child_id"], primary_by_parent.get(c["parent_id"], []))

    return {"parents_bound": bound, "children_updated": len(children)}
