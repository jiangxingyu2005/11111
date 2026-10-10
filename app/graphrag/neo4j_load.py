"""GP-05 NEO4J-01:三步导入 + validate_foreign_ids 前置校验(ems graph/neo4j_load.py 移植,免 CSV 直对象 + async)。

三步:① 实体节点 ② 父块 Segment 节点 + BINDS 边 ③ 关系边。
前置校验:任何表外 ID 抛错并零节点写入。MERGE 保证重跑幂等。
"""

from __future__ import annotations

from .models import Bind, Entity, Relation
from .neo4j_client import Neo4jClient


def validate_foreign_ids(entities: list[dict], relations: list[dict],
                         binds: list[dict], parents: list[dict]) -> list[str]:
    """表外 ID 校验:引用的 parent_id/entity_id 必须存在于父块集/实体集。"""
    parent_ids = {p["parent_id"] for p in parents}
    entity_ids = {e["entity_id"] for e in entities}
    errors: list[str] = []
    for e in entities:
        if e.get("source_parent_id") and e["source_parent_id"] not in parent_ids:
            errors.append(f"entities 表外 parent_id: {e['source_parent_id']}")
    for b in binds:
        if b["parent_id"] not in parent_ids:
            errors.append(f"binds 表外 parent_id: {b['parent_id']}")
        if b["entity_id"] not in entity_ids:
            errors.append(f"binds 表外 entity_id: {b['entity_id']}")
    for r in relations:
        if r.get("source_parent_id") and r["source_parent_id"] not in parent_ids:
            errors.append(f"relations 表外 parent_id: {r['source_parent_id']}")
        if r["head_entity_id"] not in entity_ids:
            errors.append(f"relations 表外 head entity_id: {r['head_entity_id']}")
        if r["tail_entity_id"] not in entity_ids:
            errors.append(f"relations 表外 tail entity_id: {r['tail_entity_id']}")
    return errors


async def import_graph(neo4j: Neo4jClient, entities: list[Entity], relations: list[Relation],
                       binds: list[Bind], parents: list[dict]) -> dict:
    entity_dicts = [e.to_dict() for e in entities]
    relation_dicts = [r.to_dict() for r in relations]
    bind_dicts = [b.to_dict() for b in binds]

    errors = validate_foreign_ids(entity_dicts, relation_dicts, bind_dicts, parents)
    if errors:
        raise ValueError("表外 ID 校验失败,零节点写入:\n" + "\n".join(errors))

    # ① 实体节点(合并写:跨文档对齐到已有实体时不覆盖先入库文档的名称/属性/别名)
    for e in entity_dicts:
        await neo4j.merge_entity(e, overwrite=False)
    # ② 父块 Segment + BINDS
    for p in parents:
        await neo4j.merge_segment(p)
    for b in bind_dicts:
        await neo4j.merge_binds(b["entity_id"], b["parent_id"], b["kg_role"], b["kg_confidence"])
    # ③ 关系边
    for r in relation_dicts:
        await neo4j.merge_relation(r)

    return {"entities": len(entities), "segments": len(parents),
            "binds": len(binds), "relations": len(relations)}
