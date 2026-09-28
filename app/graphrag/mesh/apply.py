"""入库「对齐」入口(纯同步):归一消歧 + 派生边 + 保真绑定。

在抽取后、Neo4j 导入前一次调用,把 ems mesh 的图增强接进 ingest:
  1. 归一消歧:别名/指纹重复的实体归一到 canonical entity_id(记录 canonical -> 原父块集合,不丢绑定)
  2. 关系重映射:head/tail 用 canonical 重映射,自环丢弃
  3. 派生边:SAME_BAND/CO_TIME/CO_LOCATE(derived=True)
  4. 保真绑定:canonical 实体对多个父块各发一条 primary 绑定(替代 build_binds 的单 source_parent_id)

mock 路径下 normalize/derive 为空操作(无别名/指纹/事件元数据),行为与 Phase 3 一致。
"""
from __future__ import annotations

from ..models import Bind, Entity, Relation
from .derived import derive_relations
from .registry import EntityRegistry


def enhance_graph(entities: list[Entity], relations: list[Relation]) -> tuple[list[Entity], list[Relation], list[Bind]]:
    """归一合并实体、重映射关系、织派生边、生成保真绑定。"""
    if not entities:
        return [], [], []

    # 1. 归一消歧:旧 entity_id -> canonical entity_id
    e_dicts = [e.to_dict() for e in entities]
    reg = EntityRegistry(e_dicts)
    canonical: dict[str, str] = {}
    for e in e_dicts:
        canonical[e["entity_id"]] = reg.resolve(name=e["name"], properties=e.get("properties")) or e["entity_id"]

    # canonical -> 原 source_parent_id 集合(归一不丢父块绑定)
    parent_map: dict[str, set] = {}
    for e in entities:
        if e.source_parent_id:
            parent_map.setdefault(canonical[e.entity_id], set()).add(e.source_parent_id)

    # 合并实体(按 canonical),合并别名;首现者字段为准(与 normalize_entities 一致)
    merged: dict[str, Entity] = {}
    for e in entities:
        cid = canonical[e.entity_id]
        if cid not in merged:
            merged[cid] = Entity(
                entity_id=cid, name=e.name, entity_type=e.entity_type,
                aliases=list(e.aliases), properties=dict(e.properties),
                confidence=e.confidence, source_parent_id=e.source_parent_id,
            )
        else:
            for a in list(e.aliases) + [e.name]:
                if a not in merged[cid].aliases:
                    merged[cid].aliases.append(a)
    out_entities = list(merged.values())

    # 2. 关系重映射(head/tail 用 canonical;重映射后自环丢弃)
    remapped: list[Relation] = []
    for r in relations:
        h = canonical.get(r.head_entity_id, r.head_entity_id)
        t = canonical.get(r.tail_entity_id, r.tail_entity_id)
        if h == t:
            continue
        remapped.append(Relation(
            relation_id=r.relation_id, head_entity_id=h, tail_entity_id=t,
            rel_type=r.rel_type, confidence=r.confidence, evidence=r.evidence,
            source_parent_id=r.source_parent_id, derived=r.derived,
        ))

    # 3. 派生边(SAME_BAND/CO_TIME/CO_LOCATE)
    for d in derive_relations([e.to_dict() for e in out_entities]):
        remapped.append(Relation(
            relation_id=d["relation_id"], head_entity_id=d["head_entity_id"],
            tail_entity_id=d["tail_entity_id"], rel_type=d["rel_type"],
            confidence=d["confidence"], evidence=d["evidence"],
            source_parent_id=d["source_parent_id"], derived=True,
        ))

    # 4. 保真绑定:canonical -> 各父块 primary
    binds: list[Bind] = []
    conf_by_cid = {e.entity_id: e.confidence for e in out_entities}
    for cid, parents in parent_map.items():
        conf = conf_by_cid.get(cid, 1.0)
        for pid in parents:
            binds.append(Bind(entity_id=cid, parent_id=pid,
                              kg_role="primary", kg_confidence=conf, evidence=""))

    return out_entities, remapped, binds
