"""消歧模块的内存替身(独立验证用;集成时换成 PG / Neo4j / Milvus)。

- EntityRegistryStore ↔ PG entity_registry(名字/别名/指纹对照 + merged 账本)
- EntityVectorIndex   ↔ Milvus ems_entities(实体名向量,类型作用域 cosine 检索)
- GraphStore          ↔ Neo4j(:Entity / :BINDS / 关系,支持边迁移/删节点)
- DocIndexStore       ↔ PG parent_document.kg_entity_id + Milvus chunk.kg_entity_ids
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _union(a: list, b: list) -> list:
    out = list(a)
    for x in b:
        if x not in out:
            out.append(x)
    return out


@dataclass
class EntityRecord:
    entity_id: str
    name: str                      # 规范化名
    entity_type: str
    aliases: list = field(default_factory=list)   # 规范化别名
    properties: dict = field(default_factory=dict)
    fingerprint: Optional[str] = None
    created_order: int = 0         # 越小越早(集成时映射 created_at)
    status: str = "active"         # active | merged
    merged_into: Optional[str] = None


class EntityRegistryStore:
    """PG entity_registry 的内存版:类型作用域的 name/alias/fingerprint 对照。"""

    def __init__(self) -> None:
        self.entries: dict[str, EntityRecord] = {}
        self._counter = 0

    def _next_order(self) -> int:
        self._counter += 1
        return self._counter

    def upsert(self, record: EntityRecord) -> None:
        existing = self.entries.get(record.entity_id)
        if existing is None:
            record.created_order = record.created_order or self._next_order()
            self.entries[record.entity_id] = record
            return
        existing.aliases = _union(existing.aliases, record.aliases)
        if existing.fingerprint is None:
            existing.fingerprint = record.fingerprint
        for k, v in (record.properties or {}).items():
            existing.properties.setdefault(k, v)
        existing.status = "active"
        existing.merged_into = None

    def lookup_name_or_alias(self, entity_type: str, key: str) -> Optional[str]:
        best: Optional[EntityRecord] = None
        for r in self.entries.values():
            if r.status != "active" or r.entity_type != entity_type:
                continue
            if r.name == key or key in r.aliases:
                if best is None or r.created_order < best.created_order:
                    best = r
        return best.entity_id if best else None

    def lookup_fingerprint(self, entity_type: str, fp: str) -> Optional[str]:
        best: Optional[EntityRecord] = None
        for r in self.entries.values():
            if r.status != "active" or r.entity_type != entity_type or r.fingerprint != fp:
                continue
            if best is None or r.created_order < best.created_order:
                best = r
        return best.entity_id if best else None

    def mark_merged(self, old_id: str, new_id: str) -> None:
        r = self.entries.get(old_id)
        if r:
            r.status = "merged"
            r.merged_into = new_id

    def all_entries(self) -> list[EntityRecord]:
        return list(self.entries.values())


class EntityVectorIndex:
    """Milvus ems_entities 的内存版:类型作用域 brute-force cosine 检索。"""

    def __init__(self) -> None:
        self.records: dict[str, tuple] = {}   # entity_id -> (name, entity_type, vector)

    def upsert(self, entity_id: str, name: str, entity_type: str, vector: list) -> None:
        self.records[entity_id] = (name, entity_type, list(vector))

    def delete(self, entity_id: str) -> None:
        self.records.pop(entity_id, None)

    def get(self, entity_id: str) -> Optional[tuple]:
        return self.records.get(entity_id)

    def search(self, vector: list, entity_type: str, topk: int) -> list[tuple]:
        """返回 [(entity_id, name, score)] 按分降序,类型作用域。"""
        scored = [
            (eid, name, cosine_sim(vector, vec))
            for eid, (name, et, vec) in self.records.items() if et == entity_type
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:topk]


class GraphStore:
    """Neo4j 的内存版:实体 + BINDS + 关系,支持边迁移/删节点(幂等)。"""

    def __init__(self) -> None:
        self.entities: set[str] = set()
        self.binds: dict[tuple, str] = {}        # (entity_id, parent_id) -> role
        self.bind_conf: dict[tuple, float] = {}  # (entity_id, parent_id) -> confidence
        self.relations: list[tuple] = []         # (head, tail, rel_type)

    def add_entity(self, eid: str) -> None:
        self.entities.add(eid)

    def bind(self, eid: str, parent_id: str, role: str = "primary", conf: float = 1.0) -> None:
        self.add_entity(eid)
        self.binds[(eid, parent_id)] = role
        self.bind_conf[(eid, parent_id)] = conf

    def relate(self, head: str, tail: str, rel_type: str) -> None:
        self.add_entity(head)
        self.add_entity(tail)
        self.relations.append((head, tail, rel_type))

    def move_binds(self, old: str, new: str) -> int:
        moved = 0
        for (eid, pid), role in list(self.binds.items()):
            if eid != old:
                continue
            key = (new, pid)
            if key in self.binds:
                if role == "primary":
                    self.binds[key] = "primary"
                self.bind_conf[key] = max(self.bind_conf.get(key, 0.0), self.bind_conf[(old, pid)])
            else:
                self.binds[key] = role
                self.bind_conf[key] = self.bind_conf[(old, pid)]
            del self.binds[(old, pid)]
            self.bind_conf.pop((old, pid), None)
            moved += 1
        return moved

    def move_relations(self, old: str, new: str) -> int:
        moved = 0
        out: list[tuple] = []
        seen: set[tuple] = set()
        for (h, t, rt) in self.relations:
            hh = new if h == old else h
            tt = new if t == old else t
            if hh != h or tt != t:
                moved += 1
            if hh == tt:          # 自环丢弃
                continue
            key = (hh, tt, rt)
            if key in seen:       # 重边去重
                continue
            seen.add(key)
            out.append((hh, tt, rt))
        self.relations = out
        return moved

    def delete_entity(self, eid: str) -> None:
        self.entities.discard(eid)


class DocIndexStore:
    """PG kg_entity_id + Milvus kg_entity_ids 的内存版(逗号串/数组的 token 级重映射)。"""

    def __init__(self) -> None:
        self.parent_entities: dict[str, list] = {}   # parent_id -> [entity_id, ...]
        self.chunk_entities: dict[str, list] = {}    # child_id  -> [entity_id, ...]

    def set_parent(self, parent_id: str, eids: list) -> None:
        self.parent_entities[parent_id] = list(eids)

    def set_chunk(self, child_id: str, eids: list) -> None:
        self.chunk_entities[child_id] = list(eids)

    def remap(self, old: str, new: str) -> tuple[int, int]:
        p = c = 0
        for pid, eids in self.parent_entities.items():
            if old in eids:
                self.parent_entities[pid] = [new if x == old else x for x in eids]
                p += 1
        for cid, eids in self.chunk_entities.items():
            if old in eids:
                self.chunk_entities[cid] = [new if x == old else x for x in eids]
                c += 1
        return p, c
