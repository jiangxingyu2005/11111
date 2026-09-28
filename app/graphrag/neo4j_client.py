"""Neo4j 客户端(ems common/neo4j_client.py 的异步移植,neo4j.asyncio)。

图模型:(:Entity) 实体、(:Segment) 父块、(:Entity)-[:BINDS]->(:Segment)、
(:Entity)-[:BELONGS_TO|SUPPORTS|CO_LINK|INVOLVES]->(:Entity)。
融合新增:Segment 带 kb_id(知识库隔离)。写入 MERGE 保证幂等。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import build_settings

CONSTRAINTS_CYPHER = Path(__file__).resolve().parent / "sql" / "03_neo4j_constraints.cypher"


class Neo4jClient:
    """neo4j.asyncio 包装。写入用 MERGE 保证幂等。"""

    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or build_settings()
        self._driver = None

    async def connect(self) -> None:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase  # 延迟导入
            s = self.settings["neo4j"]
            self._driver = AsyncGraphDatabase.driver(s["uri"], auth=(s["user"], s["password"]))

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def run(self, cypher: str, **params) -> list[dict]:
        await self.connect()
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            return await result.data()

    async def run_constraints(self) -> None:
        text = CONSTRAINTS_CYPHER.read_text(encoding="utf-8") if CONSTRAINTS_CYPHER.exists() else ""
        for stmt in text.split(";"):
            stmt = stmt.strip()
            if stmt:
                await self.run(stmt)

    # ---- 三步导入 ----
    async def merge_entity(self, entity: dict) -> None:
        # Neo4j 不接受嵌套 map 作为属性值,properties 序列化为 JSON 字符串落库
        props = entity["properties"]
        if not isinstance(props, str):
            props = json.dumps(props, ensure_ascii=False)
        await self.run(
            "MERGE (e:Entity {entity_id: $entity_id}) "
            "SET e.name = $name, e.entity_type = $entity_type, "
            "e.aliases = $aliases, e.properties = $properties, e.confidence = $confidence",
            entity_id=entity["entity_id"], name=entity["name"],
            entity_type=entity["entity_type"], aliases=entity.get("aliases", []),
            properties=props, confidence=entity.get("confidence", 1.0),
        )

    async def merge_segment(self, parent: dict) -> None:
        await self.run(
            "MERGE (s:Segment {parent_id: $parent_id}) "
            "SET s.doc_id = $doc_id, s.kb_id = $kb_id, s.title = $title, "
            "s.enabled = $enabled, s.chapter_no = $chapter_no",
            parent_id=parent["parent_id"], doc_id=parent.get("doc_id", ""),
            kb_id=str(parent["kb_id"]) if parent.get("kb_id") else None,
            title=parent.get("title", ""), enabled=bool(parent.get("enabled", True)),
            chapter_no=parent.get("chapter_no", ""),
        )

    async def merge_binds(self, entity_id: str, parent_id: str, role: str, confidence: float) -> None:
        await self.run(
            "MATCH (e:Entity {entity_id: $entity_id}) "
            "MATCH (s:Segment {parent_id: $parent_id}) "
            "MERGE (e)-[b:BINDS]->(s) "
            "SET b.kg_role = $role, b.kg_confidence = $confidence",
            entity_id=entity_id, parent_id=parent_id, role=role, confidence=confidence,
        )

    async def merge_relation(self, rel: dict) -> None:
        # rel_type 取自字典表(或 mesh 派生类型 SAME_BAND/CO_TIME/CO_LOCATE),为合法 Cypher 关系标识
        await self.run(
            "MATCH (a:Entity {entity_id: $head}) MATCH (b:Entity {entity_id: $tail}) "
            f"MERGE (a)-[r:{rel['rel_type']}]->(b) "
            "SET r.confidence = $confidence, r.evidence = $evidence, r.source_parent_id = $source_parent_id, "
            "r.derived = $derived",
            head=rel["head_entity_id"], tail=rel["tail_entity_id"],
            confidence=rel.get("confidence", 0.0), evidence=rel.get("evidence", ""),
            source_parent_id=rel.get("source_parent_id"),
            derived=bool(rel.get("derived", False)),
        )

    async def detach_delete_segment(self, parent_id: str) -> None:
        await self.run("MATCH (s:Segment {parent_id: $parent_id}) DETACH DELETE s", parent_id=parent_id)

    async def delete_bind(self, entity_id: str, parent_id: str) -> None:
        await self.run(
            "MATCH (e:Entity {entity_id: $entity_id})-[b:BINDS]->(s:Segment {parent_id: $parent_id}) DELETE b",
            entity_id=entity_id, parent_id=parent_id,
        )

    # ---- 全量对账:实体归并四层之图回填 ----
    async def move_binds(self, old: str, new: str) -> int:
        """把 old 的 :BINDS 边迁移到 new(MERGE 幂等),返迁移边数。"""
        rows = await self.run(
            "MATCH (e:Entity {entity_id: $old})-[b:BINDS]->(s:Segment) "
            "RETURN s.parent_id AS parent_id, b.kg_role AS kg_role, b.kg_confidence AS kg_confidence",
            old=old,
        )
        for r in rows:
            await self.merge_binds(new, r["parent_id"],
                                   r.get("kg_role") or "supplemental",
                                   float(r.get("kg_confidence") or 0.0))
        return len(rows)

    async def move_relations(self, old: str, new: str) -> int:
        """关系边 head/tail 的 old -> new 重映射;自环丢弃。返重建边数。"""
        rows = await self.run(
            "MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) <> 'BINDS' "
            "AND (a.entity_id = $old OR b.entity_id = $old) "
            "RETURN a.entity_id AS head, b.entity_id AS tail, type(r) AS rel_type, "
            "r.confidence AS confidence, r.evidence AS evidence, "
            "r.source_parent_id AS source_parent_id, r.derived AS derived",
            old=old,
        )
        count = 0
        for r in rows:
            head = new if r["head"] == old else r["head"]
            tail = new if r["tail"] == old else r["tail"]
            if head == tail:
                continue
            await self.merge_relation({
                "head_entity_id": head, "tail_entity_id": tail,
                "rel_type": r["rel_type"],
                "confidence": float(r.get("confidence") or 0.0),
                "evidence": r.get("evidence") or "",
                "source_parent_id": r.get("source_parent_id"),
                "derived": bool(r.get("derived", False)),
            })
            count += 1
        # 删除 old 的全部关系边(自环与已迁移的都清掉)
        await self.run(
            "MATCH (a:Entity {entity_id: $old})-[r]->() WHERE type(r) <> 'BINDS' DELETE r", old=old)
        await self.run(
            "MATCH ()-[r]->(a:Entity {entity_id: $old}) WHERE type(r) <> 'BINDS' DELETE r", old=old)
        return count

    async def delete_entity(self, entity_id: str) -> None:
        """DETACH DELETE 实体节点(残留 BINDS/关系一并清除)。"""
        await self.run("MATCH (e:Entity {entity_id: $entity_id}) DETACH DELETE e",
                       entity_id=entity_id)

    # ---- 查询 ----
    async def list_entities(self, name: Optional[str] = None,
                            entity_type: Optional[str] = None) -> list[dict]:
        """按名称(含别名)/类型查实体,供在线查询做实体匹配。实体全局共享、不分知识库。"""
        cypher = "MATCH (e:Entity)"
        clauses: list[str] = []
        params: dict = {}
        if name:
            clauses.append("(e.name CONTAINS $name OR $name IN e.aliases)")
            params["name"] = name
        if entity_type:
            clauses.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if clauses:
            cypher += " WHERE " + " AND ".join(clauses)
        cypher += (" RETURN e.entity_id AS entity_id, e.name AS name, "
                   "e.entity_type AS entity_type, e.aliases AS aliases")
        return await self.run(cypher, **params)

    async def entity_ids_of_parent(self, parent_id: str) -> list[str]:
        """返回某父块 :Segment 绑定的实体 ID 集合(供编辑差量校正 / 对账)。"""
        rows = await self.run(
            "MATCH (e:Entity)-[:BINDS]->(s:Segment {parent_id: $parent_id}) "
            "RETURN e.entity_id AS entity_id",
            parent_id=parent_id,
        )
        return [r["entity_id"] for r in rows]

    async def list_segment_ids(self) -> list[str]:
        """返回图内全部 :Segment 的 parent_id(供对账)。"""
        rows = await self.run("MATCH (s:Segment) RETURN s.parent_id AS parent_id")
        return [r["parent_id"] for r in rows]

    # ---- 全图遍历(供 mesh 指标/兜底)----
    async def list_all_entities(self) -> list[dict]:
        """返回全部实体(轻量:id/name/type),供 compute_metrics 统计。"""
        return await self.run(
            "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.name AS name, "
            "e.entity_type AS entity_type"
        )

    async def list_all_entities_full(self) -> list[dict]:
        """返回全部实体(含 aliases/properties),供 EntityRegistry 建索引。

        properties 落库时为 JSON 字符串,这里还原为 dict 供 fingerprint 用。
        """
        rows = await self.run(
            "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.name AS name, "
            "e.aliases AS aliases, e.properties AS properties"
        )
        for r in rows:
            if isinstance(r.get("properties"), str):
                try:
                    r["properties"] = json.loads(r["properties"]) if r["properties"] else {}
                except json.JSONDecodeError:
                    r["properties"] = {}
            if not isinstance(r.get("aliases"), list):
                r["aliases"] = []
        return rows

    async def list_all_binds(self) -> list[dict]:
        """返回全部 BINDS 边(entity_id -> parent_id)。"""
        return await self.run(
            "MATCH (e:Entity)-[b:BINDS]->(s:Segment) "
            "RETURN e.entity_id AS entity_id, s.parent_id AS parent_id"
        )

    async def list_all_relations(self) -> list[dict]:
        """返回全部关系边(不含 BINDS)。"""
        return await self.run(
            "MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) <> 'BINDS' "
            "RETURN a.entity_id AS head_entity_id, b.entity_id AS tail_entity_id, "
            "type(r) AS rel_type"
        )

    # ---- 图扩展 ----
    async def expand(self, entity_ids: list, depth: int) -> list[dict]:
        if not entity_ids:
            return []
        # Neo4j 5.26 可变长度边界不接受参数,须是字面量;depth 为受控整数(配置项)
        depth = int(depth)
        return await self.run(
            f"MATCH p=(a:Entity)-[*1..{depth}]-(b:Entity) "
            "WHERE a.entity_id IN $ids "
            "RETURN DISTINCT b.entity_id AS entity_id, b.name AS name, "
            "b.entity_type AS entity_type, length(p) AS hops",
            ids=list(entity_ids),
        )


def get_neo4j() -> Neo4jClient:
    return Neo4jClient()
