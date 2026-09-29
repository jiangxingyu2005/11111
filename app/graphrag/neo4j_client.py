"""Neo4j 客户端(ems common/neo4j_client.py 的异步移植,neo4j.asyncio)。

图模型:(:Entity) 实体、(:Segment) 父块、(:Entity)-[:BINDS]->(:Segment)、
(:Entity)-[:BELONGS_TO|SUPPORTS|CO_LINK|INVOLVES]->(:Entity)。
融合新增:Segment 带 kb_id(知识库隔离)。写入 MERGE 保证幂等。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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

    # ---- 直接查询/管理(供 /graph 下的 Neo4j 查询与修改接口) ----
    async def list_entity_nodes(self, name: Optional[str] = None,
                                entity_type: Optional[str] = None,
                                kb_id: Optional[str] = None,
                                limit: Optional[int] = None,
                                offset: int = 0) -> list[dict]:
        """列出实体节点(轻量:id/name/type/aliases/confidence)。

        kb_id 传入时仅返回绑定到该库 :Segment 的实体(实体全局共享、不分知识库,
        归属靠 BINDS 边关联的 Segment.kb_id 判定)。
        """
        cypher = "MATCH (e:Entity)"
        clauses: list[str] = []
        params: dict = {}
        if name:
            clauses.append("(e.name CONTAINS $name OR $name IN e.aliases)")
            params["name"] = name
        if entity_type:
            clauses.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if kb_id:
            clauses.append("EXISTS { MATCH (e)-[:BINDS]->(s:Segment {kb_id: $kb_id}) }")
            params["kb_id"] = str(kb_id)
        if clauses:
            cypher += " WHERE " + " AND ".join(clauses)
        cypher += (" RETURN e.entity_id AS entity_id, e.name AS name, "
                   "e.entity_type AS entity_type, e.aliases AS aliases, "
                   "e.confidence AS confidence ORDER BY e.name")
        if limit is not None:
            cypher += f" SKIP {int(offset)} LIMIT {int(limit)}"
        return await self.run(cypher, **params)

    async def get_entity_node(self, entity_id: str) -> Optional[dict]:
        """按 entity_id 取单个实体节点(含 properties,JSON 字符串未还原)。"""
        rows = await self.run(
            "MATCH (e:Entity {entity_id: $entity_id}) "
            "RETURN e.entity_id AS entity_id, e.name AS name, "
            "e.entity_type AS entity_type, e.aliases AS aliases, "
            "e.properties AS properties, e.confidence AS confidence",
            entity_id=entity_id,
        )
        return rows[0] if rows else None

    async def list_segment_nodes(self, kb_id: Optional[str] = None,
                                 doc_id: Optional[str] = None,
                                 limit: Optional[int] = None,
                                 offset: int = 0) -> list[dict]:
        """列出 :Segment 父块节点(Neo4j 侧,区别于 PG parent_document)。"""
        cypher = "MATCH (s:Segment)"
        clauses: list[str] = []
        params: dict = {}
        if kb_id:
            clauses.append("s.kb_id = $kb_id")
            params["kb_id"] = str(kb_id)
        if doc_id:
            clauses.append("s.doc_id = $doc_id")
            params["doc_id"] = doc_id
        if clauses:
            cypher += " WHERE " + " AND ".join(clauses)
        cypher += (" RETURN s.parent_id AS parent_id, s.doc_id AS doc_id, "
                   "s.kb_id AS kb_id, s.title AS title, s.enabled AS enabled, "
                   "s.chapter_no AS chapter_no ORDER BY s.parent_id")
        if limit is not None:
            cypher += f" SKIP {int(offset)} LIMIT {int(limit)}"
        return await self.run(cypher, **params)

    async def get_segment_node(self, parent_id: str) -> Optional[dict]:
        rows = await self.run(
            "MATCH (s:Segment {parent_id: $parent_id}) "
            "RETURN s.parent_id AS parent_id, s.doc_id AS doc_id, "
            "s.kb_id AS kb_id, s.title AS title, s.enabled AS enabled, "
            "s.chapter_no AS chapter_no",
            parent_id=parent_id,
        )
        return rows[0] if rows else None

    async def list_relations_full(self, rel_type: Optional[str] = None,
                                  kb_id: Optional[str] = None,
                                  head_entity_id: Optional[str] = None,
                                  tail_entity_id: Optional[str] = None,
                                  limit: Optional[int] = None,
                                  offset: int = 0) -> list[dict]:
        """列出关系边(非 BINDS)全字段。kb_id 按 source_parent_id 归属该库 Segment 过滤。"""
        clauses: list[str] = []
        params: dict = {}
        if rel_type:
            clauses.append("type(r) = $rel_type")
            params["rel_type"] = rel_type
        if head_entity_id:
            clauses.append("a.entity_id = $head")
            params["head"] = head_entity_id
        if tail_entity_id:
            clauses.append("b.entity_id = $tail")
            params["tail"] = tail_entity_id
        if kb_id:
            seg_rows = await self.run(
                "MATCH (s:Segment {kb_id: $kb_id}) RETURN s.parent_id AS parent_id",
                kb_id=str(kb_id))
            parent_ids = [r["parent_id"] for r in seg_rows]
            if not parent_ids:
                return []
            clauses.append("r.source_parent_id IN $parent_ids")
            params["parent_ids"] = parent_ids
        cypher = "MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) <> 'BINDS'"
        if clauses:
            cypher += " AND " + " AND ".join(clauses)
        cypher += (" RETURN a.entity_id AS head_entity_id, b.entity_id AS tail_entity_id, "
                   "type(r) AS rel_type, r.confidence AS confidence, r.evidence AS evidence, "
                   "r.source_parent_id AS source_parent_id, r.derived AS derived")
        if limit is not None:
            cypher += f" SKIP {int(offset)} LIMIT {int(limit)}"
        return await self.run(cypher, **params)

    async def list_binds_full(self, kb_id: Optional[str] = None,
                              entity_id: Optional[str] = None,
                              parent_id: Optional[str] = None,
                              limit: Optional[int] = None,
                              offset: int = 0) -> list[dict]:
        """列出 BINDS 边(含 kg_role/kg_confidence),可按库/实体/父块过滤。"""
        clauses: list[str] = []
        params: dict = {}
        if kb_id:
            clauses.append("s.kb_id = $kb_id")
            params["kb_id"] = str(kb_id)
        if entity_id:
            clauses.append("e.entity_id = $entity_id")
            params["entity_id"] = entity_id
        if parent_id:
            clauses.append("s.parent_id = $parent_id")
            params["parent_id"] = parent_id
        cypher = "MATCH (e:Entity)-[b:BINDS]->(s:Segment)"
        if clauses:
            cypher += " WHERE " + " AND ".join(clauses)
        cypher += (" RETURN e.entity_id AS entity_id, s.parent_id AS parent_id, "
                   "b.kg_role AS kg_role, b.kg_confidence AS kg_confidence")
        if limit is not None:
            cypher += f" SKIP {int(offset)} LIMIT {int(limit)}"
        return await self.run(cypher, **params)

    async def update_entity(self, entity_id: str, fields: dict) -> None:
        """部分更新实体节点(仅更新传入字段;properties 落库前序列化为 JSON 字符串)。"""
        allowed = {"name", "entity_type", "aliases", "properties", "confidence"}
        setters: list[str] = []
        params: dict = {"entity_id": entity_id}
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "properties" and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            setters.append(f"e.{k} = ${k}")
            params[k] = v
        if not setters:
            return
        await self.run(
            f"MATCH (e:Entity {{entity_id: $entity_id}}) SET {', '.join(setters)}",
            **params)

    async def update_segment(self, parent_id: str, fields: dict) -> None:
        """部分更新 :Segment 节点(仅更新传入字段)。"""
        allowed = {"doc_id", "kb_id", "title", "enabled", "chapter_no"}
        setters: list[str] = []
        params: dict = {"parent_id": parent_id}
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "kb_id" and v is not None:
                v = str(v)
            setters.append(f"s.{k} = ${k}")
            params[k] = v
        if not setters:
            return
        await self.run(
            f"MATCH (s:Segment {{parent_id: $parent_id}}) SET {', '.join(setters)}",
            **params)

    async def delete_relation(self, head_entity_id: str, tail_entity_id: str,
                              rel_type: str) -> None:
        """删除指定 (head, tail, rel_type) 的关系边。rel_type 须是合法 Cypher 关系标识。"""
        await self.run(
            f"MATCH (a:Entity {{entity_id: $head}})-[r:{rel_type}]->"
            f"(b:Entity {{entity_id: $tail}}) DELETE r",
            head=head_entity_id, tail=tail_entity_id)

    async def entity_relations(self, entity_id: str) -> list[dict]:
        """实体的一跳关系边(含方向/类型/对端实体名+类型),BINDS 除外。

        供关系型查询「看图答」:把命中的实体在图中直接相连的边作为事实喂给 LLM,
        即便对端实体是孤儿(无 BINDS 内容)也能给出「X 有哪些 Y」的结构化答案。
        """
        out = await self.run(
            "MATCH (a:Entity {entity_id: $id})-[r]->(b:Entity) WHERE type(r) <> 'BINDS' "
            "RETURN 'out' AS dir, type(r) AS rel_type, b.entity_id AS target_id, "
            "b.name AS target_name, b.entity_type AS target_type",
            id=entity_id)
        inn = await self.run(
            "MATCH (b:Entity)-[r]->(a:Entity {entity_id: $id}) WHERE type(r) <> 'BINDS' "
            "RETURN 'in' AS dir, type(r) AS rel_type, b.entity_id AS target_id, "
            "b.name AS target_name, b.entity_type AS target_type",
            id=entity_id)
        return out + inn

    # ---- 图扩展 ----
    async def expand(self, entity_ids: list, depth: int) -> list[dict]:
        if not entity_ids:
            return []
        # Neo4j 5.26 可变长度边界不接受参数,须是字面量;depth 为受控整数(配置项)
        depth = int(depth)
        return await self.run(
            f"MATCH p=(a:Entity)-[*1..{depth}]-(b:Entity) "
            "WHERE a.entity_id IN $ids "
            "RETURN b.entity_id AS entity_id, b.name AS name, "
            "b.entity_type AS entity_type, min(length(p)) AS hops",
            ids=list(entity_ids),
        )


def get_neo4j() -> Neo4jClient:
    return Neo4jClient()
