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
                   "e.entity_type AS entity_type, e.aliases AS aliases, "
                   "e.properties AS properties")
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

    @staticmethod
    def _edge_in_kb(r: str, x: str, y: str) -> str:
        """关系边是否属于 $kb_id:有来源分段的看来源分段所属库;
        无来源的(派生边/手工边)要求两端实体都在该库的分段里出现过。"""
        return (f"CASE WHEN {r}.source_parent_id IS NOT NULL "
                f"THEN EXISTS {{ MATCH (src:Segment {{parent_id: {r}.source_parent_id}}) WHERE src.kb_id = $kb_id }} "
                f"ELSE EXISTS {{ MATCH ({x})-[:BINDS]->(s1:Segment) WHERE s1.kb_id = $kb_id }} "
                f"AND EXISTS {{ MATCH ({y})-[:BINDS]->(s2:Segment) WHERE s2.kb_id = $kb_id }} END")

    async def entity_relations(self, entity_id: str, limit: Optional[int] = None,
                               kb_id=None) -> list[dict]:
        """实体的一跳关系边(含方向/类型/对端实体名+类型),BINDS 除外。

        供关系型查询「看图答」:把命中的实体在图中直接相连的边作为事实喂给 LLM,
        即便对端实体是孤儿(无 BINDS 内容)也能给出「X 有哪些 Y」的结构化答案。
        """
        # limit:枢纽实体(如某国别组织)可能连成百上千条边,不设上限会把提示词撑爆。
        # 抽取边优先于派生边(derived 为 false 排前),同类按置信度降序,两个方向各取 limit 再合并截断。
        tail = (" RETURN '{dir}' AS dir, type(r) AS rel_type, b.entity_id AS target_id, "
                "b.name AS target_name, b.entity_type AS target_type, "
                "coalesce(r.derived, false) AS derived, coalesce(r.confidence, 0.0) AS confidence "
                "ORDER BY derived, confidence DESC")
        if limit is not None:
            tail += " LIMIT $limit"
        # kb_id:实体全局共享,同一实体在别的库里的关系边不能作为本库的事实
        kb = str(kb_id) if kb_id else None
        scope = f" AND {self._edge_in_kb('r', 'a', 'b')}" if kb else ""
        out = await self.run(
            "MATCH (a:Entity {entity_id: $id})-[r]->(b:Entity) WHERE type(r) <> 'BINDS'" + scope
            + tail.format(dir="out"), id=entity_id, limit=limit, kb_id=kb)
        inn = await self.run(
            "MATCH (b:Entity)-[r]->(a:Entity {entity_id: $id}) WHERE type(r) <> 'BINDS'" + scope
            + tail.format(dir="in"), id=entity_id, limit=limit, kb_id=kb)
        rows = sorted(out + inn, key=lambda r: (bool(r.get("derived")), -float(r.get("confidence") or 0.0)))
        return rows[:limit] if limit is not None else rows

    # ---- 图扩展 ----
    _EXPAND_FRONTIER_CAP = 200   # 每跳最多带多少个节点进下一跳(防枢纽实体/分段把边界撑爆)
    _COOCCUR_MAX_HOPS = 2        # 同段共现只用于前 2 跳;3 跳起只走抽取关系边

    async def _rel_step(self, frontier: list, visited: list, kb) -> list[dict]:
        """一跳关系边扩展(实体-实体,不含 BINDS、不含派生边)。

        每个新到达节点只留一条来路(置信度最高),供回溯关系链;按置信度取前 cap 个。
        """
        scope = f" AND {self._edge_in_kb('r', 'x', 'y')}" if kb else ""
        return await self.run(
            "MATCH (x:Entity)-[r]-(y:Entity) WHERE elementId(x) IN $frontier "
            "AND type(r) <> 'BINDS' AND NOT coalesce(r.derived, false) "
            "AND NOT elementId(y) IN $visited" + scope + " "
            "WITH y, x, r, coalesce(r.confidence, 0.0) AS conf ORDER BY conf DESC "
            "WITH y, collect({prev: elementId(x), rel: type(r), "
            "dir: CASE WHEN startNode(r) = x THEN 'out' ELSE 'in' END, conf: conf})[0] AS via "
            "RETURN elementId(y) AS nid, y.entity_id AS entity_id, y.name AS name, "
            "y.entity_type AS entity_type, via ORDER BY via.conf DESC, nid LIMIT $cap",
            frontier=frontier, visited=visited, cap=self._EXPAND_FRONTIER_CAP, kb_id=kb)

    async def _cooccur_step(self, frontier: list, visited: list, kb) -> list[dict]:
        """一跳同段共现扩展:与边界实体出现在同一(本库)分段里的实体,按共享分段数排序。"""
        scope = " AND s.kb_id = $kb_id" if kb else ""
        return await self.run(
            "MATCH (x:Entity)-[:BINDS]->(s:Segment)<-[:BINDS]-(y:Entity) "
            "WHERE elementId(x) IN $frontier AND NOT elementId(y) IN $visited" + scope + " "
            "WITH y, count(DISTINCT s) AS shared "
            "RETURN elementId(y) AS nid, y.entity_id AS entity_id, y.name AS name, "
            "y.entity_type AS entity_type, shared ORDER BY shared DESC, nid LIMIT $cap",
            frontier=frontier, visited=visited, cap=self._EXPAND_FRONTIER_CAP, kb_id=kb)

    async def _bfs(self, seeds: list, depth: int, kb, cooccur: bool) -> dict:
        """逐跳 BFS,返回 {nid: {entity_id, name, entity_type, hops, via, shared}}。

        cooccur=False:只走关系边(via 记录来路,可回溯成关系链);
        cooccur=True :每跳 关系边 + 同段共现 合并(同一节点两种都到达时以关系边为准)。
        """
        found: dict = {}
        frontier = list(seeds)
        visited = set(seeds)
        for hop in range(1, depth + 1):
            if not frontier:
                break
            rows = {r["nid"]: dict(r, shared=0) for r in await self._rel_step(frontier, list(visited), kb)}
            if cooccur:
                for r in await self._cooccur_step(frontier, list(visited), kb):
                    if r["nid"] not in rows:
                        rows[r["nid"]] = dict(r, via=None)
                    else:
                        rows[r["nid"]]["shared"] = r["shared"]
            ranked = sorted(rows.values(), key=lambda r: (r.get("via") is None,
                                                          -float((r.get("via") or {}).get("conf") or 0.0),
                                                          -int(r.get("shared") or 0), r["nid"]))
            ranked = ranked[:self._EXPAND_FRONTIER_CAP]
            for r in ranked:
                found[r["nid"]] = {"entity_id": r["entity_id"], "name": r["name"],
                                   "entity_type": r["entity_type"], "hops": hop,
                                   "via": r.get("via"), "shared": int(r.get("shared") or 0)}
            frontier = [r["nid"] for r in ranked]
            visited.update(frontier)
        return found

    @staticmethod
    def _rel_path(nid: str, rel: dict, seed_names: dict) -> list[dict]:
        """沿关系 BFS 的来路回溯到种子,返回 [{from, rel_type, dir, to}, ...](种子 → 该实体)。"""
        steps = []
        cur = nid
        while cur in rel:
            via = rel[cur]["via"]
            prev = via["prev"]
            prev_name = rel[prev]["name"] if prev in rel else seed_names.get(prev, prev)
            steps.append({"from": prev_name, "rel_type": via["rel"], "dir": via["dir"],
                          "to": rel[cur]["name"]})
            cur = prev
        return list(reversed(steps))

    async def expand(self, entity_ids: list, depth: int, limit: Optional[int] = None,
                     kb_id=None, hop_quotas: Optional[list] = None) -> list[dict]:
        """从种子实体做 1..depth 跳扩展,返回关联实体(不含种子本身)。

        跳数按实体计:A 与 B 之间有一条关系边,或出现在同一分段(同段共现),都算 1 跳。
        两遍 BFS(逐跳展开去重节点,代价随节点数增长,不随路径数指数增长):
          - 关系遍:只走抽取关系边,最多 depth 跳,记录来路,可回溯成关系链;
          - 共现遍:关系边 + 同段共现,最多 _COOCCUR_MAX_HOPS(2)跳。
        合并取最短跳数,所以第 3、4 跳的实体一定是经完整关系链到达的。
        hop_quotas:每跳名额(如 [12, 10, 4, 4]),防止近处实体占满 limit;某跳不足的名额
        让给其余候选。不传 = 近的优先。经关系链(≥2 跳)到达的实体附带 path,供生成多跳事实。
        派生边(同频段/同时段/同区域)不参与遍历。kb_id:只走本库分段与本库关系边;不传 = 全库。
        """
        if not entity_ids:
            return []
        kb = str(kb_id) if kb_id else None
        depth = int(depth)
        seed_rows = await self.run(
            "MATCH (a:Entity) WHERE a.entity_id IN $ids RETURN elementId(a) AS nid, a.name AS name",
            ids=list(entity_ids))
        seeds = [r["nid"] for r in seed_rows]
        seed_names = {r["nid"]: r["name"] for r in seed_rows}

        rel = await self._bfs(seeds, depth, kb, cooccur=False)
        near = await self._bfs(seeds, min(depth, self._COOCCUR_MAX_HOPS), kb, cooccur=True)

        items: list[dict] = []
        for nid in set(rel) | set(near):
            base = rel.get(nid) or near[nid]
            hops = min(x["hops"] for x in (rel.get(nid), near.get(nid)) if x)
            item = {"entity_id": base["entity_id"], "name": base["name"],
                    "entity_type": base["entity_type"], "hops": hops}
            if nid in rel and rel[nid]["hops"] >= 2:
                item["path"] = self._rel_path(nid, rel, seed_names)
            item["_rank"] = (0 if nid in rel else 1,
                             rel[nid]["hops"] if nid in rel else 99,
                             -(near[nid]["shared"] if nid in near else 0), base["entity_id"])
            items.append(item)

        if limit is None:
            chosen = items
        elif hop_quotas:
            quotas = [int(q) for q in hop_quotas[:depth]]
            total = sum(quotas)
            if total > limit:   # 名额之和超过上限:按比例压缩,每跳至少 1 个
                quotas = [max(1, q * limit // total) for q in quotas]
            by_hop: dict = {}
            for it in items:
                by_hop.setdefault(it["hops"], []).append(it)
            chosen, rest = [], []
            for h, group in by_hop.items():
                group.sort(key=lambda it: it["_rank"])
                q = quotas[h - 1] if h - 1 < len(quotas) else 0
                chosen += group[:q]
                rest += group[q:]
            rest.sort(key=lambda it: (it["hops"], it["_rank"]))
            chosen += rest[:max(0, limit - len(chosen))]
            chosen = chosen[:limit]
        else:
            chosen = sorted(items, key=lambda it: (it["hops"], it["_rank"]))[:limit]

        for it in chosen:
            it.pop("_rank", None)
        chosen.sort(key=lambda e: (e["hops"], e["entity_id"]))
        return chosen


def get_neo4j() -> Neo4jClient:
    return Neo4jClient()
