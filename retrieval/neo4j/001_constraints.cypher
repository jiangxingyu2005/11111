// Neo4j 约束与索引（001_constraints.cypher）
// 执行：docker compose exec neo4j cypher-shell -u neo4j -p dev_only -f /import/001_constraints.cypher
//
// 数据模型（与 retrieval/app/recall/graph_route.py 的 Cypher 严格对应）：
//   (:Entity {entity_id, name, type, kb_id})
//   (:Segment {chunk_id, parent_id, content, kb_id, span_start, span_end})
//   (:Entity)-[:MOUNTED_ON|USES_MODULATION|OPERATES_IN|HAS_FUNCTION|BELONGS_TO|ACTIVE_IN]->(:Entity)
//   (:Entity)<-[:BINDS]-(:Segment)
//
// 纪律：不装 GDS 插件（V1.1 §0.1：社区检测用 igraph+leidenalg 应用侧计算）

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT segment_id IF NOT EXISTS
FOR (s:Segment) REQUIRE s.chunk_id IS UNIQUE;

CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX entity_kb IF NOT EXISTS
FOR (e:Entity) ON (e.kb_id);

CREATE INDEX segment_kb IF NOT EXISTS
FOR (s:Segment) ON (s.kb_id);
