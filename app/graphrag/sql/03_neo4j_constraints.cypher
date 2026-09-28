// 电磁信号检索分析系统 —— Neo4j 约束(自 ems sql/03_neo4j_constraints.cypher 迁移)
// MERGE 幂等前提:唯一性约束。

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT segment_parent_id_unique IF NOT EXISTS
FOR (s:Segment) REQUIRE s.parent_id IS UNIQUE;

CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity) ON (e.name);
