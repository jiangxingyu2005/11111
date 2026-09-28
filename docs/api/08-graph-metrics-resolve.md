# 图质量与消歧 `/graph`

## `GET /graph/metrics` — 图质量四指标

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/metrics
```

**传出示例**:

```json
{
  "orphan_rate": 0.125,
  "binds_coverage": 1.0,
  "max_connected_component": 6,
  "avg_degree": 1.333,
  "totals": {"entities": 8, "relations": 7, "parents": 4},
  "orphan_entities": [{"entity_id": "<entity_id>", "name": "某装备"}]
}
```

## `GET /graph/quality` — 漏抽质检门禁

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/quality
```

**传出示例**:

```json
{
  "metrics": {"orphan_rate": 0.0, "binds_coverage": 1.0, "max_connected_component": 3, "avg_degree": 1.0},
  "totals": {"entities": 8, "relations": 5, "parents": 4, "documents_checked": 2, "empty_documents": 0},
  "healthy": true,
  "gates": {
    "orphan_rate": {"value": 0.0, "max": 0.5},
    "binds_coverage": {"value": 1.0, "min": 0.3},
    "empty_ratio": {"value": 0.0, "max": 0.5}
  },
  "leak_documents": []
}
```

- `healthy` = 孤立率 ≤ `_MAX_ORPHAN_RATE` 且 BINDS 覆盖率 ≥ `_MIN_BINDS_COVERAGE` 且空图谱占比 ≤ `_MAX_EMPTY_RATIO`。

## `POST /graph/mesh/link-orphans` — 孤儿兜底绑定

无 body,幂等。对 `kg_entity_id` 为空的父块按别名/指纹兜底绑定。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/mesh/link-orphans
```

**传出示例**:

```json
{
  "linked": [{"parent_id": "8f14e45f-..._2", "entity_id": "<entity_id>"}],
  "reextract": ["8f14e45f-..._3"]
}
```

## `POST /graph/resolve` — 全量消歧归并

历史重复实体批量归并(四层回填:PG registry / Neo4j / PG parent / Milvus)。

**传入示例**:

```bash
# 预演(只返回归并计划,不落库)
curl -X POST "http://127.0.0.1:8001/graph/resolve?dry_run=true"
# 落库执行
curl -X POST http://127.0.0.1:8001/graph/resolve
```

**传出示例**:

```json
{
  "dry_run": false,
  "merged": [{"old_id": "<old>", "new_id": "<canonical>", "name": "雷达站"}],
  "binds_moved": 1,
  "relations_moved": 1,
  "parents_updated": 1,
  "chunks_updated": 1,
  "entities_before": 2,
  "entities_after": 1
}
```

- `dry_run=true` 只返回归并计划,不落库;幂等(重复调用 `merged=[]`、计数 0)。
- 错误:`502` Neo4j/Milvus 不可用。
