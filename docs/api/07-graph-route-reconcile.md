# 路由与对账 `/graph`

## `POST /graph/route` — 分流判定

按规则判定实时/离线链路,**非破坏**(不改变 `/parse*` 行为,offline 仅提示)。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | `[{size:int}]` | 二选一 | 文件对象列表(字节) |
| `sizes` | `int[]` | 二选一 | 文件大小列表(字节) |

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/route \
  -H "Content-Type: application/json" \
  -d '{"sizes": [12582912]}'
```

**传出示例**:

```json
{"lane": "offline", "rule": "R2_FILE_SIZE", "tip": "该文件体量较大，已加入后台全量索引构建任务，预计处理耗时较长，完成后将推送通知。"}
```

| 字段 | 说明 |
|---|---|
| `lane` | `realtime` / `offline` |
| `rule` | `R1`(实时)/ `R2_FILE_SIZE`(单文件超阈值)/ `R3_BATCH_COUNT`(文件数超阈值) |

## `POST /graph/reconcile` — 三方对账

无 body。比对 PG `parent_document` ↔ Neo4j `:Segment`,统计 BINDS 总数。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/reconcile
```

**传出示例**:

```json
{
  "consistent": true,
  "pg_parents": 3,
  "graph_segments": 3,
  "missing_in_graph": [],
  "orphan_in_graph": [],
  "graph_binds": 13
}
```

- `missing_in_graph` = PG 有图无;`orphan_in_graph` = 图有 PG 无。
