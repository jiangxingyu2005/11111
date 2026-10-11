# 在线查询与溯源 `/graph`

## `POST /graph/query` — 自然语言查询

意图分流 → 实体匹配 → A/B 双路召回 → 重排 →（命中关系触发词或关系型问句）图扩展 → LLM 生成带引用答案。

> 意图分流除字典关系触发词外，还内置关系型问句启发式（疑问词 + 关系动词，如「A国装备了哪些武器」），这类问法答案在图的关系边上，会走图扩展分支并把命中实体的一跳关系事实喂给 LLM（看图答）。

**URL**：`POST http://192.168.31.15:8001/graph/query`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 自然语言问题 |
| `kb_id` | UUID | 否 | 单个知识库 id（兼容旧前端） |
| `kb_ids` | UUID[] | 否 | 勾选的知识库 id 列表；与 `kb_id` 合并去重。都不传 = 全库 |

**请求示例**：

```json
{"query": "2.4GHz 的调制方式是什么", "kb_id": "e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5"}
```

多库勾选：

```json
{
  "query": "2.4GHz 有哪些装备",
  "kb_ids": [
    "e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5",
    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
  ]
}
```

知识库列表：`GET /knowledge-bases`（每条有 `id`）。只在勾选的库内做实体匹配、子块召回、图扩展，不串未勾选的库。

**响应示例**：

```json
{
  "intent": "retrieval",
  "cond": {"intent": "local_fact", "modulation_type": null, "freq_band": "2.4GHz", "raw": "2.4GHz 的调制方式是什么"},
  "answer": "根据检索结果（3 条）：\n[1] 某舰艇装备一部对空搜索雷达，工作频率 2.4GHz…",
  "hits": [
    {"child_id": "{parent_id}#0", "doc_id": "8f14e45f-...", "content": "…", "_score": 0.81, "kg_entity_ids": ["<entity_id>"]}
  ],
  "matched_entities": ["<entity_id>"],
  "expanded_entities": [],
  "graph_facts": ["AN/SPY-1D --[BELONGS_TO]--> A国"]
}
```

| 字段 | 说明 |
|---|---|
| `intent` | `retrieval`（纯检索）/ `graph`（图扩展） |
| `cond` | NL2Query 结构化条件 |
| `answer` | 带 `[n]` 引用答案 |
| `hits` | 召回子块（`_score` = COSINE 相似度 + 实体加权） |
| `matched_entities` / `expanded_entities` | 命中实体 / 扩展实体 |
| `graph_facts` | 关系型问句命中时的一跳关系事实（`实体 --[关系]--> 实体`），供 LLM 看图答 |

错误：`422` 缺 `query`；`502` Neo4j/Milvus 不可用。

## `POST /graph/query/stream` — 自然语言查询（SSE 流式）

与 `/graph/query` 同一条检索链路，但以 `text/event-stream` 逐步骤下发，供前端实时展示「双路检索每一步」与流式答案。

**URL**：`POST http://192.168.31.15:8001/graph/query/stream`

**请求数据**：同 `/graph/query`（`query` / `kb_id`）。

**SSE 事件序列**：`meta` → `progress`* → `results` → `token`* → `done`（异常时 `error`）。

| 事件 | data 关键字段 | 说明 |
|---|---|---|
| `meta` | `query` `intent` `cond` | 意图分流 + NL2Query 结果 |
| `progress`（entity_match） | `matched_entities` `matched_ids` | 查询词匹配到的实体 |
| `progress`（recall） | `route_a` `route_b` | A 路（实体锚定）/ B 路（纯向量）召回命中数 |
| `progress`（rerank） | `top` | 重排后保留条数 |
| `progress`（graph_expand） | `expanded` `facts` | 图扩展实体数 + 一跳关系事实（仅关系型） |
| `results` | `hits` `graph_facts` `expanded_entities` | 最终召回与图事实 |
| `token` | `text` | 流式答案增量（逐 token） |
| `done` / `error` | — / `message` | 正常结束 / 失败 |

**示例**（`curl -N`）：

```bash
curl -N -X POST http://192.168.31.15:8001/graph/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "A国装备了哪些武器"}'
```

```text
event: meta
data: {"query": "A国装备了哪些武器", "intent": "graph", "cond": {...}}

event: progress
data: {"step": "entity_match", "matched_entities": ["A国"], "matched_ids": ["ent_41916512"]}

event: progress
data: {"step": "recall", "route_a": 0, "route_b": 10}

event: progress
data: {"step": "rerank", "top": 5}

event: progress
data: {"step": "graph_expand", "expanded": 34, "facts": ["AN/SPY-1D --[BELONGS_TO]--> A国", ...]}

event: results
data: {"hits": [...], "graph_facts": [...], "expanded_entities": [...]}

event: token
data: {"text": "A国"}

event: token
data: {"text": "装备的武器..."}

event: done
data: {}
```

## `GET /graph/trace/files` — 文件召回列表

**URL**：`GET http://192.168.31.15:8001/graph/trace/files`

**请求数据**（query 参数，可选）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 否 | 按知识库过滤，不传 = 全库 |

**请求示例**：`GET http://192.168.31.15:8001/graph/trace/files?kb_id=e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5`

**响应示例**：

```json
{
  "files": [
    {"doc_id": "8f14e45f-...", "file_name": "ems_test.html", "doc_type": "html",
     "event_time": null, "uploaded_at": "2026-09-26T12:00:00+00:00", "recall_count": 2}
  ]
}
```

## `GET /graph/trace/files/{doc_id}` — 文件详情 + 召回历史

**URL**：`GET http://192.168.31.15:8001/graph/trace/files/{doc_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | string | 是 | 文档 id |

**请求示例**：`GET http://192.168.31.15:8001/graph/trace/files/8f14e45f-...`

**响应示例**：

```json
{
  "doc_id": "8f14e45f-...", "file_name": "ems_test.html", "found": true, "recall_count": 2,
  "history": [
    {"query": "2.4GHz 的调制方式是什么", "child_id": "{parent_id}#0", "parent_id": "{parent_id}", "ts": "2026-09-26T12:00:01+00:00"}
  ]
}
```

不存在：`{"doc_id": "...", "found": false}`（200，非 404）。

## `GET /graph/trace/files/{doc_id}/locate` — 源文件定位

**URL**：`GET http://192.168.31.15:8001/graph/trace/files/{doc_id}/locate`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | string | 是 | 文档 id |

**请求示例**：`GET http://192.168.31.15:8001/graph/trace/files/8f14e45f-.../locate`

**响应示例**：

```json
{"doc_id": "8f14e45f-...", "found": true, "source_path": "minio://.../8f14e45f-.../ems_test.html"}
```

不存在：`{"doc_id": "...", "found": false, "source_path": null}`。
