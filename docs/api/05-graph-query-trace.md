# 在线查询与溯源 `/graph`

## `POST /graph/query` — 自然语言查询

意图分流 → 实体匹配 → A/B 双路召回 → 重排 →（命中关系触发词）图扩展 → LLM 生成带引用答案。

**URL**：`POST http://192.168.31.15:8001/graph/query`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 自然语言问题 |
| `kb_id` | UUID | 否 | 知识库 id（不传 = 全库） |

**请求示例**：

```json
{"query": "2.4GHz 的调制方式是什么", "kb_id": "e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5"}
```

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
  "expanded_entities": []
}
```

| 字段 | 说明 |
|---|---|
| `intent` | `retrieval`（纯检索）/ `graph`（图扩展） |
| `cond` | NL2Query 结构化条件 |
| `answer` | 带 `[n]` 引用答案 |
| `hits` | 召回子块（`_score` = COSINE 相似度 + 实体加权） |
| `matched_entities` / `expanded_entities` | 命中实体 / 扩展实体 |

错误：`422` 缺 `query`；`502` Neo4j/Milvus 不可用。

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
