# 知识库管理 `/knowledge-bases`

## `POST /knowledge-bases` — 新建

**URL**：`POST http://192.168.31.15:8001/knowledge-bases`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 知识库名称（唯一，重复 → 409） |
| `description` | string | 否 | 描述，默认 `""` |

**请求示例**：

```json
{"name": "雷达情报库", "description": "雷达信号与情报文档"}
```

**响应示例**（201）：

```json
{"id": "2db770f2-7a09-4f04-9f07-fce5e6bf6529", "name": "雷达情报库", "description": "雷达信号与情报文档",
 "created_at": "2026-09-28T07:17:47.025473Z", "updated_at": "2026-09-28T07:17:47.025473Z"}
```

## `GET /knowledge-bases` — 列表（支持分页）

**URL**：`GET http://192.168.31.15:8001/knowledge-bases`

**请求数据**（query 参数，可选）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `page` | int | 否 | 页码，从 1 开始，默认 `1` |
| `page_size` | int | 否 | 每页条数（1~100）；不传则返回全量列表 |

**请求示例 — 分页**：`GET http://192.168.31.15:8001/knowledge-bases?page=1&page_size=10`

**响应示例 — 分页**：

```json
{
  "total": 25,
  "page": 1,
  "page_size": 10,
  "items": [
    {"id": "2db770f2-7a09-4f04-9f07-fce5e6bf6529", "name": "雷达情报库", "description": "雷达信号与情报文档",
     "created_at": "2026-09-28T07:17:47.025473Z", "updated_at": "2026-09-28T07:17:47.025473Z"}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `total` | 未删除知识库总数 |
| `page` | 当前页码 |
| `page_size` | 每页条数 |
| `items` | 当前页知识库列表 |

**请求示例 — 全量（不传分页参数）**：`GET http://192.168.31.15:8001/knowledge-bases`

**响应示例 — 全量**（保持旧格式）：

```json
[
  {"id": "2db770f2-7a09-4f04-9f07-fce5e6bf6529", "name": "雷达情报库", "description": "雷达信号与情报文档",
   "created_at": "2026-09-28T07:17:47.025473Z", "updated_at": "2026-09-28T07:17:47.025473Z"}
]
```

## `GET /knowledge-bases/{kb_id}` — 单个

**URL**：`GET http://192.168.31.15:8001/knowledge-bases/{kb_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 是 | 知识库 id |

**请求示例**：`GET http://192.168.31.15:8001/knowledge-bases/2db770f2-7a09-4f04-9f07-fce5e6bf6529`

**响应示例**：同单个对象；不存在 → 404。

## `PUT /knowledge-bases/{kb_id}` — 更新

**URL**：`PUT http://192.168.31.15:8001/knowledge-bases/{kb_id}`

**请求数据**（`application/json`，仅传要改字段）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 否 | 名称 |
| `description` | string | 否 | 描述 |

**请求示例**：

```json
{"description": "更新后的描述"}
```

**响应示例**：更新后的 `KnowledgeBaseRead`。

## `DELETE /knowledge-bases/{kb_id}` — 逻辑删除

**URL**：`DELETE http://192.168.31.15:8001/knowledge-bases/{kb_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 是 | 知识库 id |

**请求示例**：`DELETE http://192.168.31.15:8001/knowledge-bases/2db770f2-7a09-4f04-9f07-fce5e6bf6529`

**响应示例**：204 无 body（置 `deleted_at`，不物理删除）。
