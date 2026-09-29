# 抽取字典 `/graph/dict`

字典 = 抽取「哪些实体/关系」的**唯一数据源**，全局共享（不分知识库），抽取侧每次现查，改后即时生效。

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 名称（实体/关系中文名） |
| `en` | string? | 英文名（关系用，如 `BELONGS_TO`） |
| `category` | string | **`entity`（节点）/ `relation`（关系）** |
| `description` | string | 描述 |
| `json_schema` | object? | 该类型抽取时的属性结构（function-calling **硬约束**，见下） |
| `trigger_words` | string[]? | 关系触发词 |
| `min_confidence` | number? | 关系最低置信度（默认 0.75） |
| `head_types` / `tail_types` | string[]? | 关系两端允许的实体类型（domain/range） |
| `enabled` | bool | 是否启用 |
| `semantic_mergeable` | bool | 是否允许向量语义层归并（参数值类=false） |

## json_schema 写法

`json_schema` 是 LLM function-calling 用的 JSON Schema（`application/json`，可选），描述该类实体/关系抽取时的属性结构。**创建和更新都能传**。

抽取侧如何用它（硬约束，改后即时生效）：

- **实体**：每类实体单独一次 function 调用，该类的 `json_schema` 直接作为 function 的 `properties` schema，模型只能按这个结构输出 `properties`。不传 `json_schema` 则 `properties` 退回 `{"type":"object"}`（不约束，但类型仍被该次调用固定）。
- **关系**：`rel_type` 字段加 `enum` 硬约束（= 所有已定义关系的 `en`），模型只能输出字典里定义过的关系类型。

实体类（`entity`）示例：

```json
{"name": "舰艇", "category": "entity", "description": "水面/水下作战平台",
 "json_schema": {"type": "object",
   "properties": {"name": {"type": "string", "description": "舰艇名称"},
                  "ship_class": {"type": "string", "description": "舰级：驱逐舰/护卫舰/潜艇等"}}}}
```

关系类（`relation`）示例：

```json
{"name": "指挥", "en": "COMMANDS", "category": "relation",
 "trigger_words": ["指挥", "统一指挥"],
 "json_schema": {"type": "object",
   "properties": {"head": {"type": "string", "description": "指挥方实体名"},
                  "tail": {"type": "string", "description": "被指挥方实体名"}}}}
```

## `POST /graph/dict` — 新增实体类型

**URL**：`POST http://192.168.31.15:8001/graph/dict`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 实体中文名 |
| `category` | string | 是 | `entity` |
| `description` | string | 否 | 描述，默认 `""` |
| `json_schema` | object | 否 | function-calling 用的 JSON Schema（见上） |
| `semantic_mergeable` | bool | 否 | 是否允许向量语义层归并，默认 `false` |

**请求示例**：

```json
{"name": "舰艇", "category": "entity", "description": "水面/水下作战平台",
 "json_schema": {"type": "object", "properties": {"ship_class": {"type": "string"}}}}
```

**响应示例**（201）：

```json
{"id": "3a1f...", "name": "舰艇", "en": null, "category": "entity", "description": "水面/水下作战平台",
 "json_schema": null, "trigger_words": null, "min_confidence": null, "head_types": null, "tail_types": null,
 "enabled": true, "semantic_mergeable": false, "created_at": "...", "updated_at": "..."}
```

## `POST /graph/dict` — 新增关系

**URL**：`POST http://192.168.31.15:8001/graph/dict`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 关系中文名 |
| `category` | string | 是 | `relation` |
| `en` | string | 否 | 英文名 |
| `trigger_words` | string[] | 否 | 触发词 |
| `min_confidence` | number | 否 | 最低置信度 |
| `json_schema` | object | 否 | function-calling 用的 JSON Schema（见上） |

**请求示例**：

```json
{"name": "指挥", "en": "COMMANDS", "category": "relation",
 "trigger_words": ["指挥", "统一指挥", "受令"], "min_confidence": 0.8,
 "json_schema": {"type": "object", "properties": {"head": {"type": "string"}}}}
```

**响应示例**（201）：同上结构，`category="relation"`。

## `GET /graph/dict` — 列表

**URL**：`GET http://192.168.31.15:8001/graph/dict`

**请求数据**（query 参数，可选）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `category` | string | 否 | 按类别过滤：`entity` / `relation` |

**请求示例**：`GET http://192.168.31.15:8001/graph/dict?category=entity`

**响应示例**：

```json
[
  {"id": "3a1f...", "name": "舰艇", "category": "entity", "enabled": true, "...": "..."}
]
```

## `GET /graph/dict/{dict_id}` — 单个

**URL**：`GET http://192.168.31.15:8001/graph/dict/{dict_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `dict_id` | UUID | 是 | 字典条目 id |

**请求示例**：`GET http://192.168.31.15:8001/graph/dict/3a1f...`

**响应示例**：单个对象；不存在 → 404。

## `PUT /graph/dict/{dict_id}` — 更新

**URL**：`PUT http://192.168.31.15:8001/graph/dict/{dict_id}`

**请求数据**（`application/json`，仅传要改字段，见「字段说明」表）：

**请求示例 — 改 json_schema**：

```json
{"json_schema": {"type": "object", "properties": {"ship_class": {"type": "string", "description": "舰级"}}}}
```

其他字段同理（只传要改的即可，未传字段保持不变）。

**响应示例**：更新后的对象。

## `DELETE /graph/dict/{dict_id}` — 逻辑删除

**URL**：`DELETE http://192.168.31.15:8001/graph/dict/{dict_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `dict_id` | UUID | 是 | 字典条目 id |

**请求示例**：`DELETE http://192.168.31.15:8001/graph/dict/3a1f...`

**响应示例**：204 无 body。

错误：`404` 条目不存在、`422` `category` 不是 `entity|relation`。
