# 别名库 `/graph/alias`

别名库 = 人工维护的「代号/别名 → 类别字典实体类型」映射（一对多：一个实体类型可对应多个代号）。实体抽取时作为**动态提示词**注入 LLM，让模型认出文中出现的代号属于哪个类别（如 `a1 → 辐射源`），避免漏抽或判错类型。

- 独立于 `entity_registry.aliases`（那是 LLM 自动抽取产出的实体别名，本表是人工规则）。
- 初始为空，提供 **添加 / 列表 / 单个 / 更新 / 删除** 完整 CRUD。
- `entity_type` 必须是类别字典（`extraction_dict`）里当前启用的实体类型之一。

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `alias` | string | 代号/别名（如 `a1`），全局唯一 |
| `entity_type` | string | 所属类别字典实体类型名（如 `辐射源`） |

## `POST /graph/alias` — 添加

**URL**：`POST http://192.168.31.15:8001/graph/alias`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `alias` | string | 是 | 代号/别名 |
| `entity_type` | string | 是 | 类别字典实体类型名（须是当前启用的实体类型） |

**请求示例**：

```json
{"alias": "a1", "entity_type": "辐射源"}
```

**响应示例**（201）：

```json
{"id": "3a1f...", "alias": "a1", "entity_type": "辐射源",
 "created_at": "...", "updated_at": "..."}
```

错误：

- `409` — `alias` 已存在（返回已存在条目对应的类型）。
- `422` — `entity_type` 不是当前启用的实体类型。

## `GET /graph/alias` — 列表

**URL**：`GET http://192.168.31.15:8001/graph/alias`

**请求数据**（query 参数，可选）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `entity_type` | string | 否 | 按类别过滤（如 `辐射源`） |

**请求示例**：`GET http://192.168.31.15:8001/graph/alias?entity_type=辐射源`

**响应示例**：

```json
[
  {"id": "3a1f...", "alias": "a1", "entity_type": "辐射源", "created_at": "...", "updated_at": "..."}
]
```

## `GET /graph/alias/{alias_id}` — 单个

**URL**：`GET http://192.168.31.15:8001/graph/alias/{alias_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `alias_id` | UUID | 是 | 别名条目 id |

**请求示例**：`GET http://192.168.31.15:8001/graph/alias/3a1f...`

**响应示例**：单个对象；不存在 → 404。

## `PUT /graph/alias/{alias_id}` — 更新

**URL**：`PUT http://192.168.31.15:8001/graph/alias/{alias_id}`

**请求数据**（`application/json`，仅传要改的字段）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `alias` | string | 否 | 新代号 |
| `entity_type` | string | 否 | 新类别字典实体类型名 |

**请求示例 — 改类型**：

```json
{"entity_type": "平台"}
```

**响应示例**：更新后的对象。

错误：`404` 条目不存在、`409` 新 `alias` 已被占用、`422` `entity_type` 不是当前启用的实体类型。

## `DELETE /graph/alias/{alias_id}` — 删除

**URL**：`DELETE http://192.168.31.15:8001/graph/alias/{alias_id}`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `alias_id` | UUID | 是 | 别名条目 id |

**请求示例**：`DELETE http://192.168.31.15:8001/graph/alias/3a1f...`

**响应示例**：204 无 body。

错误：`404` 条目不存在。

> 删除为**物理删除**（与 `/graph/dict` 的逻辑删除不同），删后可重新添加同名别名。
