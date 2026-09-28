# 抽取字典 `/graph/dict`

字典 = 抽取「哪些实体/关系」的**唯一数据源**,全局共享(不分知识库),抽取侧每次现查,改后即时生效。

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 名称(实体/关系中文名) |
| `en` | string? | 英文名(关系用,如 `BELONGS_TO`) |
| `category` | string | **`entity`(节点)/ `relation`(关系)** |
| `description` | string | 描述 |
| `json_schema` | object? | LLM function-calling JSON Schema |
| `trigger_words` | string[]? | 关系触发词 |
| `min_confidence` | number? | 关系最低置信度(默认 0.75) |
| `head_types` / `tail_types` | string[]? | 关系两端允许的实体类型(domain/range) |
| `enabled` | bool | 是否启用 |
| `semantic_mergeable` | bool | 是否允许向量语义层归并(参数值类=false) |

## `POST /graph/dict` — 新增实体类型

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/dict \
  -H "Content-Type: application/json" \
  -d '{"name": "舰艇", "category": "entity", "description": "水面/水下作战平台"}'
```

**传出示例**(201):

```json
{"id": "3a1f...", "name": "舰艇", "en": null, "category": "entity", "description": "水面/水下作战平台",
 "json_schema": null, "trigger_words": null, "min_confidence": null, "head_types": null, "tail_types": null,
 "enabled": true, "semantic_mergeable": false, "created_at": "...", "updated_at": "..."}
```

## `POST /graph/dict` — 新增关系

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/dict \
  -H "Content-Type: application/json" \
  -d '{"name": "指挥", "en": "COMMANDS", "category": "relation",
       "trigger_words": ["指挥", "统一指挥", "受令"], "min_confidence": 0.8}'
```

**传出示例**(201):同上结构,`category="relation"`。

## `GET /graph/dict` — 列表

**传入示例**:

```bash
curl "http://127.0.0.1:8001/graph/dict?category=entity"
```

**传出示例**:

```json
[
  {"id": "3a1f...", "name": "舰艇", "category": "entity", "enabled": true, "...": "..."}
]
```

## `GET /graph/dict/{dict_id}` — 单个

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/dict/3a1f...
```

**传出示例**:单个对象;不存在 → 404。

## `PUT /graph/dict/{dict_id}` — 更新

**传入示例**(仅传要改字段):

```bash
curl -X PUT http://127.0.0.1:8001/graph/dict/3a1f... \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

**传出示例**:更新后的对象。

## `DELETE /graph/dict/{dict_id}` — 逻辑删除

**传入示例**:

```bash
curl -X DELETE http://127.0.0.1:8001/graph/dict/3a1f...
```

**传出示例**:204 无 body。

错误:`404` 条目不存在、`422` `category` 不是 `entity|relation`。
