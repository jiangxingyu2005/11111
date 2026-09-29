# Neo4j 图查询/管理 `/graph`

直接对 Neo4j 图库做**查询与增删改**的接口(区别于「在线检索」`/graph/query`,这里不做召回/生成,直接读写图)。

## 图模型与归属约定

图里有两类节点、两类边:

| 元素 | Cypher | 关键字段 | 说明 |
|---|---|---|---|
| 实体节点 | `(:Entity)` | `entity_id` `name` `entity_type` `aliases` `properties` `confidence` | 全局共享、**不分知识库** |
| 父块节点 | `(:Segment)` | `parent_id` `doc_id` `kb_id` `title` `enabled` `chapter_no` | 按知识库隔离 |
| 关系边 | `(:Entity)-[r]->(:Entity)` | `rel_type`(类型) `confidence` `evidence` `source_parent_id` `derived` | 实体↔实体 |
| 绑定边 | `(:Entity)-[:BINDS]->(:Segment)` | `kg_role` `kg_confidence` | 实体↔父块 |

**知识库过滤(kb_id)怎么算**:

- **实体**:实体本身没有 `kb_id`,归属靠它 `BINDS` 到哪些 `:Segment`,再取 `Segment.kb_id`。传 `kb_id` 只返回「绑定到该库父块的实体」。
- **关系边**:按 `source_parent_id`(抽取来源父块)归属;传 `kb_id` 只返回来源父块属于该库的关系边。
- **父块/BINDS**:直接按 `Segment.kb_id` 过滤。

Neo4j 不可用时这些接口返回 `502`,不影响其它接口与 parse 主链路。所有写操作都走 `MERGE`(幂等),重复调用不产生重复节点/边。

---

## 节点总览

### `GET /graph/nodes` — 全部节点概览

**URL**：`GET http://192.168.31.15:8001/graph/nodes`

| query | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 否 | 按知识库过滤;不传 = 全库 |

**响应示例**：

```json
{
  "entities": [
    {"entity_id": "e1", "name": "AN/TPQ-50", "entity_type": "辐射源", "aliases": ["TPQ-50"], "confidence": 0.9}
  ],
  "segments": [
    {"parent_id": "p1", "doc_id": "d1", "kb_id": "…", "title": "…", "enabled": true, "chapter_no": "1.1"}
  ],
  "totals": {"entities": 1, "segments": 1}
}
```

---

## 实体节点 `/graph/entities`

### `GET /graph/entities` — 实体列表

| query | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 否 | 按名称/别名模糊匹配 |
| `entity_type` | string | 否 | 按实体类型过滤 |
| `kb_id` | UUID | 否 | 按知识库过滤(实体归属经 BINDS 判定) |
| `limit` / `offset` | int | 否 | 分页(limit 1–1000) |

**响应示例**：

```json
{"entities": [{"entity_id": "e1", "name": "…", "entity_type": "…", "aliases": [], "confidence": 0.9}], "total": 1}
```

### `GET /graph/entities/{entity_id}` — 实体详情

返回实体全字段(`properties` 已从 JSON 字符串还原为对象)+ `binds`(绑定父块)+ `relations`(出入关系边)。不存在 → `404`。

### `POST /graph/entities` — 新建/合并实体

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `entity_id` | string | 是 | 实体唯一 id |
| `name` | string | 是 | 实体名 |
| `entity_type` | string | 是 | 实体类型 |
| `aliases` | string[] | 否 | 别名 |
| `properties` | object | 否 | 属性 |
| `confidence` | number | 否 | 置信度,默认 1.0 |

### `PUT /graph/entities/{entity_id}` — 部分更新实体

只传要改的字段(`name` / `entity_type` / `aliases` / `properties` / `confidence` 任选)。不存在 → `404`。

### `DELETE /graph/entities/{entity_id}` — 删除实体

`DETACH DELETE`,连带清除该实体的 BINDS 与关系边。不存在 → `404`。成功 `204`。

---

## 父块节点 `/graph/nodes/segments`

> Neo4j 侧的 `:Segment` 节点读写,与 PG 侧的分段管理(`/graph/segments`,见 06)是两套视角;这里直读图。

### `GET /graph/nodes/segments` — 父块节点列表

| query | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 否 | 按知识库过滤 |
| `doc_id` | string | 否 | 按文档过滤 |
| `limit` / `offset` | int | 否 | 分页 |

### `GET /graph/nodes/segments/{parent_id}` — 父块节点详情

不存在 → `404`。

### `POST /graph/nodes/segments` — 新建/合并父块节点

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `parent_id` | string | 是 | 父块唯一 id |
| `doc_id` | string | 否 | 所属文档 |
| `kb_id` | UUID | 否 | 所属知识库 |
| `title` | string | 否 | 标题 |
| `enabled` | bool | 否 | 是否启用,默认 true |
| `chapter_no` | string | 否 | 章节号 |

### `PUT /graph/nodes/segments/{parent_id}` — 部分更新父块节点

只传要改的字段(`doc_id` / `kb_id` / `title` / `enabled` / `chapter_no`)。不存在 → `404`。

### `DELETE /graph/nodes/segments/{parent_id}` — 删除父块节点

`DETACH DELETE`,连带清除 BINDS 边。不存在 → `404`。成功 `204`。

---

## 关系边 `/graph/relations`

### `GET /graph/relations` — 关系边列表

| query | 类型 | 必填 | 说明 |
|---|---|---|---|
| `rel_type` | string | 否 | 按关系类型(英文标识,如 `BELONGS_TO`)过滤 |
| `kb_id` | UUID | 否 | 按知识库过滤(source_parent_id 归属该库) |
| `head_entity_id` / `tail_entity_id` | string | 否 | 按头/尾实体过滤 |
| `limit` / `offset` | int | 否 | 分页 |

**响应示例**：

```json
{"relations": [
  {"head_entity_id": "e1", "tail_entity_id": "e2", "rel_type": "EQUIPPED_ON",
   "confidence": 0.9, "evidence": "…", "source_parent_id": "p1", "derived": false}
], "total": 1}
```

### `POST /graph/relations` — 新建/合并关系边

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `head_entity_id` | string | 是 | 头实体 id |
| `tail_entity_id` | string | 是 | 尾实体 id |
| `rel_type` | string | 是 | 关系类型(合法标识符,须是字典里已定义的关系 `en`) |
| `confidence` | number | 否 | 置信度,默认 0.0 |
| `evidence` | string | 否 | 证据句 |
| `source_parent_id` | string | 否 | 来源父块 |
| `derived` | bool | 否 | 是否元数据派生边,默认 false |

`rel_type` 必须是合法 Cypher 关系标识(字母/下划线开头,仅字母数字下划线),否则 `422`。

### `DELETE /graph/relations` — 删除关系边

**query**：`head_entity_id` `tail_entity_id` `rel_type` 三个都必填,删除指定那条边。成功 `204`。

---

## 绑定边 `/graph/binds`

### `GET /graph/binds` — BINDS 边列表

| query | 类型 | 必填 | 说明 |
|---|---|---|---|
| `kb_id` | UUID | 否 | 按知识库过滤 |
| `entity_id` | string | 否 | 按实体过滤 |
| `parent_id` | string | 否 | 按父块过滤 |
| `limit` / `offset` | int | 否 | 分页 |

**响应示例**：

```json
{"binds": [{"entity_id": "e1", "parent_id": "p1", "kg_role": "primary", "kg_confidence": 0.9}], "total": 1}
```

### `POST /graph/binds` — 新建/合并 BINDS 边

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `entity_id` | string | 是 | 实体 id |
| `parent_id` | string | 是 | 父块 id |
| `kg_role` | string | 否 | `primary` / `supplemental`,默认 supplemental |
| `kg_confidence` | number | 否 | 置信度,默认 0.5 |

### `DELETE /graph/binds` — 删除 BINDS 边

**query**：`entity_id` `parent_id` 都必填。成功 `204`。
