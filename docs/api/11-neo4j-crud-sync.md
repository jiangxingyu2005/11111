# Neo4j 增删同步接口

前端在知识图谱页面手动增删实体时，**同步写清 PG `entity_registry` + Milvus `ems_entities`**，保证消歧、向量检索、入库合并都能找到/清理这些实体。

无认证。Base URL 以实际部署为准。

---

## 1. 三库同步行为

| 操作 | Neo4j | PG `entity_registry` | Milvus `ems_entities` |
|---|---|---|---|
| `POST /graph/entities` | MERGE 节点 | upsert 注册表记录 | upsert 实体名向量 |
| `DELETE /graph/entities/{id}` | DETACH DELETE | 删除记录 | 删除向量 |
| `PUT /graph/entities/{id}` | 更新字段 | — | — |

**不传 `entity_id` 时自动生成** `canonical_entity_id(name, entity_type)`，与入库流程同一套 ID，不会出现「手动加的节点」和「入库抽的节点」重复。

---

## 2. 添加实体

```http
POST /graph/entities
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `entity_id` | string | 否 | 不传则自动生成 |
| `name` | string | 是 | 实体名 |
| `entity_type` | string | 是 | 实体类型（辐射源/平台/武器/组织/…） |
| `aliases` | string[] | 否 | 别名 |
| `properties` | object | 否 | 属性，如 `{"频率": "2.4GHz", "国别": "美国"}` |
| `confidence` | number | 否 | 置信度，默认 1.0 |

**请求示例**：

```json
{
  "name": "AN/APG-68",
  "entity_type": "辐射源",
  "aliases": ["APG-68"],
  "properties": {"频率": "8-12GHz", "国别": "美国", "型号": "AN/APG-68"},
  "confidence": 0.9
}
```

**响应 201**：

```json
{
  "entity_id": "21535d3dbc0302b7",
  "name": "an/apg-68",
  "entity_type": "辐射源",
  "aliases": ["APG-68"],
  "properties": {"频率": "8-12GHz", "国别": "美国", "型号": "AN/APG-68"},
  "confidence": 0.9
}
```

---

## 3. 删除实体

```http
DELETE /graph/entities/{entity_id}
```

`DETACH DELETE`，连带清除 BINDS 与关系边。**同步清 PG 注册表 + Milvus 向量**。不存在 → `404`。成功 `204`。

---

## 4. 更新实体（仅 Neo4j）

```http
PUT /graph/entities/{entity_id}
```

只传要改的字段（`name` / `entity_type` / `aliases` / `properties` / `confidence` 任选）。不存在 → `404`。  
**注意**：此接口只改 Neo4j，不同步 PG/Milvus。如需同步，请用删除+重建。

---

## 5. 查询（读回）

```http
GET /graph/entities              # 列表，支持 name / entity_type / kb_id 过滤
GET /graph/entities/{entity_id}  # 详情（属性 + 一跳关系 + BINDS）
```

返回结构见 [10-graph-neo4j.md](10-graph-neo4j.md)。

---

## 6. 错误码

| HTTP | 含义 |
|---|---|
| 404 | 实体不存在 |
| 422 | 缺 `name` / `entity_type` 或类型非法 |
| 502 | Neo4j 不可用 |
