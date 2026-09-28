# 知识库管理 `/knowledge-bases`

## `POST /knowledge-bases` — 新建

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "雷达情报库", "description": "雷达信号与情报文档"}'
```

**传出示例**(201):

```json
{"id": "2db770f2-7a09-4f04-9f07-fce5e6bf6529", "name": "雷达情报库", "description": "雷达信号与情报文档",
 "created_at": "2026-09-28T07:17:47.025473Z", "updated_at": "2026-09-28T07:17:47.025473Z"}
```

## `GET /knowledge-bases` — 列表

**传入示例**:

```bash
curl http://127.0.0.1:8001/knowledge-bases
```

**传出示例**:

```json
[
  {"id": "2db770f2-7a09-4f04-9f07-fce5e6bf6529", "name": "雷达情报库", "description": "雷达信号与情报文档",
   "created_at": "2026-09-28T07:17:47.025473Z", "updated_at": "2026-09-28T07:17:47.025473Z"}
]
```

## `GET /knowledge-bases/{kb_id}` — 单个

**传入示例**:

```bash
curl http://127.0.0.1:8001/knowledge-bases/2db770f2-7a09-4f04-9f07-fce5e6bf6529
```

**传出示例**:同单个对象;不存在 → 404。

## `PUT /knowledge-bases/{kb_id}` — 更新

**传入示例**:

```bash
curl -X PUT http://127.0.0.1:8001/knowledge-bases/2db770f2-7a09-4f04-9f07-fce5e6bf6529 \
  -H "Content-Type: application/json" \
  -d '{"description": "更新后的描述"}'
```

**传出示例**:更新后的 `KnowledgeBaseRead`。

## `DELETE /knowledge-bases/{kb_id}` — 逻辑删除

**传入示例**:

```bash
curl -X DELETE http://127.0.0.1:8001/knowledge-bases/2db770f2-7a09-4f04-9f07-fce5e6bf6529
```

**传出示例**:204 无 body(置 `deleted_at`,不物理删除)。
