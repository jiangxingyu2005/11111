# 分段管理 `/graph`

分段 = 父块(`parent_document` 表),`parent_id = {doc_id}_{章节号}`。三库(PG/Milvus/Neo4j)当场一致、幂等。

## `GET /graph/segments` — 分段列表

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/segments
```

**传出示例**(不含正文):

```json
{
  "segments": [
    {"parent_id": "8f14e45f-..._1", "doc_id": "8f14e45f-...", "kb_id": "e5a34e83-...",
     "title": "对空搜索雷达", "enabled": true, "content_version": 1, "recall_count": 0}
  ]
}
```

## `GET /graph/segments/{parent_id}` — 分段详情

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/segments/8f14e45f-..._1
```

**传出示例**:单分段(含正文)。不存在:`{"found": false}`(200,非 404)。

## `POST /graph/segments/{parent_id}/enabled` — 分段开关

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/segments/8f14e45f-..._1/enabled \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

**传出示例**:

```json
{"ok": true, "changed": true, "parent_id": "8f14e45f-..._1", "enabled": false, "children": 3}
```

- 幂等:再次传同值 → `{"ok": true, "changed": false, "reason": "no-op"}`。
- 不存在:`{"ok": false, "reason": "parent not found"}`。

## `POST /graph/documents/{doc_id}/enabled` — 文档级开关

对文档下所有分段一并生效。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/graph/documents/8f14e45f-.../enabled \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

**传出示例**:

```json
{"ok": true, "changed": 2, "total_segments": 2}
```

不存在:`{"ok": false, "reason": "document not found"}`。

## `PUT /graph/segments/{parent_id}` — 分段编辑

七步编辑:重切分 → 重嵌 → 增量重抽 → Neo4j 差量校正 → 回写 → CDC 留痕。

**传入示例**:

```bash
curl -X PUT http://127.0.0.1:8001/graph/segments/8f14e45f-..._1 \
  -H "Content-Type: application/json" \
  -d '{"content": "# 对空搜索雷达\n工作频率 2.4GHz，调制方式 QPSK。", "edited_by": "ops"}'
```

**传出示例**:

```json
{"ok": true, "content_version": 2, "removed_chunks": 0, "new_chunks": 2,
 "entities": 3, "relations": 1, "primary_binds": 2}
```

- `content_version` 自增;子块按新正文重切(`parent_id#1..#n`),多删少补。
- 不存在:`{"ok": false, "reason": "parent not found"}`。
