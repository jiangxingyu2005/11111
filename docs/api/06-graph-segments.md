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

## `GET /graph/documents/{doc_id}/blocks` — 文档父块查看

给定文件 uuid,返回该文档全部父块(含正文,按章节号数值排序)+ 拼接后的清洗文档(标题行按 `level` 重写回正文)。

**传入示例**:

```bash
curl http://127.0.0.1:8001/graph/documents/8f14e45f-.../blocks
```

**传出示例**:

```json
{
  "doc_id": "8f14e45f-...", "filename": "ems_test.html", "block_count": 2,
  "blocks": [
    {"parent_id": "8f14e45f-..._1", "title": "对空搜索雷达", "level": 1, "chapter_no": "1",
     "content": "工作频率 2.4GHz，调制方式 QPSK。", "enabled": true, "content_version": 1, "recall_count": 0},
    {"parent_id": "8f14e45f-..._1.1", "title": "参数", "level": 2, "chapter_no": "1.1",
     "content": "脉冲宽度 12us。", "enabled": true, "content_version": 1, "recall_count": 0}
  ],
  "content": "# 对空搜索雷达\n\n工作频率 2.4GHz，调制方式 QPSK。\n\n## 参数\n\n脉冲宽度 12us。"
}
```

| 字段 | 说明 |
|---|---|
| `doc_id` | 文档 uuid |
| `filename` | 原文件名 |
| `block_count` | 父块数量 |
| `blocks` | 父块列表,按 `chapter_no` 数值排序(`1.10` 排在 `1.2` 后) |
| `content` | 按序拼接后的完整 markdown |

- 文档未入图谱(无父块):`blocks=[]`、`block_count=0`、`content=""`;无标题文档回退为单块 `title="无标题"`。
- 拼接正文反映**图谱当前状态**(含分段编辑后内容),非原始 clean.md 字节。
- 错误:`404` 文档不存在/已删除。

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
