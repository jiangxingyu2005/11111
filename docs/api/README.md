# 接口文档目录

qingxi 服务全部 HTTP 接口,按类别拆分到本目录各文件。Base URL 默认 `http://192.168.31.15:8001`(端口以实际启动为准)。所有业务端点均为 `async def`,无认证(内网服务)。

> 实现细节、配置项、消歧/质检等机制说明见 [../api.md](../api.md)。

## 类别索引

| 类别 | 文件 |
|---|---|
| 清洗解析 | [01-cleaning.md](01-cleaning.md) |
| 知识库管理 | [02-knowledge-bases.md](02-knowledge-bases.md) |
| 文档下载 | [03-documents.md](03-documents.md) |
| 抽取字典 | [04-graph-dict.md](04-graph-dict.md) |
| 在线查询与溯源 | [05-graph-query-trace.md](05-graph-query-trace.md) |
| 分段管理 | [06-graph-segments.md](06-graph-segments.md) |
| 路由与对账 | [07-graph-route-reconcile.md](07-graph-route-reconcile.md) |
| 图质量与消歧 | [08-graph-metrics-resolve.md](08-graph-metrics-resolve.md) |
| IQ 检索库 | [09-iq.md](09-iq.md) |
| Neo4j 图查询/管理 | [10-graph-neo4j.md](10-graph-neo4j.md) |
| 预设推理规则 | [10-rules.md](10-rules.md) |

## 端点总览

| 类别 | 端点 | 方法 |
|---|---|---|
| 清洗解析 | `/parse` `/parse-image` `/parse-audio` `/parse-video` `/parse-html` `/parse-iq` | POST |
| 健康检查 | `/health` | GET |
| 知识库管理 | `/knowledge-bases` `/knowledge-bases/{kb_id}` `/knowledge-bases/{kb_id}/documents` | POST/GET/PUT/DELETE |
| 文档下载 | `/documents/{doc_id}/download` `/documents/{doc_id}/desensitize` | GET |
| 抽取字典 | `/graph/dict` `/graph/dict/{dict_id}` | POST/GET/PUT/DELETE |
| 在线查询与溯源 | `/graph/query` `/graph/query/stream` `/graph/trace/files` `/graph/trace/files/{doc_id}` `/graph/trace/files/{doc_id}/locate` | POST/GET |
| 分段管理 | `/graph/segments` `/graph/segments/{parent_id}` `/graph/segments/{parent_id}/enabled` `/graph/documents/{doc_id}/enabled` `/graph/documents/{doc_id}/blocks` | GET/POST/PUT |
| 路由与对账 | `/graph/route` `/graph/reconcile` | POST |
| 图质量与消歧 | `/graph/metrics` `/graph/quality` `/graph/mesh/link-orphans` `/graph/resolve` | GET/POST |
| IQ 检索库 | `/iq/ingest` `/iq/search` | POST |
| Neo4j 图查询/管理 | `/graph/nodes` `/graph/entities` `/graph/entities/{entity_id}` `/graph/nodes/segments` `/graph/nodes/segments/{parent_id}` `/graph/relations` `/graph/binds` | GET/POST/PUT/DELETE |
| 预设推理规则 | `/rules` `/rules/{rule_id}` | POST/GET/PUT/DELETE |

## 通用约定

- **认证**:无(内网服务)。
- **`kb_id`**:知识库 id(UUID),清洗/入图谱按知识库隔离。
- **时间**:ISO 8601 字符串。
- **逻辑删除**:删除接口均置 `deleted_at`,不物理删除。
- **失败兜底**:单文件处理失败返回 `success=false` + `error`,不中断整批;GraphRAG 入库失败不影响 parse 主链路(仅告警,`in_kb` 保持 `false`)。
- **配置**:全部在 `.env`,不走 yaml。
