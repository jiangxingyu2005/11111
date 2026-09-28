# 接口文档

本文档统一说明 qingxi 服务的全部 HTTP 接口,分三部分:

- **清洗接口**(既有,本次融合**未改动其行为**)
- **GraphRAG 接口**(新增,ems_graphrag 融合)
- **IQ 检索库接口**(新增,IQ 信号相似度检索 + 入库打标)

所有业务端点均为 `async def`。Base URL 默认 `http://127.0.0.1:8001`(端口以实际启动为准)。

---

## 0. 通用约定

| 项 | 说明 |
|---|---|
| 认证 | 无(内网服务) |
| `kb_id` | 知识库 id(UUID),清洗/入图谱均按知识库隔离 |
| 时间 | ISO 8601 字符串(`2026-09-26T12:00:00+00:00` 之类) |
| 逻辑删除 | 删除接口均置 `deleted_at`,不物理删除 |
| 失败兜底 | 单文件处理失败返回 `success=false` + `error`,不中断整批 |

---

## 1. 清洗接口(既有)

### 1.1 文档解析 `/parse`

`POST /parse`(multipart/form-data)

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | file[] | 是 | 上传文件(PDF/图片/文档) |
| `kb_id` | UUID | 是 | 所属知识库 id |
| `return_content` | query bool | 否 | 是否在响应里返回清洗后 markdown 正文,默认 `true` |

流程:MinerU 解析 → 清洗 → 打分 → 落盘 → MinIO → 登记 `documents` → **GraphRAG 入库**(见 §3)。

响应 `ParseResponse`:

```json
{
  "results": [
    {
      "filename": "eq_00012.pdf",
      "doc_id": "8f14e45f-...",
      "success": true,
      "page_count": 12,
      "clean_md": "# ...(清洗后正文,return_content=true 时返回)",
      "files": {"raw_md": "...", "raw_json": "...", "clean_md": "...", "clean_json": "...", "meta": "..."},
      "stats": {"blocks_before": 120, "blocks_after": 100, "removed_images": 3, "...": 0},
      "error": null
    }
  ]
}
```

失败示例:`{"filename": "a.pdf", "doc_id": "", "success": false, "error": "处理失败: ..."}`。

> GraphRAG 入库是 `_finalize` 内部钩子,**失败不影响本接口返回值**;成功时该文档 `in_kb` 置为 `true`。

### 1.2 图片解析 `/parse-image`

`POST /parse-image`(multipart/form-data),参数同上。
MinerU 优先,抛错/空结果降级飞桨 PaddleOCR;BLIP 语义描述(英→中)并入正文。

### 1.3 音频解析 `/parse-audio`

`POST /parse-audio`(multipart/form-data),参数同上。
ASR(faster-whisper)转文字 → 清洗 → 打分 → 入库(不过 MinerU)。

### 1.4 视频解析 `/parse-video`

`POST /parse-video`(multipart/form-data),参数同上。
抽帧 → 去重 → 逐帧走图片链路 → 合并 → 清洗 → 打分 → 入库。

### 1.5 HTML 解析 `/parse-html`

`POST /parse-html`(multipart/form-data),参数同上。
提取正文(标题/段落/列表/表格转 markdown)→ 清洗 → 打分 → 入库。

### 1.6 IQ 信号解析 `/parse-iq`

`POST /parse-iq`(multipart/form-data),参数同上。
H5 → SNR 清洗 → IQ 补全 → 信号处理 → 26 特征中文描述入库。

响应 `IqParseResponse`:在 `FileResult` 基础上附 `snr / features / prediction / param_guess / iq_raw / iq_clean / ...` 等 IQ 处理中间结果。

> **IQ 不入图谱**:`/parse-iq` 只写 `documents` 表,不触发 GraphRAG 入库,`in_kb` 保持 `false`。

### 1.7 健康检查 `/health`

`GET /health` → `{"status": "ok|degraded", "mineru": {"ok": true, "detail": "..."}}`。

### 1.8 文档下载 `/documents/{doc_id}/download`

`GET /documents/{doc_id}/download` → 返回脱敏后的源文件(文本类脱敏;视频/音频原样)。
`GET /documents/{doc_id}/desensitize` → 同上(别名)。404:文档/源文件不存在。

### 1.9 知识库 `/knowledge-bases`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/knowledge-bases` | 新建 `{name, description?}` → 201 |
| GET | `/knowledge-bases` | 列表 |
| GET | `/knowledge-bases/{kb_id}` | 单个 → 404 不存在 |
| PUT | `/knowledge-bases/{kb_id}` | 更新 `{name?, description?}` |
| DELETE | `/knowledge-bases/{kb_id}` | 逻辑删除 → 204 |

---

## 2. GraphRAG 抽取字典接口(新增)

字典 = 定义「要抽取哪些实体节点 / 关系」的**唯一数据源**,全局共享(不分知识库),
抽取侧每次现查,改字典即时生效。

### 2.1 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | UUID | 主键 |
| `name` | string | 名称(实体类型中文名 / 关系中文名) |
| `en` | string? | 英文名(关系用,如 `BELONGS_TO`) |
| `category` | string | **`entity`(节点)/ `relation`(关系)** |
| `description` | string | 描述 |
| `json_schema` | object? | LLM function-calling 的 JSON Schema(openai 抽取用) |
| `trigger_words` | string[]? | 关系触发词(仅 relation) |
| `min_confidence` | number? | 关系最低置信度(默认 0.75) |
| `head_types` | string[]? | 关系 **head 端**允许的实体类型(中文类型名集合;仅 relation,空 = 不约束) |
| `tail_types` | string[]? | 关系 **tail 端**允许的实体类型(中文类型名集合;仅 relation,空 = 不约束) |
| `enabled` | bool | 是否启用 |
| `created_at` / `updated_at` | datetime | 时间戳 |

> 服务启动时自动写入**种子数据**:10 实体(原始信号/信号样本/信号簇/辐射源/平台/组织/工作状态/目标/事件/情报产品)
> + 11 关系(来源 `SOURCE_OF` / 组成 `COMPOSED_OF` / 归并为 `MERGED_INTO` / 型号匹配 `MODEL_MATCH` /
> 装备于 `EQUIPPED_ON` / 隶属 `BELONGS_TO` / 具有状态 `HAS_STATE` / 关联目标 `RELATED_TARGET` /
> 平台候选 `PLATFORM_CANDIDATE` / 产生 `PRODUCES` / 写入 `WRITTEN_INTO`)。
>
> **受控本体方向约束**:每条关系带 `head_types`/`tail_types` 约束两端实体类型(domain/range),
> 抽取校验 `head 类型 ∈ head_types`、`tail 类型 ∈ tail_types`,不满足 → 进 `pending_extraction`,
> 防止「辐射源—装备于→辐射源」这类语义荒谬的边。实体清单无「装备/型号」,`装备于`/`型号匹配` 两端用辐射源(=发射装备)与平台承载。
>
> 另有 3 个**派生关系类型**(`SAME_BAND`/`CO_TIME`/`CO_LOCATE`)不入字典:由 mesh 图增强(§10)
> 按元数据自动织入,带 `derived=true` 标记,与抽取边区分、可整体重算。

### 2.2 增删改查

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/graph/dict` | 新增 → 201 |
| GET | `/graph/dict` | 列表(不含已删),可 `?category=entity|relation` 过滤 |
| GET | `/graph/dict/{id}` | 单个 → 404 不存在 |
| PUT | `/graph/dict/{id}` | 更新(仅传要改字段) |
| DELETE | `/graph/dict/{id}` | 逻辑删除 → 204 |

**新增实体类型**:

```bash
curl -X POST http://127.0.0.1:8001/graph/dict \
  -H "Content-Type: application/json" \
  -d '{"name": "舰艇", "category": "entity", "description": "水面/水下作战平台"}'
```

**新增关系**:

```bash
curl -X POST http://127.0.0.1:8001/graph/dict \
  -H "Content-Type: application/json" \
  -d '{"name": "指挥", "en": "COMMANDS", "category": "relation",
       "trigger_words": ["指挥", "统一指挥", "受令"], "min_confidence": 0.8}'
```

**更新 / 逻辑删除**:

```bash
curl -X PUT  http://127.0.0.1:8001/graph/dict/<id> -H "Content-Type: application/json" -d '{"enabled": false}'
curl -X DELETE http://127.0.0.1:8001/graph/dict/<id>
```

错误码:`404` 条目不存在、`422` `category` 不是 `entity|relation`。

---

## 3. GraphRAG 入库说明(非接口)

清洗接口 `_finalize` 末尾自动触发 `ingest_document(kb_id, doc_id, clean_md, engine)`:

```
清洗 md → 父子块切分 → PG(parent_document / chunk_relation) → 源文件登记(source_file_registry)
      → 嵌入 → Milvus(ems_chunks) → 抽取实体/关系(字典驱动) → mesh 图增强(归一消歧 + 派生边)
      → Neo4j(Entity/Segment/BINDS/关系) → 回写 PG kg_* + Milvus kg_entity_ids → documents.in_kb = true
```

- **kb_id 贯穿**:PG 父块/源文件登记、Milvus 子块 `kb_id` 字段、Neo4j `:Segment.kb_id` 均带知识库 id。
- **幂等**:确定性哈希发号 + PG upsert + Neo4j MERGE + Milvus upsert,重复入库不产生重复数据。
- **失败隔离**:Milvus/Neo4j 连不上时,parse 主链路照常返回 `success=true`,仅告警,`in_kb` 保持 `false`。
- **LLM/embedding 默认 `mock`**(确定性正则 + 哈希向量,离线可跑可验证);切真 LLM 只需 `.env` 改
  `GRAPHRAG_LLM_PROVIDER`(支持 `openai` 与国产 `qwen`——本地 vLLM 起 OpenAI 兼容端点,`_BASE_URL`
  指向内网,数据不出网)+ `_API_KEY`/`_MODEL` 等,代码不动。embedding 另支持 `bge`
  (本地 FlagEmbedding,`BAAI/bge-m3` / `bge-large-zh-v1.5`,需 `pip install FlagEmbedding`),
  设 `GRAPHRAG_EMBEDDING_PROVIDER=bge` + `_MODEL` 即可,`embed`/`embed_one` 接口不变。bge 需先把模型
  下载到本地 HF 缓存(如 `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-large-zh-v1.5`),
  加载时自动解析本地 snapshot 路径、不访问 hub(离线可跑,不会因网络超时卡启动)。
- **mesh 图增强**(§10):入库时对抽取结果「过一道对齐」——别名/属性指纹重复的实体归一为同一 `entity_id`
  消歧,并织入元数据派生边 `SAME_BAND`/`CO_TIME`/`CO_LOCATE`(`derived=true`);mock 下无别名/指纹/事件
  元数据,该步为空操作,行为与融合前一致。

---

## 4. 配置(全部在 `.env`)

GraphRAG 相关(其余 postgres/minio 等复用既有):

| 变量 | 默认 | 说明 |
|---|---|---|
| `GRAPHRAG_ENABLED` | `true` | 是否清洗完自动入图谱 |
| `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION` / `MILVUS_DIM` | `localhost` / `19530` / `ems_chunks` / `1024` | Milvus |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `neo4j` | Neo4j |
| `GRAPHRAG_CHUNK_SIZE` / `GRAPHRAG_CHUNK_OVERLAP` | `512` / `0.10` | 子块切分 |
| `GRAPHRAG_EMBEDDING_PROVIDER` / `_MODEL` / `_DIM` / `_BATCH_SIZE` / `_DEVICE` / `_USE_FP16` | `mock` / `text-embedding-3-large` / `1024` / `64` / `` / `false` | 嵌入(provider 可 `mock`/`openai`/`bge`) |
| `GRAPHRAG_LLM_PROVIDER` / `_BASE_URL` / `_API_KEY` / `_MODEL` / `_MAX_RETRIES` | `mock` / `...` / `` / `gpt-4o` / `2` | LLM:`mock`=确定性正则;`openai`/`qwen`=function calling(OpenAI 兼容,`qwen` 国产本地 vLLM,数据不出网);`_MAX_RETRIES`=function calling 网络/解析失败重试次数 |
| `GRAPHRAG_CONFIDENCE_HIGH` / `_LOW` | `0.9` / `0.6` | 抽取置信度三档分流 |
| `GRAPHRAG_TAU_A` / `GRAPHRAG_TAU_B` | `0.6` / `0.9` | 关系抽取阈值 |
| `GRAPHRAG_ROUTE_FILE_SIZE_MB` / `GRAPHRAG_ROUTE_BATCH_COUNT` | `10` / `5` | 离线/实时分流阈值(§8) |
| `GRAPHRAG_RESOLVE_ENABLED` / `_FINGERPRINT` / `_VECTOR_ENABLED` / `_SIM_HIGH` / `_SIM_LOW` / `_LLM_GATE` / `_TOPK` | `true` / `true` / `true` / `0.62` / `0.55` / `false` / `5` | 实体消歧:入库对齐(§11);`_ENABLED=false` 回到 Phase 4 行为;阈值按 `bge-large-zh-v1.5` 实测校准(§11.4),换模型需重校准 |
| `GRAPHRAG_ENTITY_COLLECTION` | `ems_entities` | 实体名向量库名(Milvus,消歧语义匹配层 ②) |
| `GRAPHRAG_QUALITY_ENTITY_CHUNK_RATIO` / `_MAX_ORPHAN_RATE` / `_MIN_BINDS_COVERAGE` / `_MAX_EMPTY_RATIO` | `0.2` / `0.5` / `0.3` / `0.5` | 漏抽质检门禁(§10.3):实体数<父块数×比率→`low`;孤立率/覆盖率/空图谱占比超阈值→`healthy=false`(默认值待真实数据校准) |
| `KAFKA_BOOTSTRAP_SERVERS` / `KAFKA_TOPIC` / `KAFKA_GROUP_ID` | `localhost:9092` / `ems.public.parent_document` / `ems_cdc_group` | CDC 消费(§9.2,可选) |

> 配置**不走 yaml**:env 已有的(postgres 等)直接复用,新增项进 `.env`。
> mesh 图增强(§10)**无新增配置**:纯逻辑增强,容差阈值照搬 ems 字面量,无启用开关(mock 下空操作、openai 下激活)。

---

## 5. GraphRAG 在线查询(新增)

### 5.1 查询 `/graph/query`

`POST /graph/query`(application/json)

意图分流 → 实体匹配 → A/B 双路召回 → 重排 →(命中关系触发词时)图扩展 → LLM 生成带引用答案;命中结果计入溯源(见 §6)。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 自然语言问题 |
| `kb_id` | UUID | 否 | 知识库 id;不传 = 全库 |

```bash
curl -X POST http://127.0.0.1:8001/graph/query \
  -H "Content-Type: application/json" \
  -d '{"query": "2.4GHz 的调制方式是什么", "kb_id": "e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5"}'
```

响应:

```json
{
  "intent": "retrieval",
  "cond": {"intent": "local_fact", "modulation_type": null, "freq_band": "2.4GHz", "raw": "2.4GHz 的调制方式是什么"},
  "answer": "根据检索结果（3 条）：\n[1] 某舰艇装备一部对空搜索雷达，工作频率 2.4GHz…",
  "hits": [
    {"child_id": "{parent_id}#0", "doc_id": "8f14e45f-...", "content": "…", "_score": 0.81, "kg_entity_ids": ["<entity_id>"]}
  ],
  "matched_entities": ["<entity_id>"],
  "expanded_entities": []
}
```

字段说明:

| 字段 | 说明 |
|---|---|
| `intent` | `retrieval`(纯检索)/ `graph`(命中关系触发词,走图扩展) |
| `cond` | NL2Query 结构化条件(`modulation_type` / `freq_band`) |
| `answer` | 带 `[n]` 引用的答案(`n` 对应 `hits` 下标) |
| `hits` | 召回的命中子块(`_score` 为 COSINE 相似度 + 实体加权) |
| `matched_entities` | 查询词匹配到的实体 id |
| `expanded_entities` | 图扩展关联实体(`intent=graph` 时非空) |

错误:`422` 缺 `query` 字段;`502` Neo4j/Milvus 不可用(查询失败但不影响服务/主链路)。

---

## 6. GraphRAG 溯源(新增)

命中子块计入溯源:一次查询命中某**父块** → 该父块 `recall_count` +1(与命中子块数无关);
一次查询命中某**文件** → 该文件 `source_file_registry.recall_count` +1(与命中子块数无关);
每个命中子块写一条 `recall_log`(查询原文 / 命中子块 / 时间)。

### 6.1 文件列表 `/graph/trace/files`

`GET /graph/trace/files?kb_id={uuid}`(`kb_id` 可选,不传 = 全库)

```bash
curl "http://127.0.0.1:8001/graph/trace/files?kb_id=e5a34e83-5b8b-4ad1-bb2d-b91d1e730ca5"
```

响应:

```json
{
  "files": [
    {"doc_id": "8f14e45f-...", "file_name": "ems_test.html", "doc_type": "html",
     "event_time": null, "uploaded_at": "2026-09-26T12:00:00+00:00", "recall_count": 2}
  ]
}
```

### 6.2 文件详情 `/graph/trace/files/{doc_id}`

`GET /graph/trace/files/{doc_id}` — 文件信息 + 召回历史时间线。

```json
{
  "doc_id": "8f14e45f-...", "file_name": "ems_test.html", "found": true, "recall_count": 2,
  "history": [
    {"query": "2.4GHz 的调制方式是什么", "child_id": "{parent_id}#0", "parent_id": "{parent_id}", "ts": "2026-09-26T12:00:01+00:00"}
  ]
}
```

不存在:`{"doc_id": "...", "found": false}`(200,非 404)。

### 6.3 源文件定位 `/graph/trace/files/{doc_id}/locate`

`GET /graph/trace/files/{doc_id}/locate` — 返回源文件存储路径。

```json
{"doc_id": "8f14e45f-...", "found": true, "source_path": "minio://.../8f14e45f-.../ems_test.html"}
```

不存在:`{"doc_id": "...", "found": false, "source_path": null}`。

---

## 7. GraphRAG 分段管理(新增)

分段 = 父块(`parent_document` 表),`parent_id = {doc_id}_{章节号}`。开关与编辑均**只影响该分段/该文档**,三库(PG/Milvus/Neo4j)当场一致,幂等(状态不变不产生冗余写)。

### 7.1 分段列表 `GET /graph/segments`

`GET /graph/segments`(无参,全库;不含正文)

```bash
curl http://127.0.0.1:8001/graph/segments
```

响应:

```json
{
  "segments": [
    {"parent_id": "8f14e45f-..._1", "doc_id": "8f14e45f-...", "kb_id": "e5a34e83-...",
     "title": "对空搜索雷达", "enabled": true, "content_version": 1, "recall_count": 0}
  ]
}
```

### 7.2 分段详情 `GET /graph/segments/{parent_id}`

`GET /graph/segments/{parent_id}` — 单分段(含正文)。不存在返回 `{"found": false}`(200,非 404)。

```bash
curl http://127.0.0.1:8001/graph/segments/8f14e45f-..._1
```

### 7.3 分段开关 `POST /graph/segments/{parent_id}/enabled`

`POST /graph/segments/{parent_id}/enabled`(application/json)

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `enabled` | bool | 是 | 是否启用该分段(检索时 `enabled=false` 不参与召回) |

```bash
curl -X POST http://127.0.0.1:8001/graph/segments/8f14e45f-..._1/enabled \
  -H "Content-Type: application/json" -d '{"enabled": false}'
```

响应:

```json
{"ok": true, "changed": true, "parent_id": "8f14e45f-..._1", "enabled": false, "children": 3}
```

- 幂等:再次传同值 → `{"ok": true, "changed": false, "reason": "no-op"}`。
- 不存在:`{"ok": false, "reason": "parent not found"}`。

### 7.4 文档级开关 `POST /graph/documents/{doc_id}/enabled`

`POST /graph/documents/{doc_id}/enabled`(application/json,`{enabled: bool}`)—— 对文档下所有分段一并生效。

```json
{"ok": true, "changed": 2, "total_segments": 2}
```

不存在:`{"ok": false, "reason": "document not found"}`。

### 7.5 分段编辑 `PUT /graph/segments/{parent_id}`

`PUT /graph/segments/{parent_id}`(application/json)—— 七步编辑:重切分 → 重嵌 → 增量重抽 → Neo4j 差量校正 → 回写 → CDC 留痕。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `content` | string | 是 | 新的父块正文(markdown) |
| `edited_by` | string | 否 | 编辑人 |

```bash
curl -X PUT http://127.0.0.1:8001/graph/segments/8f14e45f-..._1 \
  -H "Content-Type: application/json" \
  -d '{"content": "# 对空搜索雷达\n工作频率 2.4GHz，调制方式 QPSK。", "edited_by": "ops"}'
```

响应:

```json
{"ok": true, "content_version": 2, "removed_chunks": 0, "new_chunks": 2,
 "entities": 3, "relations": 1, "primary_binds": 2}
```

- `content_version` 自增;子块按新正文重切(`parent_id#1..#n`),多删少补。
- 编辑后该分段的实体/BINDS 按新抽取结果**差量校正**(删旧绑、补新绑)。
- 不存在:`{"ok": false, "reason": "parent not found"}`。

---

## 8. GraphRAG 分流判定(新增)

`POST /graph/route`(application/json)—— 按规则判定实时/离线链路,**非破坏**(不改变 `/parse*` 行为,offline 仅提示,离线队列另行搭建)。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | `[{size: int}]` | 二选一 | 文件对象列表(字节) |
| `sizes` | `int[]` | 二选一 | 文件大小列表(字节) |

规则(阈值走 `.env`,见 §4):`R2` 任一文件 > `file_size_mb` MB → offline;`R3` 文件数 > `batch_count` → offline;否则 `R1` realtime。

```bash
curl -X POST http://127.0.0.1:8001/graph/route \
  -H "Content-Type: application/json" -d '{"sizes": [12582912]}'
```

响应:

```json
{"lane": "offline", "rule": "R2_FILE_SIZE", "tip": "该文件体量较大，已加入后台全量索引构建任务，预计处理耗时较长，完成后将推送通知。"}
```

---

## 9. GraphRAG 三方对账 + CDC(新增)

### 9.1 对账 `POST /graph/reconcile`

`POST /graph/reconcile`(无 body)—— 比对 PG `parent_document` ↔ Neo4j `:Segment` 集合,并统计图内 BINDS 总数。

```bash
curl -X POST http://127.0.0.1:8001/graph/reconcile
```

响应:

```json
{
  "consistent": true,
  "pg_parents": 3,
  "graph_segments": 3,
  "missing_in_graph": [],
  "orphan_in_graph": [],
  "graph_binds": 13
}
```

- `consistent=true` 表示 PG 与 Neo4j 集合一致;`missing_in_graph` = PG 有图无、`orphan_in_graph` = 图有 PG 无。

### 9.2 CDC 消费循环(非 HTTP,可选旁路)

`python -m app.graphrag.cdc.consumer` 起**独立消费进程**,从 Kafka(经 Debezium 捕获 PG `parent_document` 的 WAL)增量同步 Neo4j:

- `op=c/r` 增 → `merge_segment` + 绑定;`op=u` 改 → `merge_segment` + 差量校正;`op=d` 删 → `DETACH DELETE`。
- **重放安全**:MERGE 幂等,同批消息重放两次图无重复。
- 分段编辑/开关**不依赖 Kafka**:已当场直接同步三库;Kafka 消费仅用于「外部直写 PG」触发的增量兜底。

启用外部 WAL 需另建 Debezium + Kafka Connect(连接器参考 `app/graphrag/cdc/debezium_connector.json`)并开启 PG `wal_level=logical`;qingxi 服务本身不读该 JSON,也不在启动时起 Kafka 任务。

---

## 10. GraphRAG 图增强 / 图质量(mesh,新增)

mesh = 别名+属性指纹归一消歧、孤儿兜底绑定、元数据派生边、图质量四指标。实体全局共享(不分知识库)。

### 10.1 图质量指标 `GET /graph/metrics`

`GET /graph/metrics`(无参)—— 读 Neo4j 全图(实体/BINDS/关系)+ PG 父块,返回四指标 + 总数 + 孤立实体。

```bash
curl http://127.0.0.1:8001/graph/metrics
```

响应:

```json
{
  "orphan_rate": 0.125,
  "binds_coverage": 1.0,
  "max_connected_component": 6,
  "avg_degree": 1.333,
  "totals": {"entities": 8, "relations": 7, "parents": 4},
  "orphan_entities": [{"entity_id": "<entity_id>", "name": "某装备"}]
}
```

| 字段 | 说明 |
|---|---|
| `orphan_rate` | 孤立率 = 无任何 BINDS 的实体 / 实体总数 |
| `binds_coverage` | BINDS 覆盖率 = 已绑定父块数 / 父块总数 |
| `max_connected_component` | 关系图最大连通分量大小 |
| `avg_degree` | 关系图平均度 |
| `totals` | 实体/关系/父块计数 |
| `orphan_entities` | 孤立实体列表(id + name) |

错误:`502` Neo4j 不可用。

### 10.2 孤儿兜底绑定 `POST /graph/mesh/link-orphans`

`POST /graph/mesh/link-orphans`(无 body)—— 对 `kg_entity_id` 为空的父块,按别名/指纹匹配现有实体做兜底绑定(第二道防线,防孤立星);幂等。

```bash
curl -X POST http://127.0.0.1:8001/graph/mesh/link-orphans
```

响应:

```json
{
  "linked": [{"parent_id": "8f14e45f-..._2", "entity_id": "<entity_id>"}],
  "reextract": ["8f14e45f-..._3"]
}
```

| 字段 | 说明 |
|---|---|
| `linked` | 命中并已绑定的父块(写 Neo4j `BINDS`(role=`supplemental`,conf=0.5)+ 回写 PG `kg_*`) |
| `reextract` | 未命中的父块(写入 `pending_extraction` 留痕,待离线回炉抽取) |

- 幂等:重复调用,已绑定的父块 `kg_entity_id` 非空后不再处理。
- 命中判定:父块 `title`/`keywords` 别名命中,或 `freq_band` 指纹命中现有实体。
- 错误:`502` Neo4j 不可用。

### 10.3 漏抽质检门禁 `GET /graph/quality`

`GET /graph/quality`(无参)—— 读 `ingest_quality` 表 + Neo4j 全图,返回四指标 + 健康度 + 漏抽文档清单。让「抽少/抽空」从静默失败变为可见。

```bash
curl http://127.0.0.1:8001/graph/quality
```

响应:

```json
{
  "metrics": {"orphan_rate": 0.0, "binds_coverage": 1.0, "max_connected_component": 3, "avg_degree": 1.0},
  "totals": {"entities": 8, "relations": 5, "parents": 4, "documents_checked": 2, "empty_documents": 0},
  "healthy": true,
  "gates": {
    "orphan_rate": {"value": 0.0, "max": 0.5},
    "binds_coverage": {"value": 1.0, "min": 0.3},
    "empty_ratio": {"value": 0.0, "max": 0.5}
  },
  "leak_documents": []
}
```

| 字段 | 说明 |
|---|---|
| `metrics` | 同 `GET /graph/metrics` 的四指标 |
| `totals` | 实体/关系/父块计数 + `documents_checked`(已质检文档数)+ `empty_documents`(空图谱文档数) |
| `healthy` | 健康度 = 孤立率 ≤ `_MAX_ORPHAN_RATE` 且 BINDS 覆盖率 ≥ `_MIN_BINDS_COVERAGE` 且空图谱占比 ≤ `_MAX_EMPTY_RATIO` |
| `gates` | 各门禁的当前值 vs 阈值 |
| `leak_documents` | 漏抽文档清单(`kg_status` != `ok` 的 `{doc_id, file_name, chunk_count, entity_count, relation_count, bind_count, kg_status}`) |

- **`ingest_quality` 表**(PG):入库时每文档写一行质检(`entity_count==0` → `kg_status=empty` 并告警;`entity_count < 父块数×_ENTITY_CHUNK_RATIO` → `low`;否则 `ok`)。旧库由 `create_all` 自动建表。
- 错误:`502` Neo4j 不可用。

---

## 11. 实体消歧(归一对齐,新增)

三阶段匹配管道(命中即返,类型作用域):① 确定性(名称/别名/属性指纹)→ ② 向量语义(`ems_entities` 余弦)→ ③ LLM 灰区(`_LLM_GATE=true` 且 provider=openai 才激活,默认关)。落库两条路径:**入库在线对齐** + **全图批量对账**。

### 11.1 数据层

- `entity_registry` 表(PG):全局 canonical 实体注册表,列 `entity_id/name/entity_type/aliases/fingerprint/properties/status(active|merged)/merged_into/created_at`;`status=merged` 为对账账本。
- `ems_entities` collection(Milvus):实体名向量库,字段 `entity_id(PK)/entity_type/name/name_vector`(COSINE);启动时随 `ems_chunks` 一起建(`_bootstrap_graphrag`)。
- `extraction_dict` 表加 `semantic_mergeable`(布尔,默认 `false`):标记该实体类型是否允许向量语义层做同义异名归并(§11.5)。旧库由 `seed_extraction_dict` 幂等 `ALTER TABLE ADD COLUMN IF NOT EXISTS` 补列并首次回填。

### 11.2 入库在线对齐(自动)

文档入库在实体/关系抽取后、图增强前插入 `align + persist`(ingest step 6.5/6.6;分段编辑路径同步插入):每个实体先确定性/向量匹配全局 canonical,命中则 `remap` 到 canonical(关系 head/tail 同步重映射、自环丢弃),再写回注册表 + 实体向量。`GRAPHRAG_RESOLVE_ENABLED=false` 时短路,行为回到 Phase 4。

### 11.3 全量对账 `POST /graph/resolve`

`POST /graph/resolve?dry_run=false`—— 对历史重复实体批量归并(DSU 求 canonical,组内最早者为根),并四层回填旧数据:

1. PG `entity_registry`:`old` 标记 `merged`+`merged_into`,`canonical` 吸收 aliases/指纹/属性。
2. Neo4j:BINDS/关系边迁移 `old → new`,自环丢弃,`DETACH DELETE old`。
3. PG `parent_document.kg_entity_id`(逗号串)token 级重映射去重。
4. Milvus `ems_chunks.kg_entity_ids` 数组重映射去重 + `ems_entities` 删冗余向量。

```bash
# 预演(只返回归并计划,不落库)
curl -X POST "http://127.0.0.1:8001/graph/resolve?dry_run=true"
# 落库执行
curl -X POST http://127.0.0.1:8001/graph/resolve
```

响应:

```json
{
  "dry_run": false,
  "merged": [{"old_id": "<old>", "new_id": "<canonical>", "name": "雷达站"}],
  "binds_moved": 1,
  "relations_moved": 1,
  "parents_updated": 1,
  "chunks_updated": 1,
  "entities_before": 2,
  "entities_after": 1
}
```

- `dry_run=true` 时 `merged` 为归并计划,四层不变。
- **幂等**:重复调用 `merged=[]`、计数为 0;canonical 取组内 `created_at` 最早(并列取 `entity_id` 字典序)。
- 错误:`502` Neo4j/Milvus 不可用;向量层不可用时静默降级为确定性(不阻断)。

### 11.4 真实 BGE 语义验证与阈值校准

消歧语义层 ② 用 `bge-large-zh-v1.5`(1024 维)在 `text/` 真实数据上实测(`app/graphrag/disambig/verify_text_bge.py`):

- **相似度分布**(cos):正例「雷达 ↔ 对空搜索雷达 / 雷达站」= **0.693 / 0.761**;负例「雷达 ↔ 干扰机/侦察设备/舰艇/导弹/频率」= 0.19~0.49,gap 清晰。
- **评分**:确定性层 F1=1.00、向量语义层 F1=1.00(TP=2 FP=0 FN=0)、无别名端到端归并 2/2、类型作用域负例零误并。

阈值因此从初版 `sim_high=0.95 / sim_low=0.85`(按合成向量 0.999 设定)**校准为 `0.62 / 0.55`**:真实 BGE 的语义相似度量级(0.6~0.8 已算相近)远低于合成向量,原阈值会让语义层完全失效(雷达↔雷达站 0.76 连灰区下界都够不到)。

> **注意**:BGE 语义空间里「异名同义」与「参数值实体」存在窄带重叠——`雷达站↔对空搜索雷达`≈0.63 与 `2.4GHz↔850MHz`≈0.60 的 gap 仅 ~0.03。缓解靠三点:① 类型作用域隔离(频率/辐射源等参数值实体与装备实体不同 `entity_type`);② 确定性指纹层优先(同频段/脉宽/重频的装备先命中);③ 灰区 LLM 判定(`_LLM_GATE=true` 时 0.55~0.62 交 openai 裁决,默认关)。**换 embedding 模型(mock/openai/bge-m3)后相似度分布不同,需重新校准 `_SIM_HIGH`/`_SIM_LOW`。**

### 11.5 参数值类型不进向量层(窄带重叠的根治)

§11.4 的窄带重叠根源在于:向量余弦衡量「语义相关」而非「实体同一」,而参数值实体(2.4GHz/850MHz)与异名同义实体(雷达站/对空搜索雷达)在语义空间里天然贴近。根治办法是**把不该语义合并的参数值/枚举类实体请出向量层**——它们有精确键(数值+单位/枚举名),差异是确定性的,本就不需要语义判断:

- **机制**:`extraction_dict.semantic_mergeable` 标记该类型是否允许向量语义归并。种子标注:名词性(装备/辐射源/平台/任务)= `true` 进 `ems_entities`;参数值/枚举/坐标(频率/调制方式/信号/地理位置)= `false`,只走确定性匹配(名称精确 + 指纹)。
- **三处过滤**(`resolve.py`):`resolve_vector`(查向量前)、`persist`(写向量前)、`global_resolve`(向量 union 前)统一用 `load_mergeable_types()` 返回的可向量化类型集合过滤,集合外的类型直接跳过向量层。
- **收益**:`2.4GHz↔850MHz` 这类语义相近但非同一实体的窄带误并从根源消失;`ems_entities` 只存名词性实体,规模更小、召回更干净、`_SIM_HIGH/_SIM_LOW` 的校准余量拉开。
- **兼容**:`resolve_vector` 的 `mergeable` 参数默认 `None`=不限制(旧调用不变);字典查询失败降级为不限制,消歧链路不因字典故障崩。用户可在 `/graph/dict` 改 `semantic_mergeable` 调整某类型是否参与向量归并。

---

## 12. 抽取评测与阈值校准(真实数据到位后)

「先搭框架」:抽取已切可配 LLM(`qwen`/`openai`),字典内容(description/schema/触发词)已驱动 LLM prompt,质检门禁已落地;但**种子字典内容与阈值数值仍是模拟数据所设**,真实数据到位后按下述步骤校准。

### 12.1 抽取置信度阈值(`GRAPHRAG_CONFIDENCE_HIGH/_LOW`,默认 `0.9/0.6`)

1. 填金标准:`app/graphrag/eval/extract_eval.py` 的 `GOLDEN`(text + 期望实体/关系)。
2. 跑 `python -m app.graphrag.eval.extract_eval`(需 `.env` 已切 `qwen`/`openai` + 可用的 base_url)。
3. 按输出的实体/关系 P/R/F1 与 confidence 分桶召回率,取使 F1 最大的切点作为 `_LOW`(低于弃用)、`_HIGH`(直接采用)。

### 12.2 消歧相似度阈值(`GRAPHRAG_RESOLVE_SIM_HIGH/_SIM_LOW`,默认 `0.62/0.55`)

- 换真实实体名分布重跑 `app/graphrag/disambig/verify_text_bge.py`(当前值按 `bge-large-zh-v1.5` 实测 §11.4)。
- **换 embedding 模型(mock/openai/bge-m3)必须重校准**:不同模型相似度分布不同。

### 12.3 查询参数(`GRAPHRAG_QUERY_TOPK/_RERANK_TOPK/_EXPAND_DEPTH/_ENTITY_BOOST_WEIGHT`)

- 真实查询集 + 标注答案,扫参找最佳命中率。

### 12.4 字典重定义(`/graph/dict`)

- 实体类型:改 `description`/`json_schema` 讲清每类含义(已随 `load_entity_defs` 进 LLM prompt)。
- 关系:改 `description`/`trigger_words` 讲清每类关系语义与触发词(已随 `load_relations` 进 LLM prompt)。
- 改后即时生效(抽取侧每次现查),无需改代码、无需重启。

---

## 13. IQ 信号专属检索库(新增)

IQ/H5 除「信号解析 → 26 特征中文描述入库」(`/parse-iq`,§1.6)外,另建一条**独立于现有知识库/图谱的 IQ 检索库** `ems_iq`(256 维),专做 IQ 信号相似度检索与人工打标。两条接口:

- `POST /iq/ingest`(入库打标):传文档 uuid + 类别 → IQ 向量 + 标量写入 `ems_iq`,并把清洗后 md 追加类别后正常入知识库/知识图谱。
- `POST /iq/search`(判断类型):传文档 uuid → 向量化 → 最近邻,返回「类型 + 概率」;向量库为空返回「未知」。

场景:先检索判断信号是哪种雷达型号(返回类型 + 概率);概率低时人工打标签(类别),再入库固化。

### 13.1 数据模型

`ems_iq`(Milvus,独立于 `ems_chunks`/`ems_entities`,也不进 `knowledge_bases` 表):

| 字段 | 类型 | 说明 |
|---|---|---|
| `iq_id` | VARCHAR(PK) | 文档 uuid 字符串 |
| `category` | VARCHAR(256) | 人工类别标签(雷达型号/类型,自由文本) |
| `modulation` / `mod_id` | VARCHAR(64) / INT64 | 调制方式预测 |
| `snr` / `sample_rate_hz` / `symbol_rate` / `attenuation_db` | FLOAT | 信号标量 |
| 24 个特征 | FLOAT | 26 特征里的数值子集(幅度/相位/频率/PSD/谱/累积量等) |
| `embedding` | FLOAT_VECTOR(256) | WavesFM 256 维 CLS 特征(L2 归一化,COSINE/IVF_FLAT) |

### 13.2 入库打标 `POST /iq/ingest`

`POST /iq/ingest`(application/json)

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | UUID | 是 | 文档 uuid(须先经 `/parse-iq` 登记) |
| `category` | string | 是 | 人工类别标签(雷达型号/类型);空串 → 422 |

```bash
curl -X POST http://127.0.0.1:8001/iq/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "8f14e45f-...", "category": "某型对空搜索雷达"}'
```

流程:取回原始 H5 → 重跑 IQ 管道(归一化片段 + 26 特征/md)→ 文档级 256 维向量(片段 mean-pool)→ `upsert` 进 `ems_iq`(含 category + 标量子集)→ 若 `GRAPHRAG_ENABLED=true`,`md + f"\n\n类别:{category}"` 走 `ingest_document` 入知识库/知识图谱并 `mark_document_in_kb`(与普通文本一致)。

响应 `IqIngestResponse`:

```json
{"doc_id": "8f14e45f-...", "category": "某型对空搜索雷达", "vector_dim": 256, "graph": {"...": "..."}}
```

| 字段 | 说明 |
|---|---|
| `doc_id` | 文档 uuid |
| `category` | 打标类别 |
| `vector_dim` | 向量维度(256) |
| `graph` | 图谱入库计数摘要(`GRAPHRAG_ENABLED=false` 或图谱入库失败时为 `null`) |

错误:`404` 文档不存在 / 源 H5 取回失败;`422` `category` 为空。

### 13.3 判断类型 `POST /iq/search`

`POST /iq/search`(application/json)

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | UUID | 是 | 文档 uuid |

```bash
curl -X POST http://127.0.0.1:8001/iq/search \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "8f14e45f-..."}'
```

流程:取回 H5 → 向量化 → `ems_iq` 最近邻(`topk=3`,排除自身 `iq_id != doc_id`)→ 取最近邻的 `category` 作类型、余弦相似度作概率。

命中响应 `IqSearchResponse`:

```json
{"status": "ok", "type": "某型对空搜索雷达", "probability": 0.92}
```

空库 / 无邻居响应:

```json
{"status": "unknown", "type": "未知", "probability": null}
```

| 字段 | 说明 |
|---|---|
| `status` | `ok`(命中)/ `unknown`(向量库为空或无邻居) |
| `type` | 最近邻已打标信号的类别(=`category` 人工标签) |
| `probability` | 最近邻余弦相似度(0~1);概率低 = 置信度低,可转人工打标签 |

错误:`404` 文档不存在 / 源 H5 取回失败。

### 13.4 配置(全部在 `.env`)

| 变量 | 默认 | 说明 |
|---|---|---|
| `IQ_EMBEDDING_MODEL_PATH` | `model/iq/wavesfm_embed.onnx` | WavesFM 256 维嵌入模型(从 dc `wavesfm_rml` 重导出,暴露 CLS 特征) |
| `IQ_COLLECTION` | `ems_iq` | IQ 检索库集合名(独立于 `ems_chunks`/`ems_entities`) |
| `IQ_DIM` | `256` | IQ 嵌入维度 |

> IQ 检索库集合 `ems_iq` 在服务启动时随 `lifespan` 自动创建(独立于 `GRAPHRAG_ENABLED`),连不上 Milvus 不阻断启动,仅告警。
