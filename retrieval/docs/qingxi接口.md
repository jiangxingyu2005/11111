# qingxi 连接接口说明（retrieval ↔ qingxi 对接清单）

> 本文是后续把 `retrieval/`（在线检索）从 `fake` 通道切到 `real` 通道、直连 qingxi 入库产物的**唯一对接清单**。
> 本次交付**不连接** qingxi，仅把「需要的接口 / 表 / 字段 / 账号 / 环境变量 / 待对齐差异」一次性摸清记录在案。
>
> **qingxi 侧权威出处**（以这些为准，本文如有出入以源码为准）：
> - HTTP API：`qingxi/docs/api.md`（qingxi 主服务文档解析清洗服务）
> - PG 表：`qingxi/app/models/graphrag.py`（ORM 模型）
> - Milvus：`qingxi/app/graphrag/milvus_client.py`
> - Neo4j：`qingxi/app/graphrag/neo4j_client.py` + `neo4j_load.py`
> - 字典/关系种子：`qingxi/app/graphrag/seed.py`（10 实体 + 11 关系）+ `mesh/derived.py`（3 派生关系）

---

## 0. 结论速览

| # | 事项 | 现状 | 对接动作 |
|---|---|---|---|
| 1 | 三库只读账号 | qingxi 侧已规划 `retrieval_ro`；retrieval 侧 `sql/001_init.sql` 已写同名建号脚本 | 开通账号，写操作全拒为验收（IF-9） |
| 2 | Milvus 向量字段 | qingxi=`content_vector`；retrieval dev 脚本=`embedding` | **待对齐**（见 §8 对照表） |
| 3 | PG 子块表名 | qingxi=`chunk_relation`；retrieval dev 脚本=`child_chunk` | **待对齐** |
| 4 | Neo4j 实体类型/证据字段 | qingxi=`entity_type` / `evidence`；retrieval 查询写 `type` / `evidence_child_id+span` | **待对齐** |
| 5 | 模型空间 | 双方同为 BGE-M3 dense 1024（定案 D4），`embed_m3_full()` 提供 dense+sparse 双通道 | 一致，sparse 待 IF-1 扩列后启用 |

> 一句话：**连接信息（账号/地址/库名/维度）基本对齐，但「字段名」有多处不一致**，real 联调前必须先按 §8 对照表对齐，否则会静默召回噪声或查空。

---

## 1. 只读账号契约（IF-9）

retrieval 侧**只读**消费 qingxi 的三库，禁止任何写操作。验收条件 = **写操作全拒**。

| 库 | 只读账号 | 用途 |
|---|---|---|
| PostgreSQL | `retrieval_ro` | QRY-06 父块回捞、版本核对、召回计数回读 |
| Milvus | 只读连接（`MILVUS_HOST/PORT`） | ① 路向量检索、实体链接 KNN |
| Neo4j | 只读账号 | ③-A 图遍历、图谱页 subgraph |

retrieval 侧 `sql/001_init.sql` 已内建：`CREATE ROLE retrieval_ro LOGIN` + `GRANT SELECT ON parent_document, child_chunk, scalar_extract, entity_registry` + `REVOKE INSERT/UPDATE/DELETE/TRUNCATE ... FROM retrieval_ro`。
qingxi 侧需开通同名账号并同样只授 `SELECT`（表名按 qingxi 实际：`chunk_relation` 而非 `child_chunk`，见 §8）。

---

## 2. PostgreSQL 表（只读消费）

qingxi 侧 `app/models/graphrag.py` 的表，retrieval 只读用到以下：

| 表 | 关键列（检索消费用） | 用途 |
|---|---|---|
| `parent_document` | `parent_id`(PK) `doc_id` `kb_id` `title` `level` `content` `content_version` `kg_entity_id` `kg_role` `kg_confidence` | QRY-06 父块全文回捞；`content_version` 与子块核对 |
| `chunk_relation` | `child_id`(PK=`{parent_id}#{序号}`) `parent_id`(FK) `chunk_index` `content` `enabled` | 子块留档、版本核对、兜底排查 |
| `source_file_registry` | `doc_id`(PK) `kb_id` `file_name` `content_hash` `doc_type` | 源文件溯源、内容指纹 |
| `recall_log` | `query` `doc_id` `child_id` `parent_id` `answer_text` `ts` | 召回日志（测评/被引口径） |
| `entity_registry` | `entity_id`(PK) `name` `entity_type` `aliases` `fingerprint` `status` `merged_into` | 实体消歧 canonical 账本（`merged_into` 指向 canonical） |
| `scalar_extract` | `parent_id`(PK) `kb_id` `rf_min/max` `pw_min_us/max` `prf_min_hz/max` `modulation` `snr_db` `doa_min/max` `scalar_status` | META-03 标量事实（④ 路/结构化过滤的数据源，本期只读不启用） |
| `extraction_dict` | `name` `en` `category(entity/relation)` `trigger_words` `head_types` `tail_types` `min_confidence` | 实体/关系字典（关系白名单 §6 的来源） |
| `ingest_quality` | `doc_id`(PK) `kg_status(ok/low/empty)` `entity_count` `relation_count` | 入库质检状态（图谱质量判断） |

ID 约定：`doc_id = documents.id(uuid)`，`parent_id = {doc_id}_{章节序号}`，`child_id = {parent_id}#{子块序号}`。

---

## 3. Milvus（向量检索 + 实体链接）

### 3.1 `ems_chunks`（子块向量库）

| 字段 | 类型 | 说明 |
|---|---|---|
| `child_id` | VARCHAR (PK) | 子块 ID |
| `content` | VARCHAR(65535) | 子块正文（PG 熔断时的行内兜底） |
| `parent_id` | VARCHAR | 父块 ID |
| `doc_id` | VARCHAR | 文档 ID |
| `kb_id` | VARCHAR | 知识库 ID（按库过滤） |
| `enabled` / `parent_enabled` | BOOL | 子块/父块可用开关 |
| `kg_entity_ids` | ARRAY(VARCHAR) | 子块关联实体 ID（③-A 回捞用） |
| **`content_vector`** | FLOAT_VECTOR(1024) | **dense 向量（COSINE）** |
| `rf_min/max` `pw_min_us/max` `prf_min_hz/max` `modulation` `snr_db` `doa_min/max` `scalar_status` | DOUBLE/VARCHAR | META-03 标量列（连续量 min/max 双列，缺失 NULL 禁占位） |

> ⚠️ **向量字段名是 `content_vector`**，不是 retrieval dev 脚本里的 `embedding`（见 §8）。

### 3.2 `ems_entities`（实体名向量库，消歧语义匹配）

| 字段 | 类型 | 说明 |
|---|---|---|
| `entity_id` | VARCHAR (PK) | 实体 ID |
| `entity_type` | VARCHAR | 实体类型 |
| `name` | VARCHAR | 规范化名 |
| `name_vector` | FLOAT_VECTOR(1024) | 实体名向量（COSINE，类型作用域内 KNN） |

---

## 4. Neo4j（图遍历 + 图谱页）

图模型：

```
(:Entity)                      —— 实体节点
(:Segment)                     —— 父块节点
(:Entity)-[:BINDS]->(:Segment) —— 实体↔父块绑定（kg_role=primary|supplemental, kg_confidence）
(:Entity)-[r:REL_TYPE]->(:Entity) —— 类型化关系边（11 类抽取 + 3 类派生）
```

| 元素 | 属性（qingxi 实际） |
|---|---|
| `:Entity` | `entity_id`(MERGE 键) `name` **`entity_type`** `aliases` `confidence` `source_parent_id` `properties` |
| `:Segment` | `parent_id`(MERGE 键) `doc_id` **`kb_id`** `title` … |
| `:BINDS` | `kg_role` `kg_confidence` |
| 类型关系边 `r` | `confidence` **`evidence`**（文本） `source_parent_id` `derived` |

> ⚠️ 三个与 retrieval 侧查询不一致的点（见 §8 对照表）：
> 1. 实体类型字段是 `entity_type`，retrieval `graph.py` 查询写的是 `n.type`；
> 2. `kb_id` 在 **Segment** 上、不在 Entity 上，retrieval 的 scope 过滤 `n.kb_id IN $scope` 查的是 Entity；
> 3. 关系边证据是 `evidence`（文本），retrieval 查询写 `evidence_child_id / span_start / span_end / parent_id`。

---

## 5. 模型接口（检索侧要连的两个模型）

### 5.1 BGE-M3 嵌入（dense + sparse 同源）

qingxi `app/graphrag/embeddings.py`：

```python
def embed_m3_full(self, texts: list[str]) -> dict:
    """BGE-M3 全量输出接口（dense + sparse 同源产出）——text_sparse 列的写入通道。
    返回：{"dense": [[float]], "sparse": [{token_id: weight}]}
    纪律：非 m3 模型调用即报错（sparse 无数据源，禁止静默降级为 dense-only 还当 hybrid 用）。
    """
    out = model.encode(texts, return_dense=True, return_sparse=True)
    return {"dense": out["dense"].tolist(),
            "sparse": [dict(w) for w in out["lexical_weights"]]}
```

- **dense 1024 维**，与 Milvus `content_vector` 同维同空间（BGE-M3 定案 D4）。
- **sparse** 为 lexical_weights 稀疏向量（`token_id -> weight`），对应 Milvus `text_sparse` 列，**IF-1 扩列后**才启用 hybrid。
- 检索侧当前 `models.embed.backend=hash`（占位）；接真模型后 `backend: bge_m3`，并重建 collection（dim 变更）。

### 5.2 LLM（vLLM OpenAI 兼容 function calling）

- QRY-02 L2 抽取 + RPT 流式生成共用：`GRAPHRAG_LLM_BASE_URL`（vLLM OpenAI 兼容端点）。
- 建议 14B+（function calling 需要），RPT 走 SSE 流式。
- retrieval 侧 `models.llm.backend: openai` + `endpoints.llm.url`（或 env `LLM_UPSTREAM`）。

### 5.3 Reranker

- `bge-reranker-v2-m3`，TEI/vLLM 端点：`RETRIEVAL_RERANK_SERVICE_URL`。

---

## 6. 关系白名单（与 qingxi 对齐）

qingxi 侧关系字典 = `seed.py` 的 11 类抽取关系 + `mesh/derived.py` 的 3 类派生关系。retrieval 侧已落为 `config/relations.yaml`（单一事实源，图谱接口 `relation_types` 白名单消费）：

- **抽取 11 类**：`SOURCE_OF / COMPOSED_OF / MERGED_INTO / MODEL_MATCH / EQUIPPED_ON / BELONGS_TO / HAS_STATE / RELATED_TARGET / PLATFORM_CANDIDATE / PRODUCES / WRITTEN_INTO`
- **派生 3 类**：`SAME_BAND / CO_TIME / CO_LOCATE`（带 `derived=true`，可整体重算）

qingxi 变更 `extraction_dict` 后必须同步 `retrieval/config/relations.yaml`，否则图谱接口关系过滤契约漂移。

---

## 7. env 命名契约

retrieval 侧**复用 qingxi 的连接类命名**（同一套 `POSTGRES_* / MILVUS_* / NEO4J_*` + `GRAPHRAG_EMBEDDING_MODEL` / `GRAPHRAG_LLM_*`），**检索特有参数用 `RETRIEVAL_` 前缀**；两服务各自持有独立 `.env`，互不共享敏感信息。

| 组 | qingxi 侧（入库） | retrieval 侧（在线，只读复用） |
|---|---|---|
| 连接 | `POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE` | 同名复用（`retrieval_ro` 账号） |
| | `MILVUS_HOST/PORT` + `MILVUS_COLLECTION` | 同名复用 + `MILVUS_CHUNK_COLLECTION`(默认 `ems_chunks`) `MILVUS_ENTITY_COLLECTION`(默认 `ems_entities`) |
| | `NEO4J_URI/USER/PASSWORD` | 同名复用 |
| 模型 | `GRAPHRAG_EMBEDDING_MODEL`(=`BAAI/bge-m3`) `GRAPHRAG_LLM_BASE_URL/API_KEY/MODEL` | 复用同名（同模型同空间） |
| 检索特有 | — | `RETRIEVAL_BACKEND` `RETRIEVAL_EMBED_URL` `RETRIEVAL_RERANK_SERVICE_URL` `RETRIEVAL_BM25_ENABLED` `RETRIEVAL_GRAPH_ROUTE_ENABLED` `RETRIEVAL_COMMUNITY_ENABLED` `RETRIEVAL_BUDGET_TOTAL_MS` `RETRIEVAL_TRACE_EXPORT/DIR` `RETRIEVAL_FAULT_INJECT` `RETRIEVAL_ALLOW_PLACEHOLDER` 等 |

---

## 8. 字段映射对照表（⚠️ real 联调前必读）

> 这是本次摸清的最大风险面：**连接信息已对齐，但「字段名」有多处不一致**。retrieval 自有 dev 三库（`sql/001_init.sql` + `scripts/init_milvus.py`）是**独立可跑的假三库**，与 qingxi 真实 schema 存在如下差异，real 联调必须逐条对齐，否则会静默查空 / 召回噪声。

| # | 层 | retrieval 侧（dev 脚本/查询） | qingxi 侧（真实 schema） | 影响 | 对齐动作 |
|---|---|---|---|---|---|
| 1 | Milvus | 向量字段 `embedding`（`init_milvus.py`、`vector_route.py` 的 `anns_field`） | 向量字段 **`content_vector`** | 检索按 `embedding` 搜会查空 | `vector_route.py` 的 `vector_field` 改为 `content_vector`（或 env 化） |
| 2 | Milvus | collection 曾为 `child_chunk`（已改 `ems_chunks`） | `ems_chunks` | — | ✅ 已对齐（config G3） |
| 3 | PG | 子块表 `child_chunk` | 子块表 **`chunk_relation`** | 父块回捞/版本核对接不上 | `parent_backfill.py` 的表名/查询对齐 `chunk_relation` |
| 4 | Neo4j | 实体类型 `n.type`（`graph.py` 查询） | 实体类型 **`entity_type`** | 节点 `type` 恒 NULL | 查询改 `n.entity_type` |
| 5 | Neo4j | scope 过滤 `n.kb_id IN $scope`（在 Entity 上） | `kb_id` 在 **Segment** 上、Entity 无 | scope 过滤失效（实体级） | 改走 `BINDS → Segment.kb_id`，或 Entity 冗余 `kb_id` |
| 6 | Neo4j | 关系证据 `evidence_child_id / span_start / span_end / parent_id` | 关系边 `evidence`(文本) + `source_parent_id` | 边证据锚点取不到 | 对齐证据字段（或 qingxi 侧补列） |
| 7 | 模型 | dense dim 1024（BGE-M3） | dense dim 1024 | — | ✅ 一致（同模型同空间） |
| 8 | 稀疏 | `text_sparse`（IF-1 扩列后） | `text_sparse`（`embed_m3_full` 写入） | — | ✅ 同名字段，待 IF-1 后启用 |

> 说明：#2/#7/#8 已一致；#1/#3/#4/#5/#6 是 real 联调的**待对齐项**。本项目原则「禁止静默降级」——字段对不上必须显式报错留痕，不能静默查空后当正常。

---

## 9. qingxi 已暴露的 HTTP API（供后续对接参考）

qingxi 主服务（文档解析清洗服务，`qingxi/app/main.py`）已暴露的接口面（详见 `qingxi/docs/api.md`）：

- **解析入库**：`POST /parse`（+ `image/audio/video/html/iq` 变体）、`GET /health`
- **文档**：`GET /documents/{doc_id}/download`、`GET /knowledge-bases`
- **图谱**：`/graph/dict`（字典）、`/graph/trace/*`（溯源）、`/graph/segments`、`/graph/route`、`/graph/reconcile`（对账）、`/graph/metrics`、`/graph/quality`（质检）、`/graph/mesh/link-orphans`、`/graph/resolve`（消歧）
- `graph/query` 为已废弃的测试代码，不要对接。

> retrieval 侧**不调用**这些接口（检索走三库直连，不经过 qingxi HTTP）。列在此处仅为定位 qingxi 侧的入库/对账能力。

---

## 10. real 联调 checklist（后续执行）

1. 开通三库只读账号 `retrieval_ro`，验收「写操作全拒」。
2. 按 §8 对照表对齐 #1/#3/#4/#5/#6 五处字段名。
3. 接真模型：`models.embed.backend: bge_m3` + 重建 Milvus collection（dim 一致则无需重建，仅改 `vector_field`）+ 重跑 `check_embed_space.py --expect real`。
4. 把 `.env` 的 `RETRIEVAL_BACKEND` 改 `real`，重启服务，启动日志核对生效值（口令脱敏）。
5. `python scripts/seed_real.py` 灌真三库 → `python scripts/smoke.py` 全绿 → `pytest -q` 全绿。
