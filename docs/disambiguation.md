# 实体消歧模块总结（归一对齐）

> **现状以 `app/graphrag/resolve.py` 为准**：向量不再按 sim_high 自动合并；属性冲突否决；LLM 双向判同才合并，否则记疑似。下文「三阶段 / ≥sim_high 自动并」是早期设计，已过时。
>
> 本文是消歧模块的**设计复盘 + 问题解决清单**。接口/配置/端点的权威文档见 `docs/api.md` §11。

## 1. 模块定位

解决「同一实体、不同写法」的**跨文档归并**。融合前消歧是「文档内 + 内存态」（`canonical_entity_id`
同名哈希去重、`EntityRegistry` 用后即弃），本模块把消歧落成**持久化、可对账、顺序无关**的生产实现。

「同一实体不同写法」的三种典型形态，对应三阶段匹配：

| 形态 | 例 | 匹配器 |
|---|---|---|
| 同义异名（有别名线索） | 雷达站 / 别名「雷达」 | ① 确定性 |
| 同义异名（无语义线索，仅语义相近） | 雷达 ↔ 对空搜索雷达 | ② 向量 |
| 语义相近但歧义（阈值灰区） | 需裁决的模糊对 | ③ LLM |

## 2. 架构

### 2.1 三阶段匹配管道（命中即返，类型作用域）

```
resolve_deterministic(name, type, aliases, props)   # ① 名称精确 / 别名 JSONB / 属性指纹,按最早 created_at
    → resolve_vector(name, type)                    # ② 实体名向量 ems_entities 余弦,≥sim_high 自动并
        → judge_same_entity(...)                    # ③ 灰区(≥sim_low)交 LLM 裁决,默认关
```

① 命中即返,不走到 ②③;②③ 是「模糊匹配层」,仅在 ① 未命中时激活。

### 2.2 数据层

- **`entity_registry`(PG)**：全局 canonical 实体注册表。列 `entity_id / name / entity_type / aliases(JSONB) /
  fingerprint / properties / status(active|merged) / merged_into / created_at`;`status=merged` 即对账账本。
- **`ems_entities`(Milvus)**：实体名向量库。字段 `entity_id(PK) / entity_type / name / name_vector`(COSINE);
  只存「可向量归并」的名词性实体(见 §3 问题 4 的方案 ①)。
- **`extraction_dict.semantic_mergeable`(PG)**：标记该实体类型是否允许向量语义归并——字典是唯一数据源,
  改字典即时生效,不硬编码。

### 2.3 两条落库路径

1. **入库在线对齐**(自动)：`align`(对齐到 canonical + 关系 head/tail 重映射) + `persist`(写注册表 + 实体向量),
   插在抽取后、图增强前(ingest step 6.5/6.6,分段编辑路径同步)。
2. **全量批量对账**(手动)：`POST /graph/resolve` 对历史重复实体批量归并并四层回填旧数据。

### 2.4 关键设计决策

- **canonical 发号确定性**：`canonical_entity_id(name, type) = sha256(normalize_name(name)|type)[:16]`,同名同类型
  天然同号,重复入库不翻倍。
- **canonical 选择稳定**：DSU(并查集)求连通分量,组内取 `created_at` 最早(并列取 `entity_id` 字典序),顺序无关、结果稳定。
- **四层回填一致**：一次对账同步四处——PG 注册表(absorb aliases/fingerprint/properties)、Neo4j(移 BINDS/关系、
  删旧节点)、PG parent_document(kg_entity_id 逗号串 token 重映射)、Milvus(ems_chunks.kg_entity_ids 数组重映射 +
  ems_entities 删冗余向量)。
- **归一化口径**：`normalize_name` = 全角→半角、去首尾空白、折叠空白、ASCII 小写,贯穿名称/别名/指纹/向量一致。

## 3. 问题与解决策略

### 3.1 向量存储层(Milvus)

| # | 问题 | 解决策略 |
|---|---|---|
| 1 | **pymilvus 3.0.2 `upsert` 不自动 flush**,新写实体向量 `search_entities` 看不到(NUM_ENTITIES=0) | `upsert_entity_vector` 末尾显式 `coll.flush()` |
| 2 | **`delete` 不 flush**,已删向量仍被命中,破坏对账幂等(二次 resolve 重复合并) | `delete_entity_vector` 末尾显式 `coll.flush()` |

### 3.2 语义层(embedding + 阈值)

| # | 问题 | 解决策略 |
|---|---|---|
| 3 | **阈值量级错配**:初版 `sim_high=0.95 / sim_low=0.85` 是按合成向量(雷达站↔雷达=0.999)设的,真实 BGE 的语义相似度量级只有 0.6~0.8,原阈值下「雷达↔雷达站」=0.76 连灰区下界都够不到,语义层**完全失效** | 用 `bge-large-zh-v1.5` 在 `text/` 真实数据实测分布,校准为 **`0.62 / 0.55`**;换模型需重校准(见 `docs/api.md` §11.4) |
| 4 | **窄带重叠**:向量余弦衡量「语义相关」≠「实体同一」,`雷达站↔对空搜索雷达`≈0.63 与 `2.4GHz↔850MHz`≈0.60 重叠,单一阈值切不开 | **方案①(已落地)**:参数值/枚举类实体不进向量层——`extraction_dict.semantic_mergeable` 标记,频率/调制方式/信号/地理位置标 `false`,只走确定性;装备/辐射源/平台/任务标 `true` 才进 `ems_entities`。`resolve_vector`/`persist`/`global_resolve` 三处按 `load_mergeable_types()` 过滤。`2.4GHz↔850MHz` 直接从向量比较退场,gap 拉开。后续可选增强:③交叉编码器精排 或 ④灰区 LLM(`_LLM_GATE=true`,已备好) |
| 5 | **FlagEmbedding 1.4.2 加载本地模型仍访问 HF hub** 查 chat template(`AutoTokenizer.from_pretrained`→`list_repo_templates`),离线环境网络超时 | `_local_bge_path()` 用 `snapshot_download(local_files_only=True)` 解析本地 snapshot 路径,彻底绕开网络 |
| 6 | **`HF_HUB_OFFLINE=1` 运行时 setdefault 无效**:`huggingface_hub.constants` 在 import 时已缓存该值,服务里 ASR/BLIP 先 import 了 transformers,再改环境变量不生效 | 不依赖环境变量,改走问题 5 的「本地路径解析」方案 |

### 3.3 匹配正确性

| # | 问题 | 解决策略 |
|---|---|---|
| 7 | **类型作用域**:同名异类型会误并(如「频率 2.4GHz」与「辐射源 freq_band=2.4GHz」) | 确定性(名称/别名/指纹)与向量检索都加 `entity_type` 条件,类型隔离 |
| 8 | **mock 向量无语义**:3-gram 哈希下异名不相似,语义层形同虚设 | 语义层只在 provider=bge/openai 才真正激活;mock 下仅确定性兜底,行为与融合前一致(回归安全) |

### 3.4 一致性 / 幂等

| # | 问题 | 解决策略 |
|---|---|---|
| 9 | **四层数据不一致**:实体合并后 PG/Neo4j/Milvus 三库、PG parent 与 Milvus chunks 两处 id 都要同步 | `global_resolve` 统一四层回填(§2.4),任何一层失败不影响其余层完成 |
| 10 | **幂等**:重复对账不能翻倍/漂移 | DSU canonical 确定性(最早 created_at + 字典序 tie-break)+ delete flush(问题 2);二次调用 `merged=[]` |

### 3.5 迁移 / 兼容

| # | 问题 | 解决策略 |
|---|---|---|
| 11 | **`create_all` 不给已存在的表加列**:旧库 `extraction_dict` 缺 `semantic_mergeable` 列 | `seed_extraction_dict` 里 `ALTER TABLE ADD COLUMN IF NOT EXISTS` 幂等补列 + 首次回填种子标注(只碰种子类型,不覆盖用户其他改动) |
| 12 | **接口向后兼容**:`resolve_vector` 新增类型过滤,不能破坏旧调用 | `mergeable` 参数默认 `None`=不限制;`align/persist/global_resolve` 签名不变;字典查询失败降级为「不限制」,消歧链路不因字典故障崩 |

## 4. 验证与评分

| 验证 | 结果 |
|---|---|
| `verify_resolve`(真实 PG+Milvus+Neo4j,合成向量) | **16/16**(向量层 5 项 + 全量对账四层回填 11 项) |
| `verify_text_bge`(真实 BGE 在 `text/` 数据) | 确定性层 F1=**1.00**、向量语义层 F1=**1.00**、无别名端到端归并 **2/2**、类型作用域零误并 |
| DB 端到端标注 | 8 个种子类型 semantic_mergeable 全部正确(名词性 True / 参数值 False) |

## 5. 配置与扩展点

- **配置全在 `.env`**(§`docs/api.md` 配置表):`GRAPHRAG_RESOLVE_ENABLED/_FINGERPRINT/_VECTOR_ENABLED/_SIM_HIGH/_SIM_LOW/_LLM_GATE/_TOPK`、`GRAPHRAG_ENTITY_COLLECTION`、`GRAPHRAG_EMBEDDING_*`。
- **换 embedding 模型**(mock/openai/bge-m3):改 `.env` 的 `GRAPHRAG_EMBEDDING_PROVIDER/_MODEL`,`embed`/`embed_one` 接口不变;但**需重新校准 `_SIM_HIGH/_SIM_LOW`**(不同模型相似度分布不同)。
- **调整某类型是否参与向量归并**:`/graph/dict` CRUD 改 `semantic_mergeable`(schema 已暴露该字段),改后即时生效。
- **灰区 LLM 兜底**:`_LLM_GATE=true` + `provider=openai` 时,0.55~0.62 的模糊对交 `judge_same_entity` 裁决(默认关,确定性优先)。
- **匹配器替换**:只动 `resolve_vector`/`resolve_deterministic` + `.env`,存储/合并/对账四层不动。
