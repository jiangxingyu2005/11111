# RAG 问答接口文档

面向前端 RAG 问答 / 可解释报告页。无认证。Base URL 以实际部署为准（示例 `:8001`，本机 Docker 常见 `:8003`）。

一次提问走同一条链路：意图分流 → 实体匹配（含频率/射程等属性回推）→ 双路召回 → 重排 →（关系问）图扩展 → **规则引擎求值** → LLM 生成。  
规则表全文不进模型，只把**已触发**的短 hint 写入提示词。

---

## 1. 一次返回

```http
POST /graph/query
Content-Type: application/json
```

### 请求

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 自然语言问题 |
| `kb_id` | UUID | 否 | 单个知识库，兼容旧前端 |
| `kb_ids` | UUID[] | 否 | 多库勾选，与 `kb_id` 合并去重。都不传 = 全库 |
| `iq_doc_id` | UUID | 否 | 待测 IQ 文档（先 `/parse-iq` 拿到的 `doc_id`）。传了才查 `ems_iq` 出相似信号表；不传则表为空 |

```json
{
  "query": "2.4GHz 有哪些装备",
  "kb_ids": ["9d8cc649-de38-4c34-a5d5-e5333f17d666"]
}
```

知识库列表：`GET /knowledge-bases`（用返回的 `id` 勾选）。多库细则见《多知识库勾选检索接口文档》。

### 响应（报告页用这些字段）

| 字段 | 前端用途 |
|---|---|
| `answer` | 推理结论 |
| `path` | 路径条 ①–⑦（`step` `name`，部分步带计数） |
| `graph_facts` | 知识关联路径（`实体 --[关系]--> 实体`），可画简单节点 |
| `rule_hits` | 规则命中列表 |
| `confidence` | 置信度条（见下） |
| `hits` | 引用子块（`content` `_score`） |
| `matched_entities` | 命中实体 id |
| `expanded_entities` | 图扩展实体 |
| `kb_ids` | 实际检索范围，`null` 表示全库 |
| `iq_intent` | `iq_compare` / `none`：本次是否走了 IQ 对比 |
| `iq_status` | `ok` / `empty_library` / `doc_not_found` / `failed: …`；非 ok 时表为空，问答照常 |
| `iq_query` | 待测信号概要：文件名、调制、主频 Hz、SNR |
| `similar_signals` | 相似信号表（最多 4 行，按相似度降序）；不带 `iq_doc_id` 时为 `[]` |
| `iq_charts` | 待测信号的 `time_domain` / `spectrogram` / `psd` 图数据 |
| `intent` | `retrieval` / `graph` |
| `cond` | NL2Query 结构化条件 |

`path` 七步：用户问题 → 向量检索 → 实体匹配 → 知识库条目/图扩展 → 规则库匹配 → 证据融合 → 结构化报告。

### `rule_hits[]`

| 字段 | 说明 |
|---|---|
| `id` `name` | 规则 ID、名称 |
| `triggered` | 是否命中 |
| `label` `severity` | 动作标记 / 严重度 |
| `hint` | 仅触发时有，已注入 Prompt |
| `evidence` | 触发用的正文摘录 |

### `confidence`

| 字段 | 说明 |
|---|---|
| `retrieval.score` | Top-k 子块 `_score` 平均，0~1 |
| `rule.score` / `hit` / `total` | 命中数÷启用规则数；无规则则为 `null` |
| `signal_agree.score` / `.fields` | 待测与 Top-1 的参数一致比例，`fields` 是实际参与比较的项（`frequency`、`modulation`）。调制为 Unknown 不参与。无 IQ 时为 `null` |
| `overall` | 基础权重 规则 0.4 / 检索 0.6 / 一致度 0.2，缺哪项就去掉后重归一 |
| `weights` | 实际权重 |

右上角综合环绑 `overall`，不要写死百分数。

---

## 2. 流式 SSE

```http
POST /graph/query/stream
Content-Type: application/json
```

请求 body 与 §1 相同。

| 事件 | 用途 |
|---|---|
| `meta` | `query` `intent` `cond` |
| `progress` `entity_match` | 匹配到的实体名 |
| `progress` `recall` | A/B 路条数 |
| `progress` `rerank` | 重排条数 |
| `progress` `graph_expand` | 扩展数、`facts` |
| `progress` `rules` | `triggered` / `total` |
| `progress` `iq_compare` | 仅带 `iq_doc_id` 时出现：`status` `similar` |
| `results` | `hits` `graph_facts` `rule_hits` `confidence` `similar_signals` `iq_charts` `iq_query` `kb_ids` |
| `token` | 答案增量 |
| `done` / `error` | 结束 / 失败 |

---

## 3. 规则怎么配（命中列表的数据源）

```http
GET /rules
POST /rules
PUT /rules/{id}
DELETE /rules/{id}
```

条件须能在检索命中上计算（如 `content` contains、`rerank_score`）。没有数据的条件（距离 50km 等）求值失败即不触发。细则见 `docs/api/10-rules.md`。

---

## 4. 报告页建议绑定

| 页面区块 | 字段 | 没有数据时 |
|---|---|---|
| 标题/时间/库 | 前端本地 + `kb_ids` | — |
| 路径条 | `path` 或 SSE `progress` | — |
| 推理结论 | `answer` | — |
| 知识路径图 | `graph_facts` | 空数组则不画 |
| 规则列表 | `rule_hits` | 空则「未评估」 |
| 检索相关度 | `confidence.retrieval.score` | 无 hits 则灰 |
| 规则命中率 | `confidence.rule` | `null` 则灰 |
| 综合环 | `confidence.overall` | `null` 则灰 |
| 引用 | `hits` | — |
| 相似信号表 | `similar_signals` | 空数组则隐藏表格，不要用演示假数据 |
| 相似信号一致度 | `confidence.signal_agree` | `null` 则灰 |
| 时域 / 时频 / 频谱图 | `iq_charts` | `null` 则隐藏 |

### `similar_signals[]` 每行

| 字段 | 对应表头 | 说明 |
|---|---|---|
| `id` | 样例ID | `ems_iq` 里的 IQ 文档 id |
| `name` / `category` | 名称 / 类型 | 入库 `/iq/ingest` 时打的类别 |
| `frequency_hz` | 频率 | Hz，前端换算 MHz |
| `modulation` | 调制 | 分类器结果，可能为 `Unknown` |
| `snr_db` | — | 信噪比 |
| `pri_hz` / `pulse_width_us` | PRI / 脉宽 | **当前恒 `null`**，显示「—」 |
| `source` | 来源 | 固定 `ems_iq` |
| `score` | 相似度 | 0~1 余弦，乘 100 显示 |

流程：`/parse-iq` 上传待测 H5 拿 `doc_id` → 提问时带 `iq_doc_id`。库样本须先经 `/iq/ingest`（带类别）进 `ems_iq`。

---

## 5. 错误码

| HTTP | 含义 |
|---|---|
| 422 | 缺 `query` 或 UUID 非法 |
| 502 | Neo4j / Milvus 不可用 |
