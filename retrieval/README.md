# retrieval —— 电磁信号智能检索分析系统 · 在线检索服务

> 主施工依据：《在线检索框架开发文档_V3.1》（文档库）｜工程包说明：《在线检索框架_可运行工程包_V1.0》
> 架构参考：《在线检索开发文档_V2.0.1》（存档）｜仓库定案：《检索服务独立仓库与qingxi接入方案_V1.0》（D1）

---

## 一、5 分钟跑起来（fake 通道：不需要 docker / 不需要 GPU）

```bash
pip install -r requirements.txt
python scripts/gen_seed.py                    # 生成种子数据 data/seed/*.json
python services/model_stub.py --port 8001     # 终端 A：模型 stub（embed/rerank/LLM 流式）
python -m uvicorn app.main:app --port 8100    # 终端 B：retrieval 服务
python scripts/smoke.py                       # 终端 C：一键冒烟（含 5 个降级注入场景）
curl -N -X POST http://127.0.0.1:8100/api/retrieval \
     -H 'Content-Type: application/json' \
     -d '{"query":"X波段 线性调频 雷达信号，载频 9GHz","scope":["kb_public"]}'
```

Windows 无 `make` 时按上面逐条执行；有 `make` 时 `make init && make smoke` 等价。

**这条链路是"真跑"，不是花架子**：真实解析 → 真实向量检索（130 子块）→ 真实 RRF 融合 → 真实重排 →
真实父块回捞 → 真实流式生成 → 真实图谱二次取数，且响应体逐字段过 `api/*.schema.json` 校验。

---

## 二、两条后端通道

| 通道 | 开关 | 依赖 | 用途 |
|---|---|---|---|
| `fake` | `RETRIEVAL_BACKEND=fake` | 无（内存三库 + 哈希嵌入） | 开发/CI/契约验证，零外部依赖 |
| `real` | `RETRIEVAL_BACKEND=real` | PG + Milvus + Neo4j + 模型服务 | 生产与验收 |

两条通道**共用同一套接口、同一份配置、同一套降级语义**，只有最底层的数据访问实现不同
（`app/recall/*.py` 里的 `_backend` 分支）。禁止出现第二套业务流程。

切真环境：
```bash
docker compose up -d
python scripts/init_milvus.py                 # 建 ems_chunks / ems_entities
docker compose exec neo4j cypher-shell -u neo4j -p dev_only -f /import/001_constraints.cypher
python scripts/seed_real.py
# 然后把 .env 的 RETRIEVAL_BACKEND 改成 real，重启服务
```

---

## 三、目录结构

```
retrieval/
├── app/
│   ├── main.py                 FastAPI 入口（/healthz + 两个路由）
│   ├── config.py               .env + config/*.yaml 双源；缺必需变量拒绝启动
│   ├── api/
│   │   ├── retrieval.py        POST /api/retrieval（SSE 主链 + 预算砍路）
│   │   └── graph.py            GET /api/graph/subgraph（独立图谱页）
│   ├── understand/query_parser.py   QRY-02：L1 正则 + L3 校验器（真实现）
│   ├── recall/
│   │   ├── vector_route.py     ① 路向量召回（dense / hybrid 双模）
│   │   ├── bm25_route.py       ② 路外置 BM25（默认关）
│   │   └── graph_route.py      ③-A 图遍历（实体链接 + expand + BINDS 回捞）
│   ├── fuse/
│   │   ├── rrf.py              外层 RRF（1 source 1 vote，完整实现）
│   │   ├── rerank.py           精排（真服务 / 本地打分）
│   │   └── parent_backfill.py  QRY-06 父块回捞（small-to-big）
│   ├── generate/report.py      RPT 流式生成（OpenAI 兼容 SSE）
│   ├── degrade/                降级码字典 + 编排器 + 预算看门狗
│   ├── trace/spans.py          TRACE 埋点 + trace_id 绑定（TTL 30min）
│   ├── store/
│   │   ├── hash_embed.py       dev 通道确定性哈希嵌入（无 GPU 可跑）
│   │   └── fake.py             内存三库 + 故障注入
│   └── graph/expand.py         focus 实体提取
├── config/*.yaml               算法参数唯一事实源（dev_overrides 只放宽阈值）
├── api/*.schema.json           SSE 与图谱接口的响应契约（jsonschema 校验）
├── scripts/
│   ├── gen_seed.py             生成种子数据（合成，只验逻辑不验召回率）
│   ├── smoke.py                一键端到端冒烟 + 降级注入
│   ├── init_milvus.py          建 Milvus collection
│   └── seed_real.py            灌真三库
├── services/model_stub.py      embed / rerank / LLM 流式 三合一 stub（8001）
├── sql/001_init.sql            PG 表 + 只读账号 retrieval_ro（IF-9）
├── neo4j/001_constraints.cypher
└── docker-compose.yml          PG + Milvus(etcd/minio) + Neo4j
```

---

## 四、模块状态（真跑通 / stub / 待办）

| 模块 | 状态 | 说明 |
|---|---|---|
| QRY-02 L1 正则 + L3 校验 | ✅ 真实现 | 配置驱动，波段/单位换算/量纲校验齐全 |
| QRY-02 L2 LLM 抽取 | ⚠️ 待接 | L1 未命中时记 `nl2query_llm_timeout` 继续（明确降级） |
| ① 路 dense 检索 | ✅ 真实现 | fake 真跑；real 走 Milvus |
| ① 路 sparse / hybrid | 🔧 接口就绪 | 开关 `milvus.sparse.enabled`，待 IF-1 扩 `text_sparse` 列后开 |
| ② 路 BM25 | ⏸ 关闭 | 消融 D2 裁决前不开 |
| ③-A 图遍历 | ✅ 真实现 | 实体链接 KNN + 1~2 跳 + BINDS 回捞 |
| ③-B 社区摘要 | ⏸ deferred | IF-4，降级码位已预留 |
| ④ 路标量硬过滤 | ⏸ 不启用 | 附录 A 预留，过渡期一律软语义 |
| 外层 RRF | ✅ 完整实现 | 1 source 1 vote |
| Rerank | ✅ real 真服务 / fake 本地打分 | 超时直出 RRF 序 |
| QRY-06 父块回捞 | ✅ 真实现 | PG 熔断用行内 content 兜底 |
| RPT 流式生成 | ✅ 真流式 | 未配 LLM → `generation_unavailable`，不出正文（不编造） |
| 图谱页 subgraph | ✅ 真实现 | trace_id 绑定 + scope 二次校验 + 每条边带 evidence |
| 降级编排 + 预算看门狗 | ✅ 真砍路 | >3.5s 砍重排 / >4.5s 砍图路，每砍必留痕 |
| IF-2 证据 span | ✅ fake 有真值 | real 依赖离线切分产出，缺失时 `evidence_span_missing` |
| IF-5 图片资产 assets | ⏸ 空表 | 待离线入库 |

---

## 五、模型接入（E1/E2/E3 的配置入口）

**接入自有模型不用改代码**，只改两处（或用环境变量覆盖，env 优先于 yaml）：

```yaml
models:                            # 用哪种实现去连
  embed:  {id: BAAI/bge-m3, dim: 1024, backend: hash}      # hash|bge_m3|openai|generic
  rerank: {id: bge-reranker-v2-m3, backend: hash}          # hash|bge|generic
  llm:    {id: qwen2.5-14b-instruct, backend: echo}        # echo|openai|hf|generic

endpoints:                         # 连哪里、怎么取字段
  embed:  {url: "", preset: "", request: {}, response: {}}
  rerank: {url: "", preset: "", request: {}, response: {}}
  llm:    {url: "", preset: ""}
```

三种协议：`local`（进程内）/ `openai`（OpenAI 兼容 HTTP）/ `generic`（任意 HTTP + 字段路径映射）。
`presets` 里已有 TEI、Ollama、SiliconFlow 等现成映射，填个 `preset` 名字即可。

**对外接口形状恒定**（换任何模型都不变）：`/embed`、`/rerank`、`/v1/chat/completions`；
另有 `/v1/model-info`（自省，脱敏）和 `/v1/probe`（不重启自检）。

详见 **docs/接入自有模型.md**。

**接入真模型的三件事，缺一不可**（E1 为例）：

1. `models.embed.backend: hash -> bge_m3`（或 env `EMBED_BACKEND=bge_m3`）
2. 重建 Milvus collection —— dim 变了旧向量全部失效
3. 重跑 seed：`python scripts/gen_seed.py --embed service`，再
   `python scripts/check_embed_space.py --expect real` 验收

**启动校验（防静默漂移）**：`real` 通道跑 hash 嵌入 = **拒绝启动**
（宁可停机，不许"召回全是噪声还报绿"）。staging 要放行须显式
`RETRIEVAL_ALLOW_PLACEHOLDER=1`，仍会打 ERROR。
另有 4 条一致性校验：dim 必须与模型 id 匹配、sparse 两处开关必须联动、
`reranker.model` 必须与 `models.rerank.id` 一致、backend 必须在白名单内。

---

## 六、验收命令

```bash
python scripts/smoke.py              # 端到端 + 5 个降级注入，退出码 0 = 全绿
python -m pytest -q                  # 契约/降级/配置/模型契约/Provider/流式透传 测试

# 接入自有模型自检（拿到新地址先验再配，不重启）
python scripts/check_provider.py
python scripts/check_provider.py --kind embed --url http://host:8080/v1 --preset openai_embed

# E1 哨兵：证明"当前是真模型还是哈希占位"
python scripts/check_embed_space.py                    # dev：只记基线
python scripts/check_embed_space.py --expect real      # 真模型：不过线退出码非 0

# E2 链路：证明"SSE 真流式 + 证据 [n] 拼接"是通的
python scripts/check_llm_chain.py --backend echo       # 无模型自检 harness
python scripts/check_llm_chain.py --backend openai \
    --upstream http://127.0.0.1:11434/v1 --model qwen2.5:3b
```

冒烟覆盖：SSE 事件序列 · 聚合响应过 schema · 每条命中有 child_id + 真实 span + 父块正文 ·
graph_ref focus 实体 · subgraph 契约与边证据 · scope 必填 422 · scope 隔离 · 五种故障注入的"行为 + 留痕"双验。

**E2 CPU 小模型跑法（本机无 GPU 时的逻辑级替代）**：

```bash
# 方式 A：外部 OpenAI 兼容服务（llama.cpp server / ollama）
ollama serve && ollama pull qwen2.5:3b
python scripts/check_llm_chain.py --backend openai \
    --upstream http://127.0.0.1:11434/v1 --model qwen2.5:3b

# 方式 B：进程内 transformers（无需额外服务，本机实测路径）
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers
# 模型走 hf-mirror 手动抓（hub 库在本机会下到 0 字节）：
#   curl -sL -o <dir>/<file> https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/<file>
python scripts/check_llm_chain.py --backend hf \
    --hf-model C:/tmp/models/Qwen2.5-0.5B-Instruct --max-new-tokens 128
```

实测（Qwen2.5-0.5B / CPU 20 线程）：`chunks=62` 真流式、输出含证据编号 `[1]`、
`degraded=[]`、且模型按 system prompt 主动输出"依据不足说明"。

注意：CPU 小模型**只证明链路通，不证明报告可信**。报告质量仍要人工抽检，
且必须显式标注为逻辑级替代——**首 token 46s（含冷加载）远超 5.2s 预算，
生产必须用 GPU vLLM**。

---

## 七、红线（改代码前先读）

1. **scope 必填无默认值**（R4），检索与图谱两处各校验一次，不因"检索时已过滤"而放行。
2. **禁止静默降级**：任何降级必须进 `degraded[]`，且 `step` 必须落在契约枚举内（`STEP_MAP` 是唯一映射源）。
3. **降级只做整路剥离**，不得改语义参数：RRF k、τ、切分参数、`CONFIDENCE_GATE` 在任何层级都不变。
4. **未标注 ≠ 不满足**：元数据缺失不得硬过滤排除，覆盖率不足的字段降级为软提权或关闭。
5. **禁止编造**：LLM 不可用就是不出正文；无证据锚点就是 `span=null`（禁用 `[0,0]` 占位）。
6. **配置不许绑定模拟数据**：算法参数在 `config/*.yaml`，`dev_overrides` 只允许放宽阈值/超时，
   禁止覆盖 RRF k / top_k 等语义参数（有测试守着）。
7. **缺必需变量拒绝启动**，启动日志打印生效值（口令脱敏）。
