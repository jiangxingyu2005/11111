# Nacos 配置中心（qingxi + retrieval 全量迁入 + 热更新）

## 1. 概述

配置散落在根 `.env`、`retrieval/.env`、`retrieval/config/*.yaml`，改动需重启。引入 **Nacos** 作为唯一
配置中心：`.env` 与 `config/*.yaml` 全量迁入 Nacos，本地只留 `NACOS_*` bootstrap 键，并支持**热更新**
（listener 长轮询监听到变更即时生效，无需重启）。

- 客户端：自研 `common/nacos_client.py`（`httpx`，零新依赖，不引 `nacos-sdk-python`）。
- Nacos 服务：独立容器（老地方 `C:\Users\A\Desktop\docker\nacos\docker-compose.yml`，
  `nacos/nacos-server:v2.3.2`，standalone，`NACOS_AUTH_ENABLE=false`），端口 8848（HTTP 配置 API）
  / 9848（gRPC）。
- 命名空间：public（空 tenant）；group：`DEFAULT_GROUP`。
- **listener 契约**（踩过坑）：`Listening-Configs` 是 **query/form 参数**（不是 header），只有
  `Long-Pulling-Timeout` 走 header；变更响应体是 **URL 编码** 的 `dataId\x02group(\x01…)*`
  （`%02`=`\x02`、`%01`=`\x01`），客户端须先 `unquote` 再按 `\x02`/`\x01` 拆。Nacos 长轮询有
  ~10s 最短下限（短于 10s 的 `Long-Pulling-Timeout` 会被抬到 ~10s）。

## 2. dataId 组织

| dataId | type | 内容 | 消费方 |
|---|---|---|---|
| `qingxi.env` | properties | 根 `.env`（剥 `NACOS_*`） | qingxi（8001） |
| `retrieval.env` | properties | `retrieval/.env`（剥 `NACOS_*`） | retrieval（8100）+ model_stub（8010） |
| `retrieval.yaml` | yaml | `retrieval/config/retrieval.yaml` | retrieval + model_stub |
| `query_parser.yaml` | yaml | `retrieval/config/query_parser.yaml` | retrieval |
| `relations.yaml` | yaml | `retrieval/config/relations.yaml` | retrieval |

5 个 spec-only yaml（`structured_filter`/`scalar_extract`/`scalar_adapter`/`adapter_patch_generated`/
`community_summary`）代码不读，**留在本地不迁**。

## 3. bootstrap 键（本地 .env 只留这一段）

| 键 | 默认 | 说明 |
|---|---|---|
| `NACOS_ENABLED` | `false` | 总开关；false = 完全等同旧行为（本地 .env/yaml） |
| `NACOS_SERVER_ADDR` | `http://127.0.0.1:8848` | Nacos 地址 |
| `NACOS_NAMESPACE` | 空 | public；未来隔离用 namespace **ID** |
| `NACOS_GROUP` | `DEFAULT_GROUP` | group |
| `NACOS_USERNAME` / `NACOS_PASSWORD` | 空 | 预留（本期 auth 关闭） |
| `NACOS_DATA_IDS` | 见各 `.env` | 逗号分隔的要拉取的 dataId |

## 4. 各服务行为

| 服务 | bootstrap | 热更新 | 说明 |
|---|---|---|---|
| qingxi（8001） | 启动期拉 `qingxi.env` → `os.environ`（在 `settings = Settings()` 前） | ✅ watcher | 可调标量即时生效；连接/模型身份字段打 `WARNING restart required` |
| retrieval（8100） | `get_settings()`/`load_yaml()` 首行幂等引导 | ✅ watcher | env/yaml 变更校验通过后 `reset_caches()` 生效，失败回滚 |
| model_stub（8010） | 启动期拉 `retrieval.yaml`（+可选 `retrieval.env`） | ❌ 无 watcher | 模型身份/端点变更需重启 model_stub（代价低） |

**线程模型**：每服务一条 daemon 线程跑 `NacosConfigClient.watch()`；`listen_once` 阻塞最长 ~30s 只在
子线程里，不阻塞 asyncio 事件循环。变更检测：listen 给「有变更」信号 → 重拉全量比对 md5 → 变了才回调。

## 5. 热更新 vs 重启（qingxi）

- **热更新**（请求期读的可调标量）：`SCORE_*`、`CLEAN_*`、`VIDEO_*`、`MINERU_*`、`GRAPHRAG_LLM_*`、
  `GRAPHRAG_QUERY_*`、`GRAPHRAG_ROUTE_*`、`GRAPHRAG_CONFIDENCE_*`、`GRAPHRAG_TAU_*`、`GRAPHRAG_CHUNK_*`、
  `GRAPHRAG_RESOLVE_*`、`GRAPHRAG_QUALITY_*`、`GRAPHRAG_ENABLED`、`HTML_NOISE_TAGS`、`MAX_UPLOAD_MB`、
  `IQ_SEGMENT_LEN`、`IQ_SNR_THRESHOLD_DB`、`IQ_TARGET_FS`。
- **重启才生效**（喂给 import 期 engine / 懒加载模型缓存 / 进程全局 pymilvus 连接）：
  `POSTGRES_*`、`MILVUS_*`、`NEO4J_*`、`MINIO_*`、`KAFKA_*`、`ASR_*`、`BLIP_*`、`TRANSLATOR_*`、
  `GRAPHRAG_EMBEDDING_*`、`UPLOAD_DIR`、`OUTPUT_DIR`、`IQ_*_MODEL_*`、`IQ_COLLECTION`、`IQ_DIM`。
  这些字段仍会 mutate 进 `settings`（诊断口径真实），但 watcher 打 `WARNING restart required`。
- `POSTGRES_*` 变更可在静默期手动调 `await reload_engine()`（`app/utils/database.py`）原地重建 engine，
  无需整进程重启；默认不自动重建。

## 6. 播种（把本地配置灌进 Nacos）

```bash
cd "C:/Users/A/Desktop/docker/nacos" && docker compose up -d   # 起 Nacos(老地方)
curl -s http://127.0.0.1:8848/nacos/v1/console/health/readiness   # 期望 OK(v2.3.2 返回 OK 即就绪)
python scripts/nacos_seed.py                    # 发布 5 个 dataId（幂等，重跑覆盖）
```

控制台 `http://127.0.0.1:8848/nacos` 可见 5 条配置。播种脚本会剥 `NACOS_*` 键与行内 ` # ` 注释
（只剥「空白后跟 #」，不剥口令/URL 里的 `#`）。

## 7. 启用与回退

1. 两个 `.env`（及 model_stub）置 `NACOS_ENABLED=true`，重启服务。
2. 启动日志应打 `loaded from Nacos dataId=...`。
3. **回退**：置 `NACOS_ENABLED=false` 重启，即回到本地 `.env`/yaml（前提：本地 `.env` 未裁剪）。

## 8. 回滚语义（变更校验失败）

- `retrieval.env`：快照旧 env → 更新 → `reset_caches` + `get_settings` 重校验；失败回滚旧 env，打
  `nacos env change rejected`。
- yaml：先 `yaml.safe_load` + 顶层 dict 校验 → 置 `_NACOS_YAML` → `reset_caches` + `get_settings`
  重校验（连带模型契约）；失败回滚旧内容。
- `qingxi.env`：快照旧 env → `reload_settings()`；失败回滚 env。不校验语义（类型由 pydantic 强转兜底）。

## 9. 风险与注意

- **本地 `.env` 残留回退**：迁移后应裁剪本地 `.env` 至 bootstrap-only，否则 Nacos 删除的键会回退到
  本地旧值（pydantic-settings env 优先）。裁剪前请先确认已 `nacos_seed.py` 播种成功。
- **线程竞争**：watcher 改配置与在途请求读并发，单请求可能看到混合快照（对配置可接受）。
- **auth**：本期 `NACOS_AUTH_ENABLE=false`；未来开 auth 需在 client 加 `POST /nacos/v1/auth/login`
  取 `accessToken` 并作为 query 参数带上（未实现）。
