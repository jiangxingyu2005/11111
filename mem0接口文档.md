# mem0 记忆模块接口文档（retrieval 在线服务）

> 服务：retrieval（在线检索侧，默认端口 `8100`，与 qingxi 离线侧 `8001` 是两套服务）。
> 代码位置：`retrieval/app/memory/`（backend / compressor / service / session_store）+ `retrieval/app/auth.py`。

## 0. 结论：mem0 接口「暴露」了哪几层

mem0 没有独立的记忆管理 REST 端点（无 `/memory` CRUD）。它以内嵌 + 配置位的方式暴露，共三层：

| 层 | 形态 | 谁在用 | 位置 |
|---|---|---|---|
| ① HTTP 接入面 | `POST /api/retrieval` 的 `token` / `session_id` 字段 | 前端/调用方 | `retrieval/app/api/retrieval.py` |
| ② 代码级接口 | `MemoryBackend` / `SessionStore` 两个 Protocol | 内部编排（service 层） | `retrieval/app/memory/*.py` |
| ③ 本地化部署接口 | mem0 的 vector_store / llm / embedder 三个 provider 配置位 | 部署/运维 | `.env` + `config/retrieval.yaml` |

记忆不是独立产品能力，而是检索链路的「上下文增强」：读侧把摘要/窗口/长期事实注入 prompt，写侧 fire-and-forget 落库，全程 fail-closed，失败绝不影响检索主链。

---

## 1. HTTP 接入面（唯一对外入口）

记忆只能通过检索主端点隐式触发，没有单独的查询/写入记忆的 HTTP 端点。

### `POST /api/retrieval`

请求体在既有 `query` / `scope` 基础上，新增两个可选字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | 是 | 检索问题 |
| `scope` | string[] | 是 | 知识库范围（权限硬约束） |
| `session_id` | string | 否 | 会话 id，记忆按 `(user_id, session_id)` 隔离多轮 |
| `token` | string | 否 | 身份令牌，服务端解析出 `user_id`（缺省/解析失败则本轮跳过记忆） |
| `eval_mode` | bool | 否 | 评测模式 |

```json
{
  "query": "刚才说的那个火控雷达载频是多少",
  "scope": ["kb_public"],
  "session_id": "s_8f3a",
  "token": "u_1001"
}
```

**行为**（源码 `retrieval/app/api/retrieval.py`）：

- 入口 `resolve_user_id(token)` → `user_id`；解析失败返回 `None`，本轮静默跳过记忆（fail-closed，不留 degraded）。
- 生成前 `load_context(user_id, session_id, query)` 装载三层上下文，注入 prompt 的「历史上下文」段（非证据，仅供理解指代/背景）。
- 生成后 `spawn_background(commit_turn(...))` 后台写回本轮 + 触发压缩，不阻塞 SSE 响应。

---

## 2. 身份解析接口（`retrieval/app/auth.py`）

`resolve_user_id(token: str | None) -> str | None`，由 `MEMORY_AUTH_MODE` 决定两种模式：

| 模式 | 行为 | 适用 |
|---|---|---|
| `passthrough`（默认） | token 即 user_id（调用方自报身份） | 内网/测试 |
| `jwt` | PyJWT 验签 HS256，`sub` 即 user_id | 需防伪造身份的生产 |

- `jwt` 模式依赖 `MEMORY_JWT_SECRET`；未配置或验签失败 → 返回 `None`。
- 任何解析异常只记 warning，返回 `None`（主链照跑）。

---

## 3. 三层记忆模型

| 层 | 内容 | 作用域 | 存储（real / fake） |
|---|---|---|---|
| ① 窗口 window | 最近 N 轮原文 | session | PG `messages` / 内存 dict |
| ② 摘要 summary | 递归压缩后的叙事 | session | PG `conversations.summary` / 内存 dict |
| ③ 长期事实 facts | mem0 抽取的持久事实 | user | mem0 + Milvus / 内存 dict |

每次检索注入：`摘要 + 窗口原文 + 检索到的 mem0 事实 + 本轮召回证据`。

---

## 4. 代码级接口（内部抽象层）

对齐 `store/fake.py` 的 fake/real 分发：接口签名一致，route 层只认接口不认实现。

### 4.1 长期事实抽象 `MemoryBackend`（`retrieval/app/memory/backend.py`）

```python
class MemoryBackend(Protocol):
    async def add(self, user_id: str, messages: list[dict]) -> None: ...
    async def search(self, query: str, user_id: str, k: int = 5) -> list[str]: ...
    async def get_all(self, user_id: str) -> list[str]: ...
```

| 实现 | 后端 | 说明 |
|---|---|---|
| `Mem0Backend` | real | 包 `AsyncMemory.from_config`，抽取交给 mem0（add 内部 LLM 提取+合并，不手写抽取 prompt） |
| `FakeMemoryBackend` | fake | 内存 dict，`search` 用 `lex_overlap` 词面重叠打分，300 条容量上限 |

分发：`get_memory_backend()` / `reset_memory_backend()`（按 `retrieval_backend` 返回单例）。

### 4.2 会话/消息抽象 `SessionStore`（`retrieval/app/memory/session_store.py`）

```python
class SessionStore(Protocol):
    async def get_or_create_conversation(self, user_id: str, session_id: str) -> dict: ...
    async def append_message(self, conv_id: str, role: str, content: str, token_count: int) -> None: ...
    async def load_window(self, conv_id: str, limit: int) -> list[dict]: ...
    async def update_summary(self, conv_id: str, summary: str, compress_count: int) -> None: ...
    async def trim_messages(self, conv_id: str, keep_last: int) -> None: ...
```

| 实现 | 后端 | 说明 |
|---|---|---|
| `PgSessionStore` | real | asyncpg；启动自动 `CREATE TABLE IF NOT EXISTS` |
| `FakeSessionStore` | fake | 内存 dict，重启即失 |

### 4.3 编排入口（`retrieval/app/memory/service.py`）

检索主链只调这两个函数：

```python
async def load_context(user_id, session_id, query) -> dict:
    # 读侧：返回 {"summary": str, "window": list[dict], "memories": list[str]}
    # 任一失败记 warning、该层留空（fail-closed）

async def commit_turn(user_id, session_id, query, answer) -> None:
    # 写侧：append 本轮(user+assistant) -> maybe_compress，由 spawn_background fire-and-forget
```

### 4.4 压缩流水线（`retrieval/app/memory/compressor.py`）

- `estimate_tokens(text) -> int`：轻量 token 估算（中文 1 token/字、英数 1 token/词）。
- `maybe_compress(store, memory, conv, cfg)`：按 token 用量触发（占用 ≥ 预算×`trigger_ratio`），丢弃原文前先把事实 `add` 进 mem0，压缩次数封顶 `max_compressions`，全程 fail-closed。

---

## 5. mem0 本地化部署接口（配置位）

`Mem0Backend._get()` 构造的 mem0 配置（`retrieval/app/memory/backend.py`），三处 provider 全部指向本地，零外联：

| provider | 配置键 | 取值来源 | 说明 |
|---|---|---|---|
| `vector_store` | `provider: milvus`<br>`config.collection_name`<br>`config.uri` | `memory.mem0.collection`（默认 `mem0_memory`）<br>`MILVUS_HOST/PORT`（`milvus_uri`） | 复用 qingxi 同款 Milvus |
| `llm` | `provider: openai`<br>`config.model`<br>`config.openai_base_url`<br>`config.api_key` | `graphrag_llm_model`<br>`graphrag_llm_base_url`<br>`graphrag_llm_api_key` | 走本地 Ollama/vLLM 的 OpenAI 兼容端点 |
| `embedder` | `provider: huggingface`<br>`config.model` | `graphrag_embedding_model`（`BAAI/bge-m3`） | 进程内 sentence-transformers，1024 维，与离线侧同模型同空间 |

### 关键约束（易踩坑，已在此注释固化）

1. **`openai_base_url` 不能再拼 `/v1`**：`graphrag_llm_base_url` 约定已含 `/v1`，mem0 的 openai SDK 会在 base_url 后拼 `/chat/completions`，再拼会变 `/v1/v1`。
2. **embedder 不能复用 `retrieval_embed_url`**：那是自定义 `/embed` 协议（body `{texts:[],type:"query"}`），非 OpenAI 兼容，喂给 mem0 的 openai embedder 会失败/维度错。故用 `huggingface` provider 进程内跑 BGE-M3。
3. **只用 OSS `AsyncMemory`（自托管）**，禁止托管版 `MemoryClient`（会外发数据）。

---

## 6. 配置参数

### 6.1 环境变量（`.env`，`retrieval/app/config.py`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_ENABLED` | `false` | 记忆总开关（默认关，不影响既有部署） |
| `MEMORY_AUTH_MODE` | `passthrough` | `passthrough` / `jwt` |
| `MEMORY_JWT_SECRET` | `""` | jwt 模式验签密钥 |

### 6.2 算法参数（`config/retrieval.yaml` 的 `memory:` 段）

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | `false` | 总开关（与 `MEMORY_ENABLED` 联动） |
| `window_size` | `10` | 窗口保留消息条数 |
| `budget_tokens` | `4096` | 记忆区预算（摘要+窗口估算 token 上限） |
| `trigger_ratio` | `0.8` | 占用 ≥ 预算×0.8 触发压缩 |
| `max_compressions` | `5` | 压缩次数上限，压满后只留最新 N 条 |
| `summary_target_ratio` | `0.4` | 压缩后目标占用比例（预留） |
| `mem0.vector_store` | `milvus` | mem0 向量库 provider |
| `mem0.collection` | `mem0_memory` | mem0 collection 名 |

---

## 7. 数据表（real 后端，PG）

retrieval 由「三库只读」变为对这两张表读写，需给 retrieval 账号授予这两张表写权限：

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    compress_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, session_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
```

---

## 8. 依赖

- `mem0ai>=0.1`（仅 real 通道懒加载）
- `PyJWT>=2.8`（仅 jwt 模式）
- `sentence-transformers>=2.7`（real 通道 embedding 需要，体积大默认不装，见 `retrieval/requirements.txt`）

---

## 9. fail-closed 纪律（贯穿全模块）

- 记忆读写/压缩任何失败只记 warning、该层留空，**绝不拖检索主链**，不留 degraded 码（对齐 `fuse/rules.py::load_rules`）。
- 身份解析失败 → 本轮跳过记忆；mem0 add/search 失败 → 视为空；压缩失败 → 保留原文不裁剪。
- 默认 `enabled=false`，现有部署与既有用例行为不变。
