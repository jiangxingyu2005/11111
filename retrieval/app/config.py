"""retrieval 服务配置加载（.env + config/*.yaml 双源）。

纪律（V2.0 §5.1 / V3.0 §5）：
  - 缺必需变量 -> 启动失败，禁止带默认值静默启动
  - 算法参数（tau/topk/预算）以 config/*.yaml 为准，env 只做部署级覆盖
  - 启动日志必须打印生效值，防"配了没生效"的静默漂移
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("retrieval.config")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class Settings(BaseSettings):
    """必需项无默认值：缺失即 ValidationError，服务拒绝启动。"""

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    # 三库只读连接（IF-9，必需）
    postgres_host: str
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: str
    postgres_database: str = "qingxi"

    milvus_host: str
    milvus_port: int = 19530
    milvus_chunk_collection: str = "ems_chunks"
    milvus_entity_collection: str = "ems_entities"
    milvus_community_collection: str = "community_scene"

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # 模型端点
    retrieval_embed_mode: str = "remote"          # local / remote
    retrieval_embed_url: str = ""
    graphrag_embedding_model: str = "BAAI/bge-m3"   # 定案：全量 BGE-M3，与离线侧同模型同空间

    graphrag_llm_base_url: str = ""
    graphrag_llm_api_key: str = "EMPTY"
    graphrag_llm_model: str = ""

    retrieval_rerank_service_url: str = ""

    # 功能开关（默认保守）
    retrieval_bm25_enabled: bool = False
    retrieval_bm25_url: str = ""
    retrieval_redis_url: str = ""
    retrieval_graph_route_enabled: bool = True
    retrieval_community_enabled: bool = False

    retrieval_budget_total_ms: int = 5200

    retrieval_trace_export: str = "file"          # file / kafka / http
    retrieval_trace_dir: str = "logs/trace"

    # —— 后端通道（可运行工程包新增）——
    # fake = 内存三库（data/seed/*.json），零外部依赖跑通全链路；real = 真 PG/Milvus/Neo4j
    retrieval_backend: str = "real"
    retrieval_seed_dir: str = "data/seed"
    # 故障注入：逗号分隔 vector,graph,pg,llm,rerank,bm25,community（仅 fake 后端生效，用于跑降级矩阵）
    retrieval_fault_inject: str = ""
    # 逃生阀：real 通道却仍用占位模型时，默认拒绝启动；置 1 则只报 ERROR 放行（仅供 staging）
    retrieval_allow_placeholder: bool = False

    # —— 记忆管理（多人隔离 + 会话多轮上下文 + mem0 长期事实）——
    memory_enabled: bool = False                 # 功能开关（默认关，不影响现有部署）
    memory_auth_mode: str = "passthrough"        # passthrough（token 即 user_id）/ jwt（验签 sub）
    memory_jwt_secret: str = ""                  # jwt 模式验签密钥（MEMORY_JWT_SECRET）

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()  # type: ignore[call-arg]  # 值全部来自 .env
    # 后端通道校验（fake 只能用于 dev/测试，禁止上生产）
    if s.retrieval_backend not in ("fake", "real"):
        raise ValueError(f"RETRIEVAL_BACKEND must be fake|real, got {s.retrieval_backend}")
    if s.retrieval_backend == "fake":
        seed_dir = ROOT / s.retrieval_seed_dir
        if not (seed_dir / "chunks.json").exists():
            raise FileNotFoundError(
                f"RETRIEVAL_BACKEND=fake 但 seed 缺失: {seed_dir} —— 请先跑 `python scripts/gen_seed.py`"
            )
    # 模型契约校验：占位模型在 real 通道是硬停项（E1 静默风险），fake 通道只告警
    problems = validate_model_contract(
        retrieval_cfg(), s.retrieval_backend, s.retrieval_allow_placeholder, s)
    hard = [p for p in problems if p.startswith("RETRIEVAL_BACKEND=real")]
    if hard and not s.retrieval_allow_placeholder:
        raise ValueError("模型契约校验失败：" + "；".join(problems))
    for p in problems:
        log.error("模型契约问题：%s", p)

    # 启动打印生效值（脱敏：不打印口令）
    safe = s.model_dump()
    for k in list(safe):
        if "password" in k or "api_key" in k:
            safe[k] = "***"
    log.info("retrieval settings loaded: %s", safe)
    m = (retrieval_cfg().get("models") or {})
    log.info("model contract: embed=%s(%s dim=%s sparse=%s) rerank=%s llm=%s",
             (m.get("embed") or {}).get("id"), (m.get("embed") or {}).get("backend"),
             (m.get("embed") or {}).get("dim"), (m.get("embed") or {}).get("sparse_enabled"),
             (m.get("rerank") or {}).get("backend"), (m.get("llm") or {}).get("backend"))
    return s


@lru_cache
def load_yaml(name: str) -> dict:
    """读取 config/<name>.yaml；文件缺失即启动失败（yaml 即契约，IF-8）。"""
    p = CONFIG_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"config contract missing: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def retrieval_cfg() -> dict:
    return load_yaml("retrieval")


def embed_dim() -> int:
    """嵌入 dense 维度（单一事实源 = config/retrieval.yaml:models.embed.dim）。

    换 embedding 模型时只改 yaml 的 dim + 重建 collection + 重跑 seed，代码零改动。
    scripts（init_milvus / gen_seed / check_embed_space）统一从这里取，不再各自写死 1024。
    """
    m = retrieval_cfg().get("models") or {}
    return int((m.get("embed") or {}).get("dim", 1024))


# ---- 模型契约校验（E1/E2/E3 参数位的守门人）----
# 存在理由：模型身份只登记在 config/retrieval.yaml:models。
# "配了 BAAI/bge-m3 实际跑 hash"是本项目最危险的静默漂移——链路全绿、契约合法、
# 零报错，但召回全是噪声。所以必须在启动时把声明与实际对齐，对不齐就拒绝启动。
ALLOWED_BACKENDS = {
    # openai / generic = 接入外部自有服务，走 endpoints.<kind>.url
    "embed": {"hash", "bge_m3", "openai", "generic"},
    "rerank": {"hash", "bge", "generic"},
    "llm": {"echo", "openai", "hf", "generic"},
}
# backend -> 协议（决定了 Provider 怎么去连）
BACKEND_PROTOCOL = {"hash": "local", "bge_m3": "local", "bge": "local",
                    "hf": "local", "echo": "local",
                    "openai": "openai", "generic": "generic"}


def validate_model_contract(cfg: dict, backend: str, allow_placeholder: bool,
                            settings=None) -> list[str]:
    """返回问题列表；空列表表示通过。硬性问题在调用方决定是否 raise。"""
    problems: list[str] = []
    m = cfg.get("models") or {}
    emb, rk, llm = m.get("embed") or {}, m.get("rerank") or {}, m.get("llm") or {}

    for kind, node in (("embed", emb), ("rerank", rk), ("llm", llm)):
        b = node.get("backend", "")
        if b not in ALLOWED_BACKENDS[kind]:
            problems.append(f"models.{kind}.backend={b!r} 非法，应为 {sorted(ALLOWED_BACKENDS[kind])}")

    # dim 一致性不再在此写死 1024：声明(models.embed.dim)与实际模型输出
    # 由 providers.EmbedProvider.embed 的 expect_dim 闸门校验(换模型忘重建库会在那里炸)。

    # 稀疏路两处开关必须联动，否则"以为开了 hybrid 其实没开"
    sp_yaml = bool(((cfg.get("milvus") or {}).get("sparse") or {}).get("enabled"))
    if bool(emb.get("sparse_enabled")) != sp_yaml:
        problems.append(
            f"models.embed.sparse_enabled={emb.get('sparse_enabled')} 与 "
            f"milvus.sparse.enabled={sp_yaml} 不一致（IF-1 扩列后须同改 true）")

    # reranker 两处模型名必须一致（yaml 里 reranker.model 是历史字段）
    if (cfg.get("reranker") or {}).get("model") != rk.get("id"):
        problems.append(
            f"reranker.model={(cfg.get('reranker') or {}).get('model')} 与 "
            f"models.rerank.id={rk.get('id')} 不一致")

    # 声明要连外部服务，就必须给出地址——否则 Provider 会去打空 URL，
    # 表现为"配了自有模型却每个请求都失败"，属于配置层静默漂移
    eps = cfg.get("endpoints") or {}
    for kind in ("embed", "rerank", "llm"):
        b = (m.get(kind) or {}).get("backend", "")
        if BACKEND_PROTOCOL.get(b) in ("openai", "generic"):
            url = ((eps.get(kind) or {}).get("url") or "").strip() \
                or os.getenv(f"{kind.upper()}_URL", "") \
                or (os.getenv("LLM_UPSTREAM", "") if kind == "llm" else "")
            # LLM 已收敛到统一接口 app/llm.py（复用 GRAPHRAG_LLM_*，走 settings 而非 yaml endpoints）；
            # .env 不进 os.environ，故只能从 settings.graphrag_llm_base_url 取，避免误报「没给地址」。
            if not url and kind == "llm" and settings is not None:
                url = (getattr(settings, "graphrag_llm_base_url", "") or "").strip()
            if not url:
                problems.append(
                    f"models.{kind}.backend={b} 需要 endpoints.{kind}.url（或 env {kind.upper()}_URL），"
                    f"当前为空——接了自有模型却没给地址，请求必然全部失败")

    # 红线：real 通道禁止占位嵌入后端（E1 缺失是静默的，只能硬停）
    if backend == "real" and emb.get("backend") == "hash":
        msg = ("RETRIEVAL_BACKEND=real 但 models.embed.backend=hash —— 真环境跑哈希嵌入"
               "会静默产出噪声召回且零报错。E1 未完成前禁止上生产；"
               "如需 staging 放行请显式置 RETRIEVAL_ALLOW_PLACEHOLDER=1")
        problems.append(msg)
    return problems


@lru_cache
def query_parser_cfg() -> dict:
    return load_yaml("query_parser")


def effective_cfg() -> dict:
    """生效配置 = retrieval.yaml + 后端通道覆盖（dev_overrides.<backend>）。

    存在理由：fake 通道用哈希嵌入，余弦量纲与 BGE-M3 完全不同（Top1≈0.43 vs ≈0.85），
    若复用 τ=0.72 会把召回全部杀成 0 —— 这是"配置绑定数据"的静默漂移。
    覆盖只改【阈值与超时】这类环境相关量，绝不改算法语义参数（RRF k / top_k / 融合结构）。
    """
    import copy
    cfg = copy.deepcopy(retrieval_cfg())
    s = get_settings()
    ov = cfg.pop("dev_overrides", {}).get(s.retrieval_backend, {}) or {}
    for path, val in ov.items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    cfg["_backend"] = s.retrieval_backend
    return cfg


def reset_caches() -> None:
    """测试用：env 变更后清空所有 lru_cache（否则改了 env 不生效，属静默漂移）。"""
    get_settings.cache_clear()
    load_yaml.cache_clear()
    retrieval_cfg.cache_clear()
    query_parser_cfg.cache_clear()
