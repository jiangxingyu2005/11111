import logging
import os
import sys
import threading
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("qingxi.config")

_ROOT = Path(__file__).resolve().parent.parent  # 仓库根（common/ 所在）
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Settings(BaseSettings):
    """从 .env / 环境变量读取配置,字段与 .env.example 一一对应。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MinerU(Docker 容器中的 mineru-api,v1 协议)
    mineru_api_base_url: str = "http://127.0.0.1:8008"
    mineru_api_prefix: str = "/v1"
    mineru_timeout: int = 600            # 单次 HTTP 请求超时(秒)
    mineru_poll_interval: int = 2        # 轮询任务状态间隔(秒)
    mineru_max_wait_seconds: int = 1800  # 等待任务完成的最大时长(秒)
    mineru_output_formats: str = "markdown,middle_json"  # 逗号分隔
    mineru_tier: str = ""                # 可选 flash/basic/standard/advanced,留空用默认
    mineru_ocr_mode: str = "auto"        # auto/txt/ocr

    # 本服务
    upload_dir: str = "./uploads"
    output_dir: str = "./output"
    max_upload_mb: int = 50

    # ASR(音频转文字,faster-whisper 本地模型)
    asr_model_size: str = "small"       # 模型大小:tiny/base/small/medium/large-v3
    asr_model_dir: str = "./model"      # 模型本地目录(单独 model 文件夹)
    asr_device: str = "cpu"             # 推理设备:cpu / cuda
    asr_compute_type: str = "int8"      # 计算精度:int8(CPU 推荐)/ float16 / float32
    asr_language: str = "zh"            # 识别语言,留空则自动检测

    # BLIP(图片语义理解,transformers 本地模型;BLIP2 因内存不足改用 BLIP-base)
    blip_model_id: str = "Salesforce/blip-image-captioning-base"  # 模型仓库 id
    blip_model_dir: str = "./model/blip"                          # 模型本地目录(单独 blip 文件夹)
    blip_max_new_tokens: int = 60                                 # 生成 caption 的最大 token 数

    # 翻译(英文→中文,MarianMT 本地模型,把 BLIP 英文 caption 翻成中文)
    translator_model_id: str = "Helsinki-NLP/opus-mt-en-zh"  # 模型仓库 id
    translator_model_dir: str = "./model/mt-en-zh"           # 模型本地目录(单独文件夹)
    translator_max_new_tokens: int = 128                     # 翻译生成的最大 token 数

    # 视频(抽帧 -> 每帧走图片提取链路)
    video_frame_rate: float = 2.0                    # 每秒抽帧数
    video_max_frames: int = 60                       # 最大处理帧数(去重后仍超则均匀采样)
    video_frame_similarity_threshold: float = 0.95   # 相邻帧直方图相关性阈值,>= 此值判为相似跳过

    # HTML(上传 HTML 文件 -> 提取正文)
    html_noise_tags: str = "script,style,noscript,iframe,nav,footer,header,aside,form"  # 逗号分隔的去除标签

    # IQ 信号(H5 上传 -> SNR 清洗 -> IQ 补全 -> 信号处理 -> 26 特征)
    iq_segment_len: int = 128              # IQ 片段长度(SNR/embed/分类统一分段长度)
    iq_snr_threshold_db: float = -10.0     # SNR 阈值(dB),低于此值的片段剔除
    iq_target_fs: float = 10e6             # 重采样目标采样率(Hz),默认 10MHz
    iq_model_dir: str = "model/iq"         # IQ 模型目录(单独 iq 文件夹)
    iq_xgb_model_path: str = "model/iq/xgb_15class.json"
    iq_cnn_model_path: str = "model/iq/cnn_15class_full.pth"
    iq_label_map_path: str = "model/iq/label_map_15.json"
    iq_buquan_model_path: str = "model/iq/best.pth"

    # IQ 嵌入(独立检索库 ems_iq;wavesfm_rml 重导出的 256 维 CLS 特征)
    iq_embedding_model_path: str = "model/iq/wavesfm_embed.onnx"
    iq_collection: str = "ems_iq"
    iq_dim: int = 256

    # 清洗(页眉/页脚判定参数)
    clean_header_threshold: int = 3    # 短文本重复多少次判为页眉/页脚
    clean_header_max_len: int = 60     # 页眉/页脚候选文本的最大长度

    # 打分(文本质量评分阈值与权重)
    score_min_chunk_len: int = 20      # 长度维度下界(字符),块短于此则长度分记 0
    score_max_chunk_len: int = 512     # 长度维度上界(字符),块长于此则长度分记满
    score_q_quarantine: float = 0.4    # 质量分隔离阈值,低于此分不分级保留
    score_q_keep: float = 0.6          # 质量分保留阈值,高于此分按高权重
    score_w_len: float = 0.25          # 长度维度权重
    score_w_info: float = 0.30         # 信息密度维度权重
    score_w_content: float = 0.25      # 内容词占比维度权重
    score_w_hits: float = 0.20         # 参数实体命中维度权重

    # MinIO 对象存储(S3 兼容)
    minio_endpoint: str = "http://localhost:9000"  # S3 API 地址
    minio_access_key: str = "minioadmin"           # 访问密钥
    minio_secret_key: str = "minioadmin"           # 访问私钥
    minio_raw_bucket: str = "qingxi-raw"           # 源文件桶(原始上传文件)
    minio_clean_bucket: str = "qingxi-clean"       # 清洗后文件桶(clean 结果)
    minio_secure: bool = False                     # 是否 https

    # PostgreSQL(pgvector 扩展,向量存储)
    postgres_host: str = "localhost"               # 主机
    postgres_port: int = 5434                      # 端口(宿主机映射到容器 5432)
    postgres_user: str = "postgres"                # 用户
    postgres_password: str = "postgres"            # 密码
    postgres_database: str = "postgres"            # 数据库名

    # GraphRAG(ems_graphrag 融合:清洗 md -> PG 父/子块 -> Milvus -> Neo4j)
    graphrag_enabled: bool = True                  # 是否在清洗完自动入图谱(IQ 不入)

    # Milvus 向量库(子块 1024 维向量)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "ems_chunks"
    milvus_dim: int = 1024

    # Neo4j 图库(实体/父块/关系)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"                  # 改成实际 Neo4j 密码

    # Kafka(CDC 实时增量,Phase 3 用)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "ems.public.parent_document"
    kafka_group_id: str = "ems_cdc_group"

    # 切分(父块按 H1-H3 / 子块固定字符 + 比例重叠)
    graphrag_chunk_size: int = 512
    graphrag_chunk_overlap: float = 0.10

    # 嵌入(mock = 确定性哈希向量;openai = text-embedding-3-large;bge = 本地 FlagEmbedding)
    graphrag_embedding_provider: str = "mock"
    graphrag_embedding_model: str = "text-embedding-3-large"  # bge: BAAI/bge-m3 / bge-large-zh-v1.5
    graphrag_embedding_dim: int = 1024
    graphrag_embedding_batch_size: int = 64
    graphrag_embedding_device: str = ""          # bge 设备:空串 = 自动,cuda/cpu
    graphrag_embedding_use_fp16: bool = False    # bge 半精度(省显存;CPU 上建议 False)

    # LLM(mock = 确定性正则抽取,openai/qwen = function calling;qwen = 国产模型本地 vLLM)
    graphrag_llm_provider: str = "mock"
    graphrag_llm_base_url: str = "https://api.openai.com/v1"
    graphrag_llm_api_key: str = ""
    graphrag_llm_model: str = "gpt-4o"
    graphrag_llm_temperature: float = 0.0
    graphrag_llm_max_tokens: int = 4096
    graphrag_llm_max_retries: int = 2            # function calling 网络/解析失败重试次数(准确度优先)
    graphrag_llm_use_function_calling: bool = True  # False=JSON prompt 输出(3B 等小模型不支持 function calling)
    graphrag_llm_frequency_penalty: float = 0.5  # 重复抑制(3B greedy 解码易陷入实体名重复循环)
    graphrag_llm_presence_penalty: float = 0.3   # 同上,抑制已出现 token 再被生成

    # 图抽取置信度三档分流
    graphrag_confidence_high: float = 0.9
    graphrag_confidence_low: float = 0.6
    graphrag_tau_a: float = 0.6                    # 关系抽取置信度下限
    graphrag_tau_b: float = 0.9                    # 关系抽取高置信线

    # 在线查询(链路 C)
    graphrag_query_topk: int = 10
    graphrag_query_rerank_topk: int = 5
    graphrag_query_expand_depth: int = 2
    graphrag_query_entity_boost_weight: float = 0.15
    graphrag_query_expand_limit: int = 30          # 图扩展最多返回的实体数(防枢纽节点扩爆)
    graphrag_query_facts_per_entity: int = 15      # 每个命中实体最多取的一跳关系事实数
    graphrag_query_hop_quotas: str = "12,10,4,4"   # 图扩展每跳名额(第1..4跳),不足的让给其余候选
    graphrag_query_path_facts: int = 10            # 多跳关系链事实最多条数

    # 路由(离线/实时分流阈值)
    graphrag_route_file_size_mb: int = 10
    graphrag_route_batch_count: int = 5

    # 消歧(实体归一对齐:① 名称/别名/指纹确定性 ② 向量语义 ③ LLM 灰区)
    graphrag_resolve_enabled: bool = True        # 主开关:入库时对齐到全局注册表
    graphrag_resolve_fingerprint: bool = True    # ① 指纹路径
    graphrag_resolve_vector_enabled: bool = True  # ② 向量层(ems_entities 语义匹配)
    graphrag_resolve_sim_high: float = 0.62      # 自动并阈值(bge-large-zh-v1.5 实测校准;换模型需重校准)
    graphrag_resolve_sim_low: float = 0.55       # 灰区下界
    graphrag_resolve_llm_gate: bool = False      # ③ 灰区交 LLM(默认关)
    graphrag_resolve_topk: int = 5               # 向量召回 top-k
    graphrag_entity_collection: str = "ems_entities"  # 实体名向量库名(Milvus,消歧用)

    # 漏抽质检门禁(入库质检 + /graph/quality 健康度;默认值待真实数据校准)
    graphrag_quality_entity_chunk_ratio: float = 0.2   # 实体数 < 父块数*比率 → kg_status=low
    graphrag_quality_max_orphan_rate: float = 0.5      # 孤立率门禁(超 → healthy=false)
    graphrag_quality_min_binds_coverage: float = 0.3   # BINDS 覆盖率门禁(低于 → healthy=false)
    graphrag_quality_max_empty_ratio: float = 0.5      # 空图谱文档占比门禁(超 → healthy=false)

    @property
    def output_formats(self) -> list[str]:
        return [f.strip() for f in self.mineru_output_formats.split(",") if f.strip()]

    @property
    def postgres_dsn(self) -> str:
        # +asyncpg:纯异步驱动,兼容 Windows 默认事件循环
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )


# ====================================================================
# Nacos 配置中心：全量迁入后 .env 以 Nacos 为唯一事实源，本地只留 NACOS_*。
# NACOS_ENABLED=false 时完全等同旧行为。qingxi 的 settings 是 import 时冻结的
# 单例（约 133 处 `settings.X` 直接读），热更新靠「原地 mutate 同一对象」实现，
# 绝不重绑定模块级 settings / SessionLocal（避免 import 绑定坑）。
# ====================================================================


class _BootstrapSettings(BaseSettings):
    """只读 NACOS_* bootstrap 键；其余键 extra=ignore。"""

    model_config = SettingsConfigDict(env_file=str(_ROOT / ".env"), extra="ignore")

    nacos_enabled: bool = False
    nacos_server_addr: str = "http://127.0.0.1:8848"
    nacos_namespace: str = ""
    nacos_group: str = "DEFAULT_GROUP"
    nacos_username: str = ""
    nacos_password: str = ""
    nacos_data_ids: str = "qingxi.env"


def _nacos_client():
    from common.nacos_client import NacosConfigClient  # 已把仓库根加进 sys.path
    b = _BootstrapSettings()
    return NacosConfigClient(b.nacos_server_addr, namespace=b.nacos_namespace,
                             group=b.nacos_group, username=b.nacos_username,
                             password=b.nacos_password)


def _nacos_data_ids() -> list[str]:
    return [d.strip() for d in _BootstrapSettings().nacos_data_ids.split(",") if d.strip()]


def _bootstrap_nacos() -> None:
    """启动期从 Nacos 拉 qingxi.env -> os.environ（须在 settings 构造前调用）。"""
    b = _BootstrapSettings()
    if not b.nacos_enabled:
        return
    try:
        from common.nacos_client import parse_properties
        client = _nacos_client()
    except Exception as e:
        log.error("nacos bootstrap 失败（回退本地 .env）: %s", e)
        return
    for data_id in _nacos_data_ids():
        try:
            content = client.get_config(data_id)
        except Exception as e:
            log.error("nacos get_config %s failed（回退本地 .env）: %s", data_id, e)
            continue
        env = parse_properties(content)
        os.environ.update(env)
        log.info("loaded from Nacos dataId=%s（%d keys）", data_id, len(env))


_bootstrap_nacos()
settings = Settings()


def get_settings() -> "Settings":
    """返回同一冻结单例（别名，身份稳定；供未来测试/对称 retrieval）。"""
    return settings


# 变更后需「重启」才真正生效的字段（喂给 import 期 engine / 懒加载模型缓存 /
# 进程全局 pymilvus 连接，后台线程拆建风险大于收益）。
def _requires_restart(name: str) -> bool:
    u = name.upper()
    if u in ("UPLOAD_DIR", "OUTPUT_DIR", "IQ_COLLECTION", "IQ_DIM", "IQ_MODEL_DIR",
             "IQ_XGB_MODEL_PATH", "IQ_CNN_MODEL_PATH", "IQ_LABEL_MAP_PATH",
             "IQ_BUQUAN_MODEL_PATH", "IQ_EMBEDDING_MODEL_PATH"):
        return True
    for p in ("POSTGRES_", "MILVUS_", "NEO4J_", "MINIO_", "KAFKA_", "ASR_", "BLIP_",
              "TRANSLATOR_", "GRAPHRAG_EMBEDDING_"):
        if u.startswith(p):
            return True
    return False


def reload_settings() -> list[str]:
    """原地刷新冻结单例：新实例化 Settings() 拿类型强转后的值，逐字段 setattr 写回。

    返回发生变化的字段名列表。绝不重绑定模块级 settings。
    """
    fresh = Settings()  # 读 os.environ（优先于 .env），类型强转
    changed: list[str] = []
    for name, value in fresh.model_dump().items():
        if getattr(settings, name) != value:
            setattr(settings, name, value)  # validate_assignment=False，值已强转
            changed.append(name)
    return changed


def apply_nacos_change(data_id: str, content: str) -> None:
    """Nacos 变更回调（watcher 线程调用）：只处理 qingxi.env。失败回滚 env。"""
    from common.nacos_client import parse_properties
    if data_id != "qingxi.env":
        log.warning("nacos 忽略未知 dataId=%s（service A 只监听 qingxi.env）", data_id)
        return
    new_env = parse_properties(content)
    old = {k: os.environ.get(k) for k in new_env}
    os.environ.update(new_env)
    try:
        changed = reload_settings()
    except Exception as e:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        log.error("nacos env change rejected（回滚旧值）: %s", e)
        return
    if not changed:
        log.info("nacos env applied（无字段变化）")
        return
    restart = [c for c in changed if _requires_restart(c)]
    log.info("nacos env applied, changed=%s", changed)
    if restart:
        log.warning("配置变更需重启才生效: %s（POSTGRES_* 可调 reload_engine()）", restart)


def start_nacos_watcher() -> None:
    """启动 Nacos 变更监听 daemon 线程（enabled 才起）。main.lifespan 调用。"""
    if not _BootstrapSettings().nacos_enabled:
        return
    client = _nacos_client()
    stop = threading.Event()
    threading.Thread(target=client.watch,
                     args=(_nacos_data_ids(), apply_nacos_change, stop),
                     name="qingxi-nacos-watch", daemon=True).start()
    log.info("nacos watcher started（dataIds=%s）", _nacos_data_ids())
