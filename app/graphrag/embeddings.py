"""向量嵌入客户端(ems common/embeddings.py 移植,配置源改为 qingxi .env)。

provider:
  mock   字符 3-gram 哈希,确定性(同文本同向量,余弦 1.0)。
  openai text-embedding-3-large,走网络。
  bge    本地 FlagEmbedding(BAAI/bge-m3 或 bge-large-zh-v1.5,1024 维),懒加载缓存。
同步实现:mock 纯计算;openai/bge 由调用方经 asyncio.to_thread 委托。
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from typing import Optional

from .config import build_settings


class EmbeddingClient:
    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or build_settings()
        self.provider = self.settings["embedding"]["provider"]
        self.dim = int(self.settings["embedding"]["dim"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            return self._embed_openai(texts)
        if self.provider == "bge":
            return self._embed_bge(texts)
        return [_hash_embed(t, self.dim) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI  # 延迟导入
        s = self.settings["embedding"]
        client = OpenAI(base_url=self.settings["llm"].get("base_url"),
                        api_key=self.settings["llm"].get("api_key") or "sk-dummy")
        resp = client.embeddings.create(model=s["model"], input=texts)
        return [d.embedding for d in resp.data]

    def _embed_bge(self, texts: list[str]) -> list[list[float]]:
        """BGE 本地向量(FlagEmbedding)。

        实体名两两比较是对称场景,不加 query/passage 指令,直接 dense 编码后 L2 归一化
        (与 mock 输出口径一致,配合 Milvus COSINE)。模型懒加载 + 进程级缓存,首次加载慢。
        """
        s = self.settings["embedding"]
        model_name = s["model"]
        model = _load_bge(model_name, s.get("device") or None, bool(s.get("use_fp16", False)))
        if "m3" in model_name.lower():
            dense = model.encode(texts)["dense_vecs"]
        else:
            dense = model.encode(texts)
        vecs = _l2_normalize_rows(dense)
        if vecs and len(vecs[0]) != self.dim:
            _bge_logger.warning("BGE 模型维度 %d 与配置 dim %d 不一致",
                                len(vecs[0]), self.dim)
        return vecs


def _hash_embed(text: str, dim: int) -> list[float]:
    """字符 3-gram 哈希嵌入:稳定、同文本同向量。"""
    vec = [0.0] * dim
    t = "  " + text + "  "
    for i in range(len(t) - 2):
        gram = t[i:i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


_bge_logger = logging.getLogger(__name__)
_bge_lock = threading.Lock()
_bge_models: dict[str, object] = {}


def _local_bge_path(model_name: str) -> str:
    """本地目录或已缓存 snapshot 优先,避免 FlagEmbedding 访问 HuggingFace。

    配置写成绝对/相对目录(含 config.json)时直接用;否则尝试 hub 本地缓存;
    都没有才回退 repo id(可能触发下载)。
    """
    from pathlib import Path
    p = Path(model_name)
    if p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return model_name


def _load_bge(model_name: str, device, use_fp16: bool):
    """BGE 模型懒加载 + 进程级缓存(加载慢,不能每次 embed 都重建)。

    device/use_fp16 来自 .env;device 空串 -> devices=None(FlagEmbedding 自动选设备)。
    """
    with _bge_lock:
        if model_name in _bge_models:
            return _bge_models[model_name]
        model_path = _local_bge_path(model_name)
        from FlagEmbedding import BGEM3FlagModel, FlagModel  # 延迟导入
        devices = device or None
        if "m3" in model_name.lower():
            model = BGEM3FlagModel(model_path, devices=devices, use_fp16=use_fp16)
        else:
            model = FlagModel(model_path, devices=devices, use_fp16=use_fp16)
        _bge_models[model_name] = model
        return model


def _l2_normalize_rows(rows) -> list[list[float]]:
    out: list[list[float]] = []
    for r in rows:
        v = [float(x) for x in r]
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n > 0 else v)
    return out


def get_embeddings() -> EmbeddingClient:
    return EmbeddingClient()
