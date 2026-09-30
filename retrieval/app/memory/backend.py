"""长期事实记忆抽象：real=mem0(AsyncMemory, Milvus)，fake=内存 dict。

对齐 store/fake.py 的 fake/real 分发：接口签名一致，route 层只认接口不认实现。
fail-closed：任何 mem0/嵌入/LLM 异常记 warning、返回空，绝不拖主链（同 fuse/rules.py 的 load_rules）。

注意：
  - 只用 OSS 的 AsyncMemory（自托管），禁止托管版 MemoryClient（会外发数据）。
  - 抽取交给 mem0 自己（add 内部 LLM 提取+合并），不手写「抽事实」prompt（避免双重 LLM）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol, runtime_checkable

from ..config import effective_cfg, get_settings, embed_dim

log = logging.getLogger("retrieval.memory.backend")

# mem0 的 telemetry 默认开启，其 telemetry 块会把 vector_store 配置二次实例化
# （MilvusDBConfig(**model_dump())）；而 mem0 把 MilvusDBConfig.token 错写成非 Optional 的
# str（默认 None），model_dump() 出的 token=None 被 pydantic 判成非法字符串抛 ValidationError，
# 直接炸掉整个 _build()。mem0 读的是 os.environ（进程环境，非 .env），故必须在首次
# `from mem0 import Memory` 之前在此关闭 telemetry（项目本就禁止外发数据），绕过该块。
os.environ.setdefault("MEM0_TELEMETRY", "false")


@runtime_checkable
class MemoryBackend(Protocol):
    async def add(self, user_id: str, messages: list[dict]) -> None: ...
    async def search(self, query: str, user_id: str, k: int = 5) -> list[str]: ...
    async def get_all(self, user_id: str) -> list[str]: ...


def _split_segments(text: str, max_len: int = 300) -> list[str]:
    """把一段消息按句切成长度可控的 fact 片段（fake 不做 LLM 抽取，只切分）。"""
    text = (text or "").strip()
    if not text:
        return []
    segs, cur = [], ""
    for ch in text:
        cur += ch
        if ch in "。！？!?\n" and len(cur) >= 8:
            segs.append(cur.strip())
            cur = ""
    if cur.strip():
        segs.append(cur.strip())
    # 超长段再硬切
    out = []
    for s in segs:
        while len(s) > max_len:
            out.append(s[:max_len])
            s = s[max_len:]
        if s:
            out.append(s)
    return out


class FakeMemoryBackend:
    """内存长期记忆：user_id -> [fact 文本]。search 用词面重叠（lex_overlap）打分。"""

    def __init__(self) -> None:
        self._mem: dict[str, list[str]] = {}

    async def add(self, user_id: str, messages: list[dict]) -> None:
        facts = self._mem.setdefault(user_id, [])
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            for seg in _split_segments(content):
                if seg and seg not in facts:
                    facts.append(seg)
        if len(facts) > 300:  # 容量上限，防无限膨胀（最旧挤出）
            self._mem[user_id] = facts[-300:]

    async def search(self, query: str, user_id: str, k: int = 5) -> list[str]:
        from ..store.hash_embed import hash_sparse, lex_overlap

        facts = self._mem.get(user_id, [])
        if not facts or not query:
            return []
        q = hash_sparse(query)
        scored = [(lex_overlap(q, hash_sparse(f)), f) for f in facts]
        scored = sorted((s, f) for s, f in scored if s > 0.0)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:k]]

    async def get_all(self, user_id: str) -> list[str]:
        return list(self._mem.get(user_id, []))


class Mem0Backend:
    """mem0 长期记忆（自托管 Memory + Milvus + 本地 BGE-M3 / DeepSeek LLM）。

    全程 to_thread：加载 2.2GB 本地 BGE-M3 与 encode 都在 worker 线程跑，绝不阻塞事件循环
    （否则首查 load_context 会卡住整条 SSE 主链，与「记忆 fail-closed 不拖主链」纪律相悖）。
    """

    def __init__(self) -> None:
        self._mem = None

    def _local_embed_model(self) -> str:
        """本地 BGE-M3 snapshot 优先（ModelScope 缓存，免联网）；无则回退声明模型名（联网下载）。"""
        from ..config import ROOT
        p = ROOT / "model" / "models" / "BAAI--bge-m3" / "snapshots" / "master"
        return str(p) if p.is_dir() else get_settings().graphrag_embedding_model

    def _build(self):
        """同步构建 mem0（在 worker 线程内执行，加载大模型不阻塞事件循环）。"""
        from mem0 import Memory  # 懒加载：仅 real 通道才需要 mem0ai

        s = get_settings()
        mc = (effective_cfg().get("memory") or {}).get("mem0", {}) or {}
        config = {
            "vector_store": {
                "provider": mc.get("vector_store", "milvus"),
                "config": {
                    "collection_name": mc.get("collection", "mem0_memory"),
                    # mem0>=2.x 字段名是 url（旧版 uri 已被移除，多余字段会被 pydantic 直接拒绝）
                    "url": s.milvus_uri,
                    # 必须显式给维度：mem0 默认 1536，与 BGE-M3 的 1024 错位会导致建库即炸
                    "embedding_model_dims": embed_dim(),
                    # 与检索侧统一余弦量纲（bge-m3 归一化后 L2/COSINE 排序等价，语义一致更清晰）
                    "metric_type": "COSINE",
                },
            },
            # LLM 走 OpenAI 兼容端点（线上 DeepSeek 或本地 Ollama），由 .env 的 GRAPHRAG_LLM_* 决定。
            # openai SDK 会在 base_url 后拼 /chat/completions，故此处不能再 +"/v1"，否则变 /v1/v1。
            "llm": {
                "provider": "openai",
                "config": {
                    "model": s.graphrag_llm_model,
                    "openai_base_url": (s.graphrag_llm_base_url or "").rstrip("/"),
                    "api_key": s.graphrag_llm_api_key or "EMPTY",
                },
            },
            # embedder 本地化：huggingface provider 进程内 sentence-transformers 跑本地 BGE-M3。
            # 不能复用 retrieval_embed_url——那是自定义 /embed 协议（body {texts:[]}），非 OpenAI 兼容。
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self._local_embed_model(),
                },
            },
        }
        return Memory.from_config(config_dict=config)

    async def _get(self):
        if self._mem is None:
            self._mem = await asyncio.to_thread(self._build)
        return self._mem

    @staticmethod
    def _extract(res) -> list[str]:
        results = (res or {}).get("results", []) if isinstance(res, dict) else []
        return [r.get("memory", "") for r in results if r.get("memory")]

    async def add(self, user_id: str, messages: list[dict]) -> None:
        try:
            m = await self._get()
            await asyncio.to_thread(m.add, messages, user_id=user_id)
        except Exception as e:
            log.warning("mem0 add failed (视为未写入): %s", e)

    async def search(self, query: str, user_id: str, k: int = 5) -> list[str]:
        try:
            m = await self._get()
            res = await asyncio.to_thread(m.search, query, filters={"user_id": user_id}, top_k=k)
            return self._extract(res)
        except Exception as e:
            log.warning("mem0 search failed (返回空): %s", e)
            return []

    async def get_all(self, user_id: str) -> list[str]:
        try:
            m = await self._get()
            res = await asyncio.to_thread(m.get_all, filters={"user_id": user_id})
            return self._extract(res)
        except Exception as e:
            log.warning("mem0 get_all failed (返回空): %s", e)
            return []


_backend: MemoryBackend | None = None


def get_memory_backend() -> MemoryBackend:
    """按后端通道返回 fake/real 单例（fake=内存，real=mem0）。"""
    global _backend
    if _backend is None:
        _backend = FakeMemoryBackend() if get_settings().retrieval_backend == "fake" else Mem0Backend()
    return _backend


def reset_memory_backend() -> None:
    """测试用：清空单例（改了 env/backend 后重新装载）。"""
    global _backend
    _backend = None
