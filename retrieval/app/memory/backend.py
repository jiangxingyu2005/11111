"""长期事实记忆抽象：real=mem0(AsyncMemory, Milvus)，fake=内存 dict。

对齐 store/fake.py 的 fake/real 分发：接口签名一致，route 层只认接口不认实现。
fail-closed：任何 mem0/嵌入/LLM 异常记 warning、返回空，绝不拖主链（同 fuse/rules.py 的 load_rules）。

注意：
  - 只用 OSS 的 AsyncMemory（自托管），禁止托管版 MemoryClient（会外发数据）。
  - 抽取交给 mem0 自己（add 内部 LLM 提取+合并），不手写「抽事实」prompt（避免双重 LLM）。
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..config import effective_cfg, get_settings

log = logging.getLogger("retrieval.memory.backend")


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
    """mem0 长期记忆（自托管 AsyncMemory + Milvus + 本地 Ollama/BGE-M3）。"""

    def __init__(self) -> None:
        self._mem = None

    async def _get(self):
        if self._mem is not None:
            return self._mem
        from mem0 import AsyncMemory  # 懒加载：仅 real 通道才需要 mem0ai

        s = get_settings()
        mc = (effective_cfg().get("memory") or {}).get("mem0", {}) or {}
        config = {
            "vector_store": {
                "provider": mc.get("vector_store", "milvus"),
                "config": {
                    "collection_name": mc.get("collection", "mem0_memory"),
                    "uri": s.milvus_uri,
                },
            },
            # LLM 走本地 Ollama/vLLM 的 OpenAI 兼容端点（graphrag_llm_base_url 约定已含 /v1，
            # 同 report.py 直接拼 /chat/completions；mem0 的 openai SDK 会在 base_url 后拼 /chat/completions，
            # 故此处不能再 +"/v1"，否则变 /v1/v1）
            "llm": {
                "provider": "openai",
                "config": {
                    "model": s.graphrag_llm_model,
                    "openai_base_url": (s.graphrag_llm_base_url or "").rstrip("/"),
                    "api_key": s.graphrag_llm_api_key or "EMPTY",
                },
            },
            # embedder 本地化（已定案）：huggingface provider 进程内 sentence-transformers
            # 跑 graphrag_embedding_model（BAAI/bge-m3），与离线 qingxi 侧同模型同空间（1024 维），零外联。
            # 注意：不能复用 retrieval_embed_url——那是自定义 /embed 协议（body {texts:[]}），
            # 非 OpenAI 兼容，喂给 mem0 的 openai embedder 会请求失败/维度错。
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": s.graphrag_embedding_model,
                },
            },
        }
        self._mem = AsyncMemory.from_config(config_dict=config)
        return self._mem

    @staticmethod
    def _extract(res) -> list[str]:
        results = (res or {}).get("results", []) if isinstance(res, dict) else []
        return [r.get("memory", "") for r in results if r.get("memory")]

    async def add(self, user_id: str, messages: list[dict]) -> None:
        try:
            m = await self._get()
            await m.add(messages, user_id=user_id)
        except Exception as e:
            log.warning("mem0 add failed (视为未写入): %s", e)

    async def search(self, query: str, user_id: str, k: int = 5) -> list[str]:
        try:
            m = await self._get()
            return self._extract(await m.search(query, user_id=user_id, limit=k))
        except Exception as e:
            log.warning("mem0 search failed (返回空): %s", e)
            return []

    async def get_all(self, user_id: str) -> list[str]:
        try:
            m = await self._get()
            return self._extract(await m.get_all(user_id=user_id))
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
