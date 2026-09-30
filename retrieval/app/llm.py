"""统一大模型调用接口（OpenAI 兼容 /chat/completions）。

为什么存在：
  「调用大模型」此前散在 4 处，各自手写 httpx.post(graphrag_llm_base_url + "/chat/completions")，
  超时/错误处理/响应解析不一致，且换线上/本地 LLM 要改 4 个文件。这里收敛成单一事实源。

用法：
  - chat(...)        非流式，返回完整正文（记忆摘要压缩 / L2 问句理解）
  - chat_stream(...) 流式，逐 token yield（RPT 报告生成）
  - chat_sync(...)   同步非流式（启动预热 / keepalive 心跳这类线程内 ping）

线上/本地切换只改 .env 的 GRAPHRAG_LLM_BASE_URL / GRAPHRAG_LLM_MODEL / GRAPHRAG_LLM_API_KEY，
代码零改动。所有函数在「未配置」或「请求失败」时抛异常，由调用方按各自降级策略 fail-closed。
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .config import get_settings

log = logging.getLogger("retrieval.llm")

DEFAULT_TIMEOUT_S = 60.0


def _configured(s) -> bool:
    return bool(s.graphrag_llm_base_url and s.graphrag_llm_model)


def _endpoint(s) -> str:
    # graphrag_llm_base_url 约定已含 /v1（如 http://host:11434/v1），此处再拼 /chat/completions
    return s.graphrag_llm_base_url.rstrip("/") + "/chat/completions"


def _headers(s) -> dict:
    return {"Authorization": f"Bearer {s.graphrag_llm_api_key}"}


def _body(s, messages, *, temperature: float, max_tokens: int | None, stream: bool) -> dict:
    body = {"model": s.graphrag_llm_model, "messages": messages,
            "stream": stream, "temperature": temperature}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def _extract_content(data: dict) -> str:
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""


async def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """非流式调用，返回去首尾空白的完整正文。失败抛异常（调用方 fail-closed）。"""
    s = get_settings()
    if not _configured(s):
        raise RuntimeError("LLM 未配置（graphrag_llm_base_url / model 为空）")
    body = _body(s, messages, temperature=temperature, max_tokens=max_tokens, stream=False)
    async with httpx.AsyncClient(timeout=timeout) as cli:
        r = await cli.post(_endpoint(s), headers=_headers(s), json=body)
        r.raise_for_status()
        return _extract_content(r.json()).strip()


async def chat_stream(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> AsyncIterator[str]:
    """流式调用（SSE），逐 token yield。失败抛异常（调用方捕获后降级）。"""
    s = get_settings()
    if not _configured(s):
        raise RuntimeError("LLM 未配置（graphrag_llm_base_url / model 为空）")
    body = _body(s, messages, temperature=temperature, max_tokens=None, stream=True)
    async with httpx.AsyncClient(timeout=timeout) as cli:
        async with cli.stream("POST", _endpoint(s), headers=_headers(s), json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
                tok = delta.get("content")
                if tok:
                    yield tok


def chat_sync(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """同步非流式调用（启动预热 / keepalive 这类线程内 ping 用，不阻塞事件循环）。"""
    s = get_settings()
    if not _configured(s):
        raise RuntimeError("LLM 未配置（graphrag_llm_base_url / model 为空）")
    body = _body(s, messages, temperature=temperature, max_tokens=max_tokens, stream=False)
    with httpx.Client(timeout=timeout) as cli:
        r = cli.post(_endpoint(s), headers=_headers(s), json=body)
        r.raise_for_status()
        return _extract_content(r.json()).strip()
