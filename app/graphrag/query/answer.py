"""GP-08:LLM 生成 + 引用标注(Plan §3.9,async 移植)。

答案带原文引用([n] 对应 context 下标),TRACE 计数钩子挂载点由 router 注入。
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

from ..llm import LLMClient


def build_contexts(hits: list[dict]) -> list[dict]:
    """把召回命中归一为生成上下文(含来源标注)。"""
    return [{"child_id": h.get("child_id"), "doc_id": h.get("doc_id"),
             "content": h.get("content", "")} for h in hits]


async def generate_answer(question: str, hits: list[dict],
                          llm: Optional[LLMClient] = None,
                          trace_hook: Optional[Callable] = None) -> str:
    llm = llm or LLMClient()
    contexts = build_contexts(hits)
    # openai 走网络,mock 纯计算;统一经线程池,不阻塞事件循环
    answer = await asyncio.to_thread(llm.generate_answer, question, contexts)
    if trace_hook:
        await trace_hook(question, hits, answer)
    return answer
