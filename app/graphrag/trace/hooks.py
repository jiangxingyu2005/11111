"""GP-09:查询命中钩子(Plan §3.11,按查询去重口径,async 移植)。

挂到 GP-08 查询链路。两级计数:
  分段级——一次查询命中某父块,该父块 recall_count +1(与子块数无关);
  文件级——一次查询命中某文件,该文件 source_file_registry.recall_count +1(与子块数无关)。
recall_log 记录每个子块命中的时间线(查询原文/命中子块/时间)。
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ParentDocument, RecallLog, SourceFileRegistry
from .. import db


def make_trace_hook(session: AsyncSession) -> Callable:
    """返回 async 钩子:由 query 链路在生成答案后调用,写入溯源计数与日志。"""

    async def hook(query: str, hits: list[dict], answer: str = "") -> None:
        # 分段级:按父块去重,一个查询命中同一父块多个子块 -> recall_count 只 +1
        parents_hit: dict[str, list[dict]] = {}
        files_hit: set[str] = set()
        for h in hits:
            pid = h.get("parent_id") or h.get("child_id", "").rsplit("#", 1)[0]
            parents_hit.setdefault(pid, []).append(h)
            if h.get("doc_id"):
                files_hit.add(h["doc_id"])

        for pid, phits in parents_hit.items():
            await db.increment(session, ParentDocument, "parent_id", pid, "recall_count", 1)
            for h in phits:
                await db.insert(session, RecallLog, {
                    "query": query,
                    "doc_id": h.get("doc_id", ""),
                    "child_id": h.get("child_id"),
                    "parent_id": pid,
                    "answer_text": answer,
                })

        # 文件级:按查询去重,一个查询命中同一文件多个子块 -> recall_count 只 +1
        for fid in files_hit:
            await db.increment(session, SourceFileRegistry, "doc_id", fid, "recall_count", 1)

    return hook
