"""QRY-06 父块回捞（M-C3）。

small-to-big：检索在子块，上下文在父块。PG 按 parent_id 批量 IN 读父块全文，
比对 content_version（版本不一致 -> hit 级 parent_fetch_failed 标记）。
PG 熔断 -> Milvus 行内 content 兜底 + parent_fetch_unavailable（降级矩阵 #17）。

双通道：real = asyncpg 批量 IN；fake = FakeStore.parents_batch（同返回结构）。
"""
from __future__ import annotations

import logging

from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from .rrf import Hit

log = logging.getLogger("retrieval.fuse.backfill")


async def backfill(hits: list[Hit], collector: DegradedCollector) -> dict[str, dict]:
    """返回 {parent_id: {content, content_version, summary, keywords, degraded?}}。

    失败整批兜底（行内 content 已在 Hit.content），不中断主链。
    """
    parent_ids = sorted({h.parent_id for h in hits if h.parent_id})
    if not parent_ids:
        return {}
    if effective_cfg().get("_backend") == "fake":
        from ..store.fake import get_fake_store
        try:
            return get_fake_store().parents_batch(parent_ids)
        except Exception as e:
            log.warning("parent backfill(fake) failed: %s", e)
            collector.record(DegradedCode.PARENT_FETCH_UNAVAILABLE, str(e)[:200], "backfill")
            return {}

    try:
        import asyncpg
        conn = await asyncpg.connect(get_settings().pg_dsn)
        try:
            rows = await conn.fetch(
                """
                SELECT parent_id, content, content_version, summary, keywords
                FROM parent_document WHERE parent_id = ANY($1::text[])
                """,
                parent_ids,
            )
        finally:
            await conn.close()
        # kg_confidence 不在 parent_document 表(real 通道该字段属 Neo4j :BINDS 关系)，
        # 故不 SELECT；规则引擎/报告对缺失 kg_confidence 已 fail-closed（p.get() 返回 None）。
        return {
            r["parent_id"]: {
                "content": r["content"],
                "content_version": r["content_version"],
                "summary": r["summary"],
                "keywords": r["keywords"],
            }
            for r in rows
        }
    except Exception as e:
        log.warning("parent backfill failed: %s", e)
        collector.record(DegradedCode.PARENT_FETCH_UNAVAILABLE, str(e)[:200], "backfill")
        return {}  # 主链用 Hit.content（行内冗余）兜底，不中断


def check_version(hit: Hit, parent: dict) -> bool:
    """版本一致性校验（离线重切/重灌会导致父块版本漂移）。

    返回 False = 该 hit 记 parent_fetch_failed（hit 级标记，非整路降级）。
    """
    want = hit.meta.get("content_version")
    got = parent.get("content_version")
    return want is None or got is None or want == got
