"""GP-06 CDC:实时增量同步(ems cdc/consumer.py 的异步移植)。

WAL -> Kafka(Debezium)-> 规则判定 -> MERGE / DETACH DELETE。
  op=c/r 增:merge_segment + 绑定(coalesce 保底)
  op=u   改:merge_segment + 旧绑定按差量校正
  op=d   删:DETACH DELETE
重放安全:MERGE 幂等,同批消息重放两次图无重复。
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ParentDocument
from .. import db
from ..embeddings import EmbeddingClient
from ..milvus_client import (
    F_CHILD_ID, F_CONTENT, F_DOC_ID, F_ENABLED, F_KB_ID, F_KG_ENTITY_IDS,
    F_PARENT_ENABLED, F_PARENT_ID, F_VECTOR, AsyncMilvusClient,
)
from ..neo4j_client import Neo4jClient

# binds 解析器:parent_id -> [(entity_id, kg_role, kg_confidence), ...]
BindResolver = Callable[[str], list[tuple]]


def coalesce_parent(after: dict) -> dict:
    """coalesce 保底:CDC 事件字段可能为 null,用默认值兜底。"""
    return {
        "parent_id": after["parent_id"],
        "doc_id": after.get("doc_id") or "",
        "kb_id": after.get("kb_id"),
        "title": after.get("title") or "无标题",
        "enabled": True if after.get("enabled") is None else bool(after.get("enabled")),
        "chapter_no": after.get("chapter_no") or "",
    }


async def _sync_binds(after: dict, neo4j: Neo4jClient, resolver: Optional[BindResolver]) -> None:
    if resolver is None:
        return
    for entity_id, role, conf in resolver(after["parent_id"]):
        await neo4j.merge_binds(entity_id, after["parent_id"], role, conf)


async def _diff_binds(after: dict, neo4j: Neo4jClient, resolver: Optional[BindResolver]) -> None:
    """差量校正:现有绑定与目标集合比对,删多余、补缺失。"""
    if resolver is None:
        return
    pid = after["parent_id"]
    desired = {eid: (role, conf) for eid, role, conf in resolver(pid)}
    current = set(await neo4j.entity_ids_of_parent(pid))
    for eid in current - set(desired):
        await neo4j.delete_bind(eid, pid)
    for eid, (role, conf) in desired.items():
        await neo4j.merge_binds(eid, pid, role, conf)


async def handle_event(event: dict, neo4j: Neo4jClient,
                       resolver: Optional[BindResolver] = None) -> str:
    """处理单条 Debezium CDC 事件。返回动作描述。"""
    op = event.get("op")
    before = event.get("before")
    after = event.get("after")
    if op in ("c", "r") and after:
        await neo4j.merge_segment(coalesce_parent(after))
        await _sync_binds(after, neo4j, resolver)
        return f"create/refresh segment {after['parent_id']}"
    if op == "u" and after:
        await neo4j.merge_segment(coalesce_parent(after))
        await _diff_binds(after, neo4j, resolver)
        return f"update segment {after['parent_id']}"
    if op == "d" and before:
        await neo4j.detach_delete_segment(before["parent_id"])
        return f"delete segment {before['parent_id']}"
    return "ignored"


async def process_messages(messages: list[dict], neo4j: Neo4jClient,
                           resolver: Optional[BindResolver] = None) -> list[str]:
    """批量处理(重放安全:MERGE 幂等)。"""
    return [await handle_event(m, neo4j, resolver) for m in messages]


async def handle_child_event(event: dict, milvus: AsyncMilvusClient,
                             embeddings: Optional[EmbeddingClient] = None,
                             session: Optional[AsyncSession] = None) -> str:
    """消费者B:Child(chunk_relation) 事件 → 重新嵌入 → Milvus upsert/delete。

    op=c/u 增改:重嵌子块文本 → upsert(幂等);op=d 删:delete child_id。
    适配 qingxi:chunk_relation 无 doc_id/kb_id/parent_enabled/kg_entity_ids 列,
    传入 session 时从 parent_document 反查 doc_id/kb_id/parent_enabled 补齐。
    """
    op = event.get("op")
    after = event.get("after")
    before = event.get("before")

    # 从父块反查缺失字段(qingxi 子块表不含 doc_id/kb_id/parent_enabled)
    doc_id, kb_id, parent_enabled = "", "", True
    pid = (after or before or {}).get("parent_id", "")
    if session is not None and pid:
        rows = await db.select_where(session, ParentDocument, parent_id=pid)
        if rows:
            p = rows[0]
            doc_id = p.get("doc_id", "")
            kb_id = str(p["kb_id"]) if p.get("kb_id") else ""
            parent_enabled = bool(p.get("enabled", True))

    if op in ("c", "u") and after:
        embeddings = embeddings or EmbeddingClient()
        child_id = after["child_id"]
        content = after.get("content", "")
        vec = await asyncio.to_thread(embeddings.embed_one, content)
        await milvus.upsert([{
            F_CHILD_ID: child_id, F_CONTENT: content,
            F_PARENT_ID: after.get("parent_id", "") or pid,
            F_DOC_ID: after.get("doc_id") or doc_id,
            F_KB_ID: after.get("kb_id") or kb_id,
            F_ENABLED: True if after.get("enabled") is None else bool(after["enabled"]),
            F_PARENT_ENABLED: parent_enabled,
            F_KG_ENTITY_IDS: list(after.get("kg_entity_ids") or []),
            F_VECTOR: vec,
        }])
        return f"upsert child {child_id}"
    if op == "d" and before:
        await milvus.delete([before["child_id"]])
        return f"delete child {before['child_id']}"
    return "ignored"


async def process_child_messages(messages: list[dict], milvus: AsyncMilvusClient,
                                 embeddings: Optional[EmbeddingClient] = None,
                                 session: Optional[AsyncSession] = None) -> list[str]:
    """消费者B 批量处理(upsert/delete 幂等)。"""
    return [await handle_child_event(m, milvus, embeddings, session) for m in messages]


class CDCStream:
    """真实 Kafka 消费循环(confluent_kafka,经线程池 poll)。"""

    def __init__(self, settings: Optional[dict] = None):
        from ..config import build_settings
        self.settings = settings or build_settings()

    async def run(self, neo4j: Neo4jClient, resolver: Optional[BindResolver] = None,
                  max_polls: Optional[int] = None) -> int:
        from confluent_kafka import Consumer  # 延迟导入
        import json
        k = self.settings["kafka"]
        c = Consumer({
            "bootstrap.servers": k["bootstrap_servers"],
            "group.id": k["group_id"],
            "auto.offset.reset": "earliest",
        })
        c.subscribe([k["topic"]])
        count = 0
        try:
            while True:
                msg = await asyncio.to_thread(c.poll, 1.0)
                if msg is None:
                    if max_polls is not None and count >= max_polls:
                        break
                    continue
                if msg.error():
                    continue
                event = json.loads(msg.value().decode("utf-8"))
                payload = event.get("payload", event)   # Debezium 信封
                await handle_event(payload, neo4j, resolver)
                count += 1
        finally:
            c.close()
        return count


async def main() -> None:
    """独立消费进程入口:python -m app.graphrag.cdc.consumer。"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cdc")
    from ..neo4j_client import Neo4jClient
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.run_constraints()
            await CDCStream().run(neo4j)
    except Exception as e:
        logger.warning("CDC 消费循环退出(不影响主服务): %s", e)


if __name__ == "__main__":
    asyncio.run(main())
