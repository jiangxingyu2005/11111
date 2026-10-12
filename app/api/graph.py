"""GraphRAG 接口:抽取字典增删改查(Phase 1)。

字典 = 定义「要抽取哪些实体节点 / 关系」的唯一数据源,全局共享、不分知识库。
抽取侧每次现查,改字典即时生效。逻辑删除(置 deleted_at)后不参与抽取。
"""

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EntityAlias, ExtractionDict
from ..schemas import (
    EntityAliasCreate,
    EntityAliasRead,
    EntityAliasUpdate,
    ExtractionDictCreate,
    ExtractionDictRead,
    ExtractionDictUpdate,
    GraphQueryRequest,
)
from ..utils.database import get_db

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

# derived = 规则派生关系(同频段/同时段/同区域),登记用于图例与溯源,不参与 LLM 抽取
_ALLOWED_CATEGORY = {"entity", "relation", "derived"}


def _validate_category(category: str) -> None:
    if category not in _ALLOWED_CATEGORY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"category 必须是 entity / relation / derived,收到: {category!r}",
        )


async def _get_row(db: AsyncSession, dict_id: UUID) -> ExtractionDict | None:
    stmt = select(ExtractionDict).where(
        ExtractionDict.id == dict_id,
        ExtractionDict.deleted_at.is_(None),
    )
    return await db.scalar(stmt)


@router.post("/dict", response_model=ExtractionDictRead, status_code=status.HTTP_201_CREATED)
async def create_dict(payload: ExtractionDictCreate, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """新增字典条目(实体类型或关系定义)。"""
    _validate_category(payload.category)
    row = ExtractionDict(**payload.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/dict", response_model=list[ExtractionDictRead])
async def list_dict(
    db: AsyncSession = Depends(get_db),
    category: str | None = Query(None, description="按类别过滤:entity / relation"),
) -> list[ExtractionDictRead]:
    """列表(不含已逻辑删除),可按类别过滤。"""
    stmt = select(ExtractionDict).where(ExtractionDict.deleted_at.is_(None))
    if category is not None:
        _validate_category(category)
        stmt = stmt.where(ExtractionDict.category == category)
    stmt = stmt.order_by(ExtractionDict.created_at)
    return list(await db.scalars(stmt))


@router.get("/dict/{dict_id}", response_model=ExtractionDictRead)
async def get_dict(dict_id: UUID, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """查询单条字典(不含已逻辑删除)。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    return row


@router.put("/dict/{dict_id}", response_model=ExtractionDictRead)
async def update_dict(dict_id: UUID, payload: ExtractionDictUpdate, db: AsyncSession = Depends(get_db)) -> ExtractionDictRead:
    """更新字典(仅传要改的字段)。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    if payload.category is not None:
        _validate_category(payload.category)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(row, k, v)
    row.updated_at = func.now()
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/dict/{dict_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dict(dict_id: UUID, db: AsyncSession = Depends(get_db)) -> None:
    """逻辑删除:置 deleted_at,不再参与抽取。"""
    row = await _get_row(db, dict_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="字典条目不存在")
    row.deleted_at = func.now()
    row.updated_at = func.now()
    await db.commit()


@router.post("/alias", response_model=EntityAliasRead, status_code=status.HTTP_201_CREATED)
async def create_alias(payload: EntityAliasCreate, db: AsyncSession = Depends(get_db)) -> EntityAliasRead:
    """新增别名库条目(代号 → 类别字典实体类型)。"""
    from ..graphrag.dict import load_entity_defs

    entity_defs = await load_entity_defs(db)
    valid_types = {d["name"] for d in entity_defs}
    if payload.entity_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"entity_type 必须是当前启用的实体类型之一 {sorted(valid_types)}，收到: {payload.entity_type!r}",
        )
    exists = await db.scalar(select(EntityAlias).where(EntityAlias.alias == payload.alias))
    if exists is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"别名 {payload.alias!r} 已存在(对应类型 {exists.entity_type!r})",
        )
    row = EntityAlias(**payload.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/alias", response_model=list[EntityAliasRead])
async def list_alias(
    db: AsyncSession = Depends(get_db),
    entity_type: str | None = Query(None, description="按类别过滤(如 辐射源)"),
) -> list[EntityAliasRead]:
    """别名库列表,可按实体类型过滤。"""
    stmt = select(EntityAlias)
    if entity_type is not None:
        stmt = stmt.where(EntityAlias.entity_type == entity_type)
    stmt = stmt.order_by(EntityAlias.created_at)
    return list(await db.scalars(stmt))


@router.get("/alias/{alias_id}", response_model=EntityAliasRead)
async def get_alias(alias_id: UUID, db: AsyncSession = Depends(get_db)) -> EntityAliasRead:
    """查询单条别名库条目。"""
    row = await db.scalar(select(EntityAlias).where(EntityAlias.id == alias_id))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="别名条目不存在")
    return row


@router.put("/alias/{alias_id}", response_model=EntityAliasRead)
async def update_alias(alias_id: UUID, payload: EntityAliasUpdate, db: AsyncSession = Depends(get_db)) -> EntityAliasRead:
    """更新别名库条目(仅传要改的字段)。"""
    from ..graphrag.dict import load_entity_defs

    row = await db.scalar(select(EntityAlias).where(EntityAlias.id == alias_id))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="别名条目不存在")

    data = payload.model_dump(exclude_unset=True)
    new_alias = data.get("alias")
    new_type = data.get("entity_type")

    if new_alias is not None and new_alias != row.alias:
        dup = await db.scalar(select(EntityAlias).where(EntityAlias.alias == new_alias))
        if dup is not None and dup.id != row.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"别名 {new_alias!r} 已存在(对应类型 {dup.entity_type!r})",
            )

    if new_type is not None and new_type != row.entity_type:
        entity_defs = await load_entity_defs(db)
        valid_types = {d["name"] for d in entity_defs}
        if new_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"entity_type 必须是当前启用的实体类型之一 {sorted(valid_types)}，收到: {new_type!r}",
            )

    if new_alias is not None:
        row.alias = new_alias
    if new_type is not None:
        row.entity_type = new_type
    row.updated_at = func.now()
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/alias/{alias_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alias(alias_id: UUID, db: AsyncSession = Depends(get_db)) -> None:
    """删除别名库条目(物理删除)。"""
    row = await db.scalar(select(EntityAlias).where(EntityAlias.id == alias_id))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="别名条目不存在")
    await db.delete(row)
    await db.commit()


@router.post("/query")
async def graph_query(payload: GraphQueryRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """在线查询:意图分流 → 实体匹配 → A/B 双路召回 → 重排 → (关系型)图扩展 → LLM 生成。

    命中结果计入溯源(parent/file recall_count + recall_log)。Neo4j/Milvus 不可用时
    返回 502,不影响其它接口与主链路。
    """
    from ..graphrag.dict import load_relations
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.llm import LLMClient
    from ..graphrag.milvus_client import AsyncMilvusClient
    from ..graphrag.neo4j_client import Neo4jClient
    from ..graphrag.query import router as query_router
    from ..graphrag.trace.hooks import make_trace_hook

    relation_defs = await load_relations(db)
    from ..utils.rules import list_rules
    rule_rows = [
        {"id": r.id, "name": r.name, "condition": r.condition, "action": r.action,
         "enabled": r.enabled, "kb_id": str(r.kb_id) if r.kb_id else None, "priority": r.priority}
        for r in await list_rules(db)
    ]
    try:
        async with Neo4jClient() as neo4j:
            result = await query_router.search(
                query=payload.query,
                milvus=AsyncMilvusClient(),
                neo4j=neo4j,
                embeddings=EmbeddingClient(),
                llm=LLMClient(),
                relation_defs=relation_defs,
                kb_id=payload.scope_ids(),
                trace_hook=make_trace_hook(db),
                rules=rule_rows,
                iq_doc_id=payload.iq_doc_id,
            )
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception("图谱查询失败: %s", payload.query)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"图谱查询失败: {e}",
        )
    return result


@router.post("/query/stream")
async def graph_query_stream(payload: GraphQueryRequest, db: AsyncSession = Depends(get_db)):
    """在线查询(SSE 流式):逐步返回 意图分流→实体匹配→双路召回→重排→图扩展→生成。

    SSE 事件序列:meta → progress* → results → token* → done / error。
    供前端实时展示「双路检索每一步」与流式答案;逻辑与 /graph/query 一致,只是逐步骤下发。
    """
    from ..graphrag.dict import load_relations
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.llm import LLMClient
    from ..graphrag.milvus_client import AsyncMilvusClient
    from ..graphrag.neo4j_client import Neo4jClient
    from ..graphrag.query import router as query_router
    from ..graphrag.query.answer import build_contexts
    from ..graphrag.query.nl2query import nl2query
    from ..graphrag.query.recall import dual_recall, match_entities
    from ..graphrag.query.rerank import rerank
    from ..graphrag.query.expand import expand

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        relation_defs = await load_relations(db)
        intent = query_router.route(payload.query, relation_defs)
        cond = nl2query(payload.query)
        yield _sse("meta", {"query": payload.query, "intent": intent, "cond": cond})
        try:
            async with Neo4jClient() as neo4j:
                milvus = AsyncMilvusClient()
                embeddings = EmbeddingClient()
                llm = LLMClient()

                scope = payload.scope_ids()
                entities = await neo4j.list_entities(kb_id=scope)
                matched_ids = match_entities(payload.query, entities)
                matched_names = [e["name"] for e in entities if e["entity_id"] in matched_ids]
                yield _sse("progress", {"step": "entity_match",
                                        "matched_entities": matched_names,
                                        "matched_ids": matched_ids})

                rec = await dual_recall(payload.query, milvus, embeddings,
                                        matched_ids, kb_id=scope)
                yield _sse("progress", {"step": "recall",
                                        "route_a": len(rec["a"]),
                                        "route_b": len(rec["b"])})

                hits = rerank(rec["a"], rec["b"], matched_ids=matched_ids)
                yield _sse("progress", {"step": "rerank", "top": len(hits)})

                expanded = []
                graph_facts = []
                extra_hits = []
                if intent == "graph" and matched_ids:
                    expanded = await expand(neo4j, matched_ids, kb_id=scope)
                    id2name = {e["entity_id"]: e["name"] for e in entities}
                    graph_facts = await query_router.collect_graph_facts(
                        neo4j, matched_ids, id2name, kb_id=scope, expanded=expanded)
                    from ..graphrag.query.recall import recall_expanded
                    extra_hits = await recall_expanded(payload.query, milvus, embeddings,
                                                       [e["entity_id"] for e in expanded], kb_id=scope)
                    hits = rerank(rec["a"] + extra_hits, rec["b"], matched_ids=matched_ids)
                    yield _sse("progress", {"step": "graph_expand",
                                            "expanded": len(expanded),
                                            "facts": graph_facts})

                from ..utils.rules import list_rules
                from ..graphrag.query.eval_rules import confidence_block, eval_stored_rules, rule_hints
                rule_rows = [
                    {"id": r.id, "name": r.name, "condition": r.condition, "action": r.action,
                     "enabled": r.enabled, "kb_id": str(r.kb_id) if r.kb_id else None}
                    for r in await list_rules(db)
                ]
                rule_hit_rows = eval_stored_rules(rule_rows, hits, kb_scope=scope)
                hints = rule_hints(rule_hit_rows)
                yield _sse("progress", {"step": "rules",
                                        "triggered": sum(1 for x in rule_hit_rows if x.get("triggered")),
                                        "total": len(rule_hit_rows)})

                from ..graphrag.query.iq_compare import compare_iq, resolve_iq_doc_id
                iq_doc = resolve_iq_doc_id(payload.query, payload.iq_doc_id)
                iq = await compare_iq(milvus, iq_doc) if iq_doc else None
                if iq:
                    yield _sse("progress", {"step": "iq_compare", "status": iq["status"],
                                            "similar": len(iq["similar_signals"])})
                conf = confidence_block(hits, rule_hit_rows, iq["signal_agree"] if iq else None,
                                        iq["agree_fields"] if iq else None)

                yield _sse("results", {"hits": hits,
                                       "matched_entities": matched_ids,
                                       "expanded_entities": expanded,
                                       "graph_facts": graph_facts,
                                       "rule_hits": rule_hit_rows,
                                       "confidence": conf,
                                       "iq_intent": "iq_compare" if iq else "none",
                                       "iq_status": iq["status"] if iq else None,
                                       "iq_query": iq["iq_query"] if iq else None,
                                       "similar_signals": iq["similar_signals"] if iq else [],
                                       "iq_charts": iq["iq_charts"] if iq else None,
                                       "sample_charts": iq["sample_charts"] if iq else None,
                                       "kb_ids": scope})
                contexts = build_contexts(hits)
                gen = llm.generate_answer_stream(payload.query, contexts, graph_facts, hints)
                sentinel = object()

                def _next():
                    try:
                        return next(gen)
                    except StopIteration:
                        return sentinel

                while True:
                    tok = await asyncio.to_thread(_next)
                    if tok is sentinel:
                        break
                    yield _sse("token", {"text": tok})

                yield _sse("done", {})
        except Exception as e:
            logger.exception("图谱流式查询失败: %s", payload.query)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/trace/files")
async def trace_files(kb_id: UUID | None = Query(None, description="按知识库过滤,不传 = 全库"),
                      db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:文件列表 + 文件级召回次数。"""
    from ..graphrag.trace import registry

    return {"files": await registry.list_files(db, kb_id=kb_id)}


@router.get("/trace/files/{doc_id}")
async def trace_file_detail(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:文件详情(召回历史时间线)。"""
    from ..graphrag.trace import registry

    return await registry.file_detail(db, doc_id)


@router.get("/trace/files/{doc_id}/locate")
async def trace_file_locate(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """溯源:源文件定位(存储路径)。"""
    from ..graphrag.trace import registry

    return await registry.locate(db, doc_id)
