"""Neo4j 直接查询/管理接口(Phase 5):查节点/关系、按知识库过滤、增删改。

实体(:Entity)全局共享、不分知识库;归属知识库靠 BINDS 边关联的 :Segment.kb_id 判定。
关系边的知识库归属靠 r.source_parent_id(抽取来源父块)判定。

全部 async def;只新增,不动既有接口。Neo4j 不可用时返回 502,不影响主链路。
"""
from __future__ import annotations

import json
import logging
import re
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..graphrag.ids import canonical_entity_id, normalize_name
from ..graphrag.models import Entity
from ..graphrag.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph-neo4j"])

_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_rel_type(rel_type: str) -> None:
    """关系类型须是合法 Cypher 关系标识,防止 Cypher 注入。"""
    if not _REL_TYPE_RE.fullmatch(rel_type):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"rel_type 必须是合法标识符(字母/下划线开头,仅字母数字下划线),收到: {rel_type!r}",
        )


def _aliases(v) -> list:
    return v if isinstance(v, list) else []


def _props(v) -> dict:
    if isinstance(v, str):
        try:
            return json.loads(v) if v else {}
        except json.JSONDecodeError:
            return {}
    return v if isinstance(v, dict) else {}


def _entity_light(r: dict) -> dict:
    return {
        "entity_id": r.get("entity_id"),
        "name": r.get("name"),
        "entity_type": r.get("entity_type"),
        "aliases": _aliases(r.get("aliases")),
        "confidence": r.get("confidence"),
    }


def _entity_full(r: dict) -> dict:
    return {
        **_entity_light(r),
        "properties": _props(r.get("properties")),
    }


# ---- 请求模型 ----
class EntityCreate(BaseModel):
    entity_id: str | None = None   # 不传则用 canonical_entity_id(name, entity_type) 自动生成
    name: str
    entity_type: str
    aliases: list[str] = []
    properties: dict = {}
    confidence: float = 1.0


class EntityUpdate(BaseModel):
    name: str | None = None
    entity_type: str | None = None
    aliases: list[str] | None = None
    properties: dict | None = None
    confidence: float | None = None


class SegmentCreate(BaseModel):
    parent_id: str
    doc_id: str = ""
    kb_id: UUID | None = None
    title: str = ""
    enabled: bool = True
    chapter_no: str = ""


class SegmentUpdate(BaseModel):
    doc_id: str | None = None
    kb_id: UUID | None = None
    title: str | None = None
    enabled: bool | None = None
    chapter_no: str | None = None


class RelationCreate(BaseModel):
    head_entity_id: str
    tail_entity_id: str
    rel_type: str
    confidence: float = 0.0
    evidence: str = ""
    source_parent_id: str | None = None
    derived: bool = False


class BindCreate(BaseModel):
    entity_id: str
    parent_id: str
    kg_role: str = "supplemental"
    kg_confidence: float = 0.5


# ---- 节点总览 ----
@router.get("/nodes")
async def graph_nodes(kb_id: UUID | None = Query(None, description="按知识库过滤,不传 = 全库")) -> dict:
    """全量节点概览:实体(:Entity) + 父块(:Segment),可按知识库过滤。"""
    try:
        async with Neo4jClient() as neo4j:
            entities = await neo4j.list_entity_nodes(kb_id=str(kb_id) if kb_id else None)
            segments = await neo4j.list_segment_nodes(kb_id=str(kb_id) if kb_id else None)
    except Exception as e:
        logger.exception("读取图节点失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"读取图节点失败: {e}")
    return {
        "entities": [_entity_light(r) for r in entities],
        "segments": segments,
        "totals": {"entities": len(entities), "segments": len(segments)},
    }


# ---- 实体节点 ----
@router.get("/entities")
async def list_graph_entities(
    name: str | None = Query(None, description="按名称/别名模糊匹配"),
    entity_type: str | None = Query(None, description="按实体类型过滤"),
    kb_id: UUID | None = Query(None, description="按知识库过滤(实体归属经 BINDS 判定)"),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """列出实体节点。"""
    try:
        async with Neo4jClient() as neo4j:
            rows = await neo4j.list_entity_nodes(
                name=name, entity_type=entity_type,
                kb_id=str(kb_id) if kb_id else None, limit=limit, offset=offset)
    except Exception as e:
        logger.exception("列出实体失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"列出实体失败: {e}")
    return {"entities": [_entity_light(r) for r in rows], "total": len(rows)}


@router.get("/entities/{entity_id}")
async def get_graph_entity(entity_id: str) -> dict:
    """单个实体详情(含绑定父块 + 出入关系边)。不存在返回 404。"""
    try:
        async with Neo4jClient() as neo4j:
            node = await neo4j.get_entity_node(entity_id)
            if node is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="实体不存在")
            binds = await neo4j.list_binds_full(entity_id=entity_id)
            out = await neo4j.list_relations_full(head_entity_id=entity_id)
            inn = await neo4j.list_relations_full(tail_entity_id=entity_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("读取实体详情失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"读取实体详情失败: {e}")
    return {**_entity_full(node), "binds": binds,
            "relations": {"out": out, "in": inn}}


async def _sync_registry_and_vector(entity_id: str, name: str, entity_type: str,
                                     aliases: list, properties: dict):
    """前端增实体时同步写 PG entity_registry + Milvus ems_entities(消歧/向量检索可用)。"""
    from ..graphrag import db
    from ..graphrag.embeddings import EmbeddingClient
    from ..graphrag.mesh.registry import fingerprint
    from ..graphrag.milvus_client import AsyncMilvusClient
    from ..models import EntityRegistryEntry
    from ..utils.database import SessionLocal

    async with SessionLocal() as session:
        key = normalize_name(name)
        alias_keys = [normalize_name(a) for a in (aliases or [])]
        fp = fingerprint(properties)
        existing = await db.select_where(session, EntityRegistryEntry, entity_id=entity_id)
        if existing:
            prev = existing[0]
            merged = list(prev.get("aliases") or [])
            for a in [key] + alias_keys:
                if a and a not in merged:
                    merged.append(a)
            props = dict(prev.get("properties") or {})
            for k, v in (properties or {}).items():
                props.setdefault(k, v)
            await db.upsert(session, EntityRegistryEntry, {
                "entity_id": entity_id, "name": prev.get("name") or key,
                "entity_type": entity_type, "aliases": merged,
                "fingerprint": fp or prev.get("fingerprint"),
                "properties": props, "status": "active"}, "entity_id")
        else:
            await db.upsert(session, EntityRegistryEntry, {
                "entity_id": entity_id, "name": key, "entity_type": entity_type,
                "aliases": list(dict.fromkeys([key] + alias_keys)),
                "fingerprint": fp, "properties": dict(properties or {}),
                "status": "active"}, "entity_id")
        await session.commit()
    try:
        milvus = AsyncMilvusClient()
        vec = EmbeddingClient().embed_one(name)
        await milvus.upsert_entity_vector(entity_id, name, entity_type, vec)
    except Exception:
        logger.warning("实体向量写入失败(entity_id=%s),不影响 Neo4j/PG", entity_id)


async def _remove_registry_and_vector(entity_id: str):
    """前端删实体时同步清 PG entity_registry + Milvus ems_entities。"""
    from ..graphrag import db
    from ..graphrag.milvus_client import AsyncMilvusClient
    from ..models import EntityRegistryEntry
    from ..utils.database import SessionLocal

    async with SessionLocal() as session:
        await db.delete(session, EntityRegistryEntry, entity_id, "entity_id")
        await session.commit()
    try:
        await AsyncMilvusClient().delete_entity_vector(entity_id)
    except Exception:
        logger.warning("实体向量删除失败(entity_id=%s)", entity_id)


@router.post("/entities", status_code=status.HTTP_201_CREATED)
async def create_graph_entity(body: EntityCreate) -> dict:
    """新建/合并实体节点(MERGE 幂等)。同步写 PG 注册表 + Milvus 实体向量。"""
    eid = body.entity_id or canonical_entity_id(body.name, body.entity_type)
    data = {**body.model_dump(), "entity_id": eid}
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.merge_entity(data)
            node = await neo4j.get_entity_node(eid)
    except Exception as e:
        logger.exception("创建实体失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"创建实体失败: {e}")
    await _sync_registry_and_vector(eid, body.name, body.entity_type,
                                     body.aliases, body.properties)
    return _entity_full(node)


@router.put("/entities/{entity_id}")
async def update_graph_entity(entity_id: str, body: EntityUpdate) -> dict:
    """部分更新实体(仅传要改的字段)。不存在返回 404。"""
    fields = body.model_dump(exclude_unset=True)
    try:
        async with Neo4jClient() as neo4j:
            if await neo4j.get_entity_node(entity_id) is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="实体不存在")
            await neo4j.update_entity(entity_id, fields)
            node = await neo4j.get_entity_node(entity_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("更新实体失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"更新实体失败: {e}")
    return _entity_full(node)


@router.delete("/entities/{entity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph_entity(entity_id: str) -> None:
    """删除实体节点(DETACH DELETE,连带 BINDS/关系边)。同步清 PG 注册表 + Milvus 向量。"""
    try:
        async with Neo4jClient() as neo4j:
            if await neo4j.get_entity_node(entity_id) is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="实体不存在")
            await neo4j.delete_entity(entity_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("删除实体失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"删除实体失败: {e}")
    await _remove_registry_and_vector(entity_id)


# ---- 父块 Segment 节点(Neo4j 侧) ----
@router.get("/nodes/segments")
async def list_graph_segment_nodes(
    kb_id: UUID | None = Query(None),
    doc_id: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """列出 :Segment 父块节点(Neo4j 侧)。"""
    try:
        async with Neo4jClient() as neo4j:
            rows = await neo4j.list_segment_nodes(
                kb_id=str(kb_id) if kb_id else None, doc_id=doc_id,
                limit=limit, offset=offset)
    except Exception as e:
        logger.exception("列出 Segment 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"列出 Segment 失败: {e}")
    return {"segments": rows, "total": len(rows)}


@router.get("/nodes/segments/{parent_id}")
async def get_graph_segment_node(parent_id: str) -> dict:
    """单个 :Segment 节点详情。不存在返回 404。"""
    try:
        async with Neo4jClient() as neo4j:
            node = await neo4j.get_segment_node(parent_id)
    except Exception as e:
        logger.exception("读取 Segment 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"读取 Segment 失败: {e}")
    if node is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment 不存在")
    return node


@router.post("/nodes/segments", status_code=status.HTTP_201_CREATED)
async def create_graph_segment_node(body: SegmentCreate) -> dict:
    """新建/合并 :Segment 节点(MERGE 幂等)。"""
    parent = body.model_dump()
    parent["kb_id"] = str(parent["kb_id"]) if parent.get("kb_id") else None
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.merge_segment(parent)
            node = await neo4j.get_segment_node(body.parent_id)
    except Exception as e:
        logger.exception("创建 Segment 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"创建 Segment 失败: {e}")
    return node


@router.put("/nodes/segments/{parent_id}")
async def update_graph_segment_node(parent_id: str, body: SegmentUpdate) -> dict:
    """部分更新 :Segment 节点。不存在返回 404。"""
    fields = body.model_dump(exclude_unset=True)
    try:
        async with Neo4jClient() as neo4j:
            if await neo4j.get_segment_node(parent_id) is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment 不存在")
            await neo4j.update_segment(parent_id, fields)
            node = await neo4j.get_segment_node(parent_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("更新 Segment 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"更新 Segment 失败: {e}")
    return node


@router.delete("/nodes/segments/{parent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph_segment_node(parent_id: str) -> None:
    """删除 :Segment 节点(DETACH DELETE,连带 BINDS 边)。不存在返回 404。"""
    try:
        async with Neo4jClient() as neo4j:
            if await neo4j.get_segment_node(parent_id) is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment 不存在")
            await neo4j.detach_delete_segment(parent_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("删除 Segment 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"删除 Segment 失败: {e}")


# ---- 关系边(实体↔实体) ----
@router.get("/relations")
async def list_graph_relations(
    rel_type: str | None = Query(None, description="按关系类型(英文标识)过滤"),
    kb_id: UUID | None = Query(None, description="按知识库过滤(source_parent_id 归属该库)"),
    head_entity_id: str | None = Query(None),
    tail_entity_id: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """列出关系边(不含 BINDS)。"""
    try:
        async with Neo4jClient() as neo4j:
            rows = await neo4j.list_relations_full(
                rel_type=rel_type, kb_id=str(kb_id) if kb_id else None,
                head_entity_id=head_entity_id, tail_entity_id=tail_entity_id,
                limit=limit, offset=offset)
    except Exception as e:
        logger.exception("列出关系失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"列出关系失败: {e}")
    return {"relations": rows, "total": len(rows)}


@router.post("/relations", status_code=status.HTTP_201_CREATED)
async def create_graph_relation(body: RelationCreate) -> dict:
    """新建/合并关系边(MERGE 幂等;rel_type 须为字典已定义的合法标识)。"""
    _require_rel_type(body.rel_type)
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.merge_relation(body.model_dump())
    except Exception as e:
        logger.exception("创建关系失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"创建关系失败: {e}")
    return body.model_dump()


@router.delete("/relations", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph_relation(
    head_entity_id: str = Query(...),
    tail_entity_id: str = Query(...),
    rel_type: str = Query(...),
) -> None:
    """删除指定 (head, tail, rel_type) 的关系边。"""
    _require_rel_type(rel_type)
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.delete_relation(head_entity_id, tail_entity_id, rel_type)
    except Exception as e:
        logger.exception("删除关系失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"删除关系失败: {e}")


# ---- BINDS 边 ----
@router.get("/binds")
async def list_graph_binds(
    kb_id: UUID | None = Query(None),
    entity_id: str | None = Query(None),
    parent_id: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """列出 BINDS 边(实体↔父块)。"""
    try:
        async with Neo4jClient() as neo4j:
            rows = await neo4j.list_binds_full(
                kb_id=str(kb_id) if kb_id else None,
                entity_id=entity_id, parent_id=parent_id, limit=limit, offset=offset)
    except Exception as e:
        logger.exception("列出 BINDS 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"列出 BINDS 失败: {e}")
    return {"binds": rows, "total": len(rows)}


@router.post("/binds", status_code=status.HTTP_201_CREATED)
async def create_graph_bind(body: BindCreate) -> dict:
    """新建/合并 BINDS 边(MERGE 幂等)。"""
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.merge_binds(body.entity_id, body.parent_id,
                                    body.kg_role, body.kg_confidence)
    except Exception as e:
        logger.exception("创建 BINDS 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"创建 BINDS 失败: {e}")
    return body.model_dump()


@router.delete("/binds", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph_bind(
    entity_id: str = Query(...),
    parent_id: str = Query(...),
) -> None:
    """删除指定 (entity_id, parent_id) 的 BINDS 边。"""
    try:
        async with Neo4jClient() as neo4j:
            await neo4j.delete_bind(entity_id, parent_id)
    except Exception as e:
        logger.exception("删除 BINDS 失败")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"删除 BINDS 失败: {e}")
