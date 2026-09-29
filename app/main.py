import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import documents, graph, iq, knowledge_bases, mesh, neo4j, parse, seg
from .config import settings
from .utils.database import SessionLocal, init_db

logger = logging.getLogger(__name__)


async def _bootstrap_graphrag() -> None:
    """GraphRAG 启动装配:字典种子 + Milvus collection + Neo4j 约束(全部容错)。

    三库连不上不阻断启动,仅告警 —— parse 主链路照常;ingest 时若连不上会失败并被
    _finalize 捕获记日志(documents.in_kb 保持 False)。
    """
    if not settings.graphrag_enabled:
        return

    # 1. 抽取字典种子(幂等:表非空跳过)
    try:
        from .graphrag.seed import seed_extraction_dict

        async with SessionLocal() as db:
            await seed_extraction_dict(db)
    except Exception as e:
        logger.warning("抽取字典种子写入失败: %s", e)

    # 2. Milvus collection(不存在则建,pymilvus 同步调用经线程池;子块 + 实体名双库)
    try:
        from .graphrag.milvus_client import AsyncMilvusClient

        milvus = AsyncMilvusClient()
        await milvus.create_collection()
        await milvus.create_entity_collection()
    except Exception as e:
        logger.warning("Milvus 初始化失败(不影响 parse 主链路): %s", e)

    # 3. Neo4j 约束(IF NOT EXISTS,幂等)
    try:
        from .graphrag.neo4j_client import Neo4jClient

        async with Neo4jClient() as neo4j:
            await neo4j.run_constraints()
    except Exception as e:
        logger.warning("Neo4j 约束初始化失败(不影响 parse 主链路): %s", e)


async def _bootstrap_iq() -> None:
    """IQ 检索库集合(ems_iq)初始化,独立于 GraphRAG;连不上不阻断启动。"""
    try:
        from .graphrag.milvus_client import AsyncMilvusClient

        await AsyncMilvusClient().create_iq_collection()
    except Exception as e:
        logger.warning("IQ 检索库初始化失败(不影响启动): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await _bootstrap_graphrag()
    await _bootstrap_iq()
    yield


app = FastAPI(title="文档解析清洗服务", version="0.1.0", lifespan=lifespan)

app.include_router(parse.router)
app.include_router(knowledge_bases.router)
app.include_router(documents.router)
app.include_router(graph.router)
app.include_router(seg.router)
app.include_router(mesh.router)
app.include_router(neo4j.router)
app.include_router(iq.router)
