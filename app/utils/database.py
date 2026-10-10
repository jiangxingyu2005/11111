"""SQLAlchemy 异步数据库核心:engine / session / Base / 建表初始化。

- engine / SessionLocal 全局单例(异步);
- Base 是所有 ORM 模型的基类(见 app/models/);
- get_db() 是 FastAPI 依赖,给每个请求一个异步 session;
- init_db() 按 ORM 模型建表(幂等)并启用 pgvector。
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
SessionLocal = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""
    pass


async def get_db():
    """FastAPI 依赖:每个请求一个异步 session,用完关闭。"""
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """初始化数据库表(幂等):按 ORM 模型建表 + 启用 pgvector。"""
    from .. import models  # noqa: F401  触发模型注册到 Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_vector_extension()


async def ensure_vector_extension() -> None:
    """确保 pgvector 扩展已启用(幂等,可重复调用)。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
    except Exception as e:
        logger.warning("pgvector 不可用，已跳过（向量在 Milvus）: %s", e)


async def ping() -> bool:
    """健康检查:能否连上并执行 SELECT 1。失败返回 False,不抛异常。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_engine():
    """当前 engine（诊断用）。"""
    return engine


async def reload_engine() -> bool:
    """在 POSTGRES_* 变更后按需重建 engine（opt-in，静默期调用）。

    关键：`SessionLocal.configure(bind=new_engine)` 原地 mutate 同一 sessionmaker，
    ~15 个 `from ...database import SessionLocal` 的 importer 无需任何改动。
    旧 engine dispose 后其连接池关闭；活跃查询不中断，仅归还时关闭池化连接。
    返回 True 表示已重建；DSN 未变则返回 False。
    """
    global engine
    if str(engine.url) == settings.postgres_dsn:
        return False
    new_engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
    old, engine = engine, new_engine
    SessionLocal.configure(bind=new_engine)  # 原地 mutate sessionmaker
    await old.dispose()
    return True
