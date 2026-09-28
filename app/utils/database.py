"""SQLAlchemy 异步数据库核心:engine / session / Base / 建表初始化。

- engine / SessionLocal 全局单例(异步);
- Base 是所有 ORM 模型的基类(见 app/models/);
- get_db() 是 FastAPI 依赖,给每个请求一个异步 session;
- init_db() 按 ORM 模型建表(幂等)并启用 pgvector。
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..config import settings

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
    async with engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()


async def ping() -> bool:
    """健康检查:能否连上并执行 SELECT 1。失败返回 False,不抛异常。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
