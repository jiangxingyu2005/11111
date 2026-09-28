"""GraphRAG 异步数据访问层(替换 ems 同步 common/db.py)。

基于 SQLAlchemy async:Core insert(pg_insert) + on_conflict 幂等 upsert / 纯 insert / 条件查询。
- 客户端标量 default(title/enabled/recall_count 等)在 insert 时自动补齐(server_default 由 DB 补);
- upsert 的 UPDATE 只覆盖调用方显式传入的列,避免重跑时覆盖 enabled/recall_count 等运行态字段。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession


def _with_defaults(model: Any, data: dict) -> dict:
    """把模型列的客户端标量 default 补进 insert 数据(未显式提供时;server_default 交给 DB)。"""
    out = dict(data)
    for col in model.__table__.columns:
        if col.name in out or col.server_default is not None:
            continue
        default = col.default
        if default is not None and getattr(default, "is_scalar", False):
            out[col.name] = default.arg
    return out


async def insert(session: AsyncSession, model: Any, data: dict) -> None:
    """纯插入(自增主键表用),补齐客户端标量默认值。"""
    await session.execute(pg_insert(model).values(**_with_defaults(model, data)))


async def upsert(session: AsyncSession, model: Any, data: dict, id_col: str) -> None:
    """幂等 upsert:插入补齐默认值;冲突时仅更新调用方显式提供的列。"""
    stmt = pg_insert(model).values(**_with_defaults(model, data))
    update_cols = {k: stmt.excluded[k] for k in data if k != id_col}
    stmt = stmt.on_conflict_do_update(index_elements=[id_col], set_=update_cols)
    await session.execute(stmt)


async def update(session: AsyncSession, model: Any, data: dict, id_col: str) -> None:
    """按主键更新指定列(行须已存在)。不补默认值、不插新行,用于回写运行态字段。"""
    stmt = (
        sa_update(model)
        .where(getattr(model, id_col) == data[id_col])
        .values(**{k: v for k, v in data.items() if k != id_col})
    )
    await session.execute(stmt)


async def delete(session: AsyncSession, model: Any, id_val: Any, id_col: str) -> None:
    """按主键删行(供分段编辑删子块等)。行不存在时不报错。"""
    stmt = sa_delete(model).where(getattr(model, id_col) == id_val)
    await session.execute(stmt)


async def increment(session: AsyncSession, model: Any, id_col: str,
                    id_val: Any, col: str, delta: int = 1) -> None:
    """原子自增某列(如 recall_count +1),行不存在时不动(用于溯源计数)。"""
    col_attr = getattr(model, col)
    stmt = (
        sa_update(model)
        .where(getattr(model, id_col) == id_val)
        .values({col: col_attr + delta})
    )
    await session.execute(stmt)


def _row_to_dict(row: Any, cols: list[str]) -> dict:
    return {c: getattr(row, c) for c in cols}


async def select_where(session: AsyncSession, model: Any, **filters: Any) -> list[dict]:
    """条件查询,返回 list[dict](列名 -> 值)。"""
    stmt = select(model).where(*[getattr(model, k) == v for k, v in filters.items()])
    res = await session.execute(stmt)
    cols = [c.name for c in model.__table__.columns]
    return [_row_to_dict(r, cols) for r in res.scalars().all()]


async def select_where_in(session: AsyncSession, model: Any, col: str, values: list[Any]) -> list[dict]:
    """IN 查询(如按 parent_id 集合取子块)。"""
    stmt = select(model).where(getattr(model, col).in_(values))
    res = await session.execute(stmt)
    cols = [c.name for c in model.__table__.columns]
    return [_row_to_dict(r, cols) for r in res.scalars().all()]


async def select_all(session: AsyncSession, model: Any) -> list[dict]:
    return await select_where(session, model)
