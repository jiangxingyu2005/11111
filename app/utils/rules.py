"""预设推理规则数据访问层（CRUD，逻辑删除，SQLAlchemy 异步）。

与 kb.py 同构：所有函数接受 AsyncSession（接口层 Depends(get_db) 注入），
事务函数内部 commit；删除为逻辑删除（置 deleted_at），查询均过滤 deleted_at IS NULL。
规则主键是语义字符串（R_XXX），非 UUID，故 get/update/delete 直接按 str 取。
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Rule


async def create_rule(
    session: AsyncSession,
    *,
    id: str,
    name: str,
    description: str,
    condition: dict,
    action: dict,
    priority: int,
    enabled: bool,
    kb_id,
) -> Rule:
    """创建规则，返回完整记录。id 冲突时 sqlalchemy.exc.IntegrityError 冒泡。"""
    rule = Rule(
        id=id,
        name=name,
        description=description,
        condition=condition,
        action=action,
        priority=priority,
        enabled=enabled,
        kb_id=kb_id,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


async def list_rules(session: AsyncSession) -> list[Rule]:
    """未删除的规则列表，按 priority 降序（高优先级在前）、创建时间升序。"""
    stmt = (
        select(Rule)
        .where(Rule.deleted_at.is_(None))
        .order_by(Rule.priority.desc(), Rule.created_at)
    )
    return list(await session.scalars(stmt))


async def get_rule(session: AsyncSession, rule_id: str) -> Rule | None:
    """按语义主键查未删除的记录，不存在或已删除返回 None。"""
    stmt = select(Rule).where(
        Rule.id == rule_id,
        Rule.deleted_at.is_(None),
    )
    return await session.scalar(stmt)


async def update_rule(
    session: AsyncSession,
    rule_id: str,
    *,
    name=None,
    description=None,
    condition=None,
    action=None,
    priority=None,
    enabled=None,
    kb_id=None,
) -> Rule | None:
    """更新未删除记录的非 None 字段，updated_at 置 now()。不存在/已删除返回 None。"""
    rule = await get_rule(session, rule_id)
    if rule is None:
        return None
    if name is not None:
        rule.name = name
    if description is not None:
        rule.description = description
    if condition is not None:
        rule.condition = condition
    if action is not None:
        rule.action = action
    if priority is not None:
        rule.priority = priority
    if enabled is not None:
        rule.enabled = enabled
    if kb_id is not None:
        rule.kb_id = kb_id
    rule.updated_at = func.now()
    await session.commit()
    await session.refresh(rule)
    return rule


async def delete_rule(session: AsyncSession, rule_id: str) -> bool:
    """逻辑删除：置 deleted_at = now()。返回是否删到（仅未删除记录）。"""
    rule = await get_rule(session, rule_id)
    if rule is None:
        return False
    rule.deleted_at = func.now()
    rule.updated_at = func.now()
    await session.commit()
    return True
