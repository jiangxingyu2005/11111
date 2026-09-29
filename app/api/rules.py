"""预设推理规则 CRUD 接口（需求(7)/(8)：规则表增删改查）。"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import RuleCreate, RuleRead, RuleUpdate
from ..utils import rules
from ..utils.database import get_db

router = APIRouter(prefix="/rules", tags=["rules"])


@router.post("", response_model=RuleRead, status_code=status.HTTP_201_CREATED)
async def create_rule(payload: RuleCreate, db: AsyncSession = Depends(get_db)) -> RuleRead:
    try:
        return await rules.create_rule(
            db,
            id=payload.id,
            name=payload.name,
            description=payload.description,
            condition=payload.condition,
            action=payload.action,
            priority=payload.priority,
            enabled=payload.enabled,
            kb_id=payload.kb_id,
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"规则 id 已存在: {payload.id}")


@router.get("")
async def list_rules(db: AsyncSession = Depends(get_db)):
    """规则列表（未删除），按 priority 降序。"""
    rows = await rules.list_rules(db)
    return [RuleRead.model_validate(r).model_dump(mode="json") for r in rows]


@router.get("/{rule_id}", response_model=RuleRead)
async def get_rule(rule_id: str, db: AsyncSession = Depends(get_db)) -> RuleRead:
    row = await rules.get_rule(db, rule_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="规则不存在")
    return row


@router.put("/{rule_id}", response_model=RuleRead)
async def update_rule(rule_id: str, payload: RuleUpdate, db: AsyncSession = Depends(get_db)) -> RuleRead:
    row = await rules.update_rule(
        db,
        rule_id,
        name=payload.name,
        description=payload.description,
        condition=payload.condition,
        action=payload.action,
        priority=payload.priority,
        enabled=payload.enabled,
        kb_id=payload.kb_id,
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="规则不存在")
    return row


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(rule_id: str, db: AsyncSession = Depends(get_db)) -> None:
    if not await rules.delete_rule(db, rule_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="规则不存在")
