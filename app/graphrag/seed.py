"""抽取字典种子数据:10 实体 + 11 关系(领域模型重新固定)。

字典为全局共享(不分知识库);抽取侧每次现查,改字典即时生效。
seed_extraction_dict 幂等:表空时整体写入;旧库按「种子全集」同步(补齐/刷新/软删)。
关系带 head_types/tail_types 两端实体类型约束(domain/range),抽取时校验方向。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ExtractionDict

# (name, en, description, semantic_mergeable):末位标记该类型是否允许向量语义层做同义异名归并。
# 名词性(辐射源/平台/组织/目标/事件/情报产品)=True 进 ems_entities;
# 数据对象/枚举(原始信号/信号样本/信号簇/工作状态)=False,只走确定性匹配,
# 避免「原始信号A↔原始信号B」这类语义相近但非同一实体的误并。
_ENTITIES = [
    ("原始信号", "RawSignal", "未经处理的原始电磁信号/波形数据", False),
    ("信号样本", "SignalSample", "采集到的信号样本/采样数据", False),
    ("信号簇", "SignalCluster", "聚类形成的信号簇/信号集合", False),
    ("辐射源", "RadiationSource", "电磁辐射的发射源/信号源", True),
    ("平台", "Platform", "搭载装备的载体(舰艇/飞机/地面站/车辆等)", True),
    ("组织", "Organization", "军队/单位/机构等组织实体", True),
    ("工作状态", "WorkState", "装备/系统的工作状态(待机/工作/关机等)", False),
    ("目标", "Target", "侦察/打击/跟踪的目标对象", True),
    ("事件", "Event", "作战/演习/侦察等事件", True),
    ("情报产品", "IntelProduct", "情报报告/情报产品", True),
]

# (name, en, description, trigger_words, min_confidence, head_types, tail_types):
# en 为合法 Cypher 关系标识(全大写);head/tail_types 用中文实体类型名,约束该关系两端
# 能连哪类节点(空 = 不约束)。关系即「本体」:只允许清单内类型、且方向受此表约束,不能随便连边。
# 注:实体清单无「装备/型号」类型,装备于/型号匹配 两端用辐射源(=发射装备)与平台承载。
_RELATIONS = [
    ("来源", "SOURCE_OF", "信号/样本的来源关系",
     ["来源", "来自", "源自"], 0.75,
     ["原始信号", "信号样本", "信号簇", "情报产品", "事件"],
     ["辐射源", "平台", "组织", "事件", "原始信号", "信号样本"]),
    ("组成", "COMPOSED_OF", "集合/平台的组成关系",
     ["组成", "构成", "包含", "包括"], 0.75,
     ["信号簇", "平台", "组织", "事件", "情报产品"],
     ["原始信号", "信号样本", "信号簇", "辐射源", "平台", "组织"]),
    ("归并为", "MERGED_INTO", "信号/样本归并/聚类为簇的关系",
     ["归并", "合并", "聚为", "聚成", "聚类"], 0.75,
     ["原始信号", "信号样本", "信号簇"],
     ["信号簇"]),
    ("型号匹配", "MODEL_MATCH", "信号/辐射源与平台型号的匹配关系",
     ["型号匹配", "匹配", "对应", "识别为"], 0.75,
     ["辐射源", "原始信号", "信号样本", "信号簇"],
     ["平台", "辐射源"]),
    ("装备于", "EQUIPPED_ON", "辐射源搭载/部署于平台的关系",
     ["装备于", "搭载", "安装于", "部署于"], 0.75,
     ["辐射源", "平台", "信号样本"],
     ["平台", "组织"]),
    ("隶属", "BELONGS_TO", "单元/组织之间的隶属、配属、下辖、编成关系",
     ["隶属", "属于", "下辖", "配属", "归属", "编成", "辖"], 0.75,
     ["平台", "组织", "辐射源"],
     ["组织", "平台"]),
    ("具有状态", "HAS_STATE", "辐射源/平台/目标具有某种工作状态",
     ["具有状态", "处于", "状态为", "工作状态"], 0.75,
     ["辐射源", "平台", "目标", "组织"],
     ["工作状态"]),
    ("关联目标", "RELATED_TARGET", "辐射源/平台/事件与目标的关联关系",
     ["关联目标", "针对", "指向", "跟踪", "打击"], 0.75,
     ["辐射源", "平台", "事件", "信号簇"],
     ["目标"]),
    ("平台候选", "PLATFORM_CANDIDATE", "信号/辐射源可能的平台候选关系",
     ["平台候选", "候选", "可能为", "疑似"], 0.75,
     ["辐射源", "原始信号", "信号样本", "信号簇"],
     ["平台"]),
    ("产生", "PRODUCES", "辐射源/平台产生信号/情报",
     ["产生", "生成", "输出", "发射"], 0.75,
     ["辐射源", "平台", "组织", "事件"],
     ["原始信号", "信号样本", "信号簇", "情报产品"]),
    ("写入", "WRITTEN_INTO", "事件/情报写入到情报产品",
     ["写入", "记录于", "输出到", "汇入"], 0.75,
     ["事件", "情报产品", "辐射源", "平台", "原始信号", "信号样本", "信号簇"],
     ["情报产品"]),
]


async def _ensure_mergeable_column(session: AsyncSession) -> bool:
    """旧库补 semantic_mergeable 列(create_all 不给已存在的表加列)。返回是否本次新增。"""
    exists = await session.scalar(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'extraction_dict' AND column_name = 'semantic_mergeable'"
    ))
    if exists:
        return False
    await session.execute(text(
        "ALTER TABLE extraction_dict "
        "ADD COLUMN semantic_mergeable BOOLEAN NOT NULL DEFAULT FALSE"
    ))
    await session.commit()
    return True


async def _ensure_relation_type_columns(session: AsyncSession) -> bool:
    """旧库补 head_types/tail_types 列(关系两端实体类型约束;create_all 不给已有表加列)。"""
    rows = (await session.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'extraction_dict' AND column_name IN ('head_types', 'tail_types')"
    ))).scalars().all()
    existing = set(rows)
    added = False
    for col in ("head_types", "tail_types"):
        if col not in existing:
            await session.execute(text(f"ALTER TABLE extraction_dict ADD COLUMN {col} JSONB"))
            added = True
    if added:
        await session.commit()
    return added


async def _sync_seed(session: AsyncSession) -> int:
    """把「种子全集」同步到 extraction_dict:补齐新增 / 刷新字段 / 软删旧类型。幂等。

    seed 只在表空时整体写入;旧库靠这里以当前 _ENTITIES/_RELATIONS 为唯一标准:
    不在全集里的旧类型软删(deleted_at,可逆;硬删会破坏 entity_registry 等引用)。
    """
    changed = 0

    async def _row(name: str, category: str):
        return await session.scalar(
            select(ExtractionDict).where(
                ExtractionDict.category == category,
                ExtractionDict.name == name,
            )
        )

    for name, en, desc, mergeable in _ENTITIES:
        row = await _row(name, "entity")
        if row is None:
            session.add(ExtractionDict(name=name, en=en, category="entity",
                                       description=desc, semantic_mergeable=mergeable,
                                       enabled=True))
            changed += 1
        elif row.deleted_at is not None or (row.en or "") != en \
                or (row.description or "") != desc \
                or bool(row.semantic_mergeable) != mergeable or not bool(row.enabled):
            row.deleted_at = None
            row.en = en
            row.description = desc
            row.semantic_mergeable = mergeable
            row.enabled = True
            changed += 1

    for name, en, desc, triggers, min_conf, head_types, tail_types in _RELATIONS:
        row = await _row(name, "relation")
        if row is None:
            session.add(ExtractionDict(name=name, en=en, category="relation",
                                       description=desc, trigger_words=list(triggers),
                                       min_confidence=min_conf, enabled=True,
                                       head_types=list(head_types), tail_types=list(tail_types)))
            changed += 1
        elif row.deleted_at is not None or (row.en or "") != en \
                or (row.description or "") != desc \
                or list(row.trigger_words or []) != list(triggers) \
                or list(row.head_types or []) != list(head_types) \
                or list(row.tail_types or []) != list(tail_types) \
                or not bool(row.enabled):
            row.deleted_at = None
            row.en = en
            row.description = desc
            row.trigger_words = list(triggers)
            row.head_types = list(head_types)
            row.tail_types = list(tail_types)
            row.enabled = True
            changed += 1

    entity_names = {e[0] for e in _ENTITIES}
    relation_names = {r[0] for r in _RELATIONS}
    now = datetime.now(timezone.utc)
    rows = (await session.execute(select(ExtractionDict))).scalars().all()
    for r in rows:
        keep = r.name in (entity_names if r.category == "entity" else relation_names)
        if r.deleted_at is None and not keep:
            r.deleted_at = now
            changed += 1

    await session.commit()
    return changed


async def seed_extraction_dict(session: AsyncSession) -> int:
    """表空时整体写入种子;旧库补列后按种子全集同步(补齐/刷新/软删)。幂等。"""
    await _ensure_mergeable_column(session)
    await _ensure_relation_type_columns(session)

    count = await session.scalar(select(func.count()).select_from(ExtractionDict))
    if count == 0:
        for name, en, desc, mergeable in _ENTITIES:
            session.add(ExtractionDict(name=name, en=en, category="entity",
                                       description=desc, semantic_mergeable=mergeable))
        for name, en, desc, triggers, min_conf, head_types, tail_types in _RELATIONS:
            session.add(ExtractionDict(name=name, en=en, category="relation", description=desc,
                                       trigger_words=list(triggers), min_confidence=min_conf,
                                       head_types=list(head_types), tail_types=list(tail_types)))
        await session.commit()
        return len(_ENTITIES) + len(_RELATIONS)

    return await _sync_seed(session)
