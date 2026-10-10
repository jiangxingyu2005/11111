"""抽取字典种子数据:11 实体 + 14 关系 + 3 派生关系(按需求文档固定的领域本体)。

需求依据:
  (5)  向量与「辐射源型号节点」绑定 -> 辐射源/平台带 粒度(型号/个体)属性
  (7)  关联「辐射源型号、所属平台、历史事件」,频段重合触发规则 -> 涉及关系 + 同频段派生边
  (10) 抽取「属性(频率、脉宽、重频)」与「隶属、支援」等关系 -> 参数是属性不是实体;补 支援/交联
字典为全局共享(不分知识库);抽取侧每次现查,改字典即时生效。
seed_extraction_dict 幂等:按「种子全集」同步(补齐/刷新/软删)。
关系带 head_types/tail_types 两端实体类型约束(domain/range),抽取时校验方向。
派生关系(category=derived)由规则从属性计算,不进 LLM 抽取提示词。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ExtractionDict
from .mesh.derived import DERIVED_RELATIONS


def _props(*fields: str, enums: dict | None = None) -> dict:
    """实体属性 JSON Schema:字段全部可选字符串,enums 给枚举约束。"""
    enums = enums or {}
    properties = {}
    for f in fields:
        properties[f] = {"type": "string", "enum": enums[f]} if f in enums else {"type": "string"}
    return {"type": "object", "properties": properties}


# 信号参数:需求(10)明确为「属性」,写进所属实体 properties,不单独成实体
_SIGNAL_PARAMS = ("频率", "频段", "调制方式", "脉宽", "重频", "带宽")
_GRAIN = {"粒度": ["型号", "个体"]}

# (name, en, description, semantic_mergeable, json_schema):
# semantic_mergeable 标记该类型是否允许向量语义层做同义异名归并。
# 名词性(辐射源/平台/组织/目标/事件/情报产品/武器)=True 进 ems_entities;
# 数据对象/枚举(原始信号/信号样本/信号簇/工作状态)=False,只走确定性匹配,
# 避免「原始信号A↔原始信号B」这类语义相近但非同一实体的误并。
_ENTITIES = [
    ("原始信号", "RawSignal",
     "未经处理的原始电磁信号/波形(如某跳频信号、某干扰信号);单独的频率/调制数值不是信号实体",
     False, _props(*_SIGNAL_PARAMS)),
    ("信号样本", "SignalSample",
     "采集到的某段信号样本/采样数据记录;单独的频率数值(如 850MHz)不是样本,应作为属性",
     False, _props(*_SIGNAL_PARAMS, "采样率", "时间")),
    ("信号簇", "SignalCluster", "聚类形成的信号簇/信号集合",
     False, _props(*_SIGNAL_PARAMS)),
    ("辐射源", "RadiationSource",
     "发射电磁信号的装备(雷达/电台/干扰机/侦察设备等)。粒度=型号 指某型装备(如某型雷达),"
     "粒度=个体 指具体某一部装备",
     True, _props("粒度", "型号", "国别", *_SIGNAL_PARAMS, enums=_GRAIN)),
    ("平台", "Platform",
     "搭载装备的载体(舰艇/飞机/地面站/车辆/指挥所等)。粒度=型号 指某型平台(如某型飞机),"
     "粒度=个体 指具体某一艘/架/处",
     True, _props("粒度", "型号", "国别", "区域", enums=_GRAIN)),
    ("组织", "Organization", "军队/单位/机构/制造商等组织实体",
     True, _props("国别")),
    ("工作状态", "WorkState", "装备/系统的工作状态(待机/工作/关机/搜索/跟踪等)",
     False, None),
    ("目标", "Target", "侦察/打击/跟踪的目标对象",
     True, _props("区域")),
    ("事件", "Event", "作战/演习/侦察/历史活动等事件",
     True, _props("时间", "区域")),
    ("情报产品", "IntelProduct", "情报报告/情报产品",
     True, _props("时间")),
    ("武器", "Weapon",
     "导弹/鱼雷/炮弹/火箭弹等打击武器;不以发射电磁信号为主,区别于辐射源;发射车/舰艇是平台",
     True, _props("粒度", "型号", "国别", "射程", "制导方式", enums=_GRAIN)),
]

# (name, en, description, trigger_words, min_confidence, head_types, tail_types):
# en 为合法 Cypher 关系标识(全大写);head/tail_types 用中文实体类型名,约束该关系两端
# 能连哪类节点(空 = 不约束)。关系即「本体」:只允许清单内类型、且方向受此表约束,不能随便连边。
# 注:实体清单无「装备/型号」类型,装备于/型号匹配 两端用辐射源(=发射装备)与平台承载。
_RELATIONS = [
    ("来源", "SOURCE_OF", "信号/样本的来源关系(信号→发出它的辐射源/平台)",
     ["来源", "来自", "源自"], 0.75,
     ["原始信号", "信号样本", "信号簇", "情报产品", "事件"],
     ["辐射源", "平台", "组织", "事件", "原始信号", "信号样本"]),
    ("组成", "COMPOSED_OF", "集合/平台的组成关系(整体→部分)",
     ["组成", "构成", "包含", "包括"], 0.75,
     ["信号簇", "平台", "组织", "事件", "情报产品"],
     ["原始信号", "信号样本", "信号簇", "辐射源", "平台", "组织", "武器"]),
    ("归并为", "MERGED_INTO", "信号/样本归并/聚类为簇的关系",
     ["归并", "合并", "聚为", "聚成", "聚类"], 0.75,
     ["原始信号", "信号样本", "信号簇"],
     ["信号簇"]),
    ("型号匹配", "MODEL_MATCH", "个体/信号与型号节点的匹配关系(个体或信号→型号),需求(5)(6)的型号识别落点",
     ["型号匹配", "匹配", "对应", "识别为", "属于型号"], 0.75,
     ["辐射源", "平台", "武器", "原始信号", "信号样本", "信号簇"],
     ["辐射源", "平台", "武器"]),
    ("装备于", "EQUIPPED_ON", "装备搭载/部署于平台或配发给单位(装备→平台)。"
     "「某舰艇装备一部雷达」记为 雷达 装备于 某舰艇",
     ["装备于", "装备", "搭载", "安装于", "部署于", "挂载"], 0.75,
     ["辐射源", "平台", "武器"],
     ["平台", "组织"]),
    ("隶属", "BELONGS_TO", "隶属、配属、下辖、编成关系(下级→上级)。"
     "「A下辖B」记为 B 隶属 A;辐射源之间允许(如干扰机隶属雷达系统)",
     ["隶属", "属于", "下辖", "配属", "归属", "编成", "辖"], 0.75,
     ["平台", "组织", "辐射源"],
     ["组织", "平台", "辐射源"]),
    ("具有状态", "HAS_STATE", "辐射源/平台/目标具有某种工作状态",
     ["具有状态", "处于", "状态为", "工作状态"], 0.75,
     ["辐射源", "平台", "目标", "组织"],
     ["工作状态"]),
    ("关联目标", "RELATED_TARGET", "辐射源/平台/武器/事件与目标的关联关系",
     ["关联目标", "针对", "指向", "跟踪", "打击"], 0.75,
     ["辐射源", "平台", "武器", "事件", "信号簇"],
     ["目标"]),
    ("平台候选", "PLATFORM_CANDIDATE", "信号/辐射源可能的平台候选关系(推断,非确定)",
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
    # —— 需求(10)明文的「支援」;交联/涉及补齐数据链与历史事件(需求(7))的表达 ——
    ("支援", "SUPPORTS", "为其他装备/平台/单位提供引导、目标指示、掩护、保障(支援方→被支援方)",
     ["支援", "掩护", "保障", "引导", "目标指示", "协同"], 0.75,
     ["辐射源", "平台", "组织"],
     ["辐射源", "平台", "组织", "武器"]),
    ("交联", "CO_LINK", "装备/平台/单位之间的数据链交联与信息传输",
     ["交联", "数据链", "传输给", "交换", "上报", "链接"], 0.75,
     ["辐射源", "平台", "组织"],
     ["辐射源", "平台", "组织"]),
    ("涉及", "INVOLVES", "事件涉及/参与的辐射源、平台、单位、武器或目标(事件→参与方)",
     ["涉及", "参与", "参加", "出现于", "出动"], 0.75,
     ["事件"],
     ["辐射源", "平台", "组织", "武器", "目标"]),
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

    for name, en, desc, mergeable, schema in _ENTITIES:
        row = await _row(name, "entity")
        if row is None:
            session.add(ExtractionDict(name=name, en=en, category="entity",
                                       description=desc, semantic_mergeable=mergeable,
                                       json_schema=schema, enabled=True))
            changed += 1
        elif row.deleted_at is not None or (row.en or "") != en \
                or (row.description or "") != desc \
                or bool(row.semantic_mergeable) != mergeable \
                or (row.json_schema or None) != schema or not bool(row.enabled):
            row.deleted_at = None
            row.en = en
            row.description = desc
            row.semantic_mergeable = mergeable
            row.json_schema = schema
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

    # 派生关系:登记进字典便于溯源/图例,category=derived 不进 LLM 抽取(load_relations 只取 relation)
    for d in DERIVED_RELATIONS:
        name, en, desc = d["name"], d["en"], d["description"]
        head_types, tail_types = list(d["head_types"]), list(d["tail_types"])
        row = await _row(name, "derived")
        if row is None:
            session.add(ExtractionDict(name=name, en=en, category="derived", description=desc,
                                       head_types=head_types, tail_types=tail_types,
                                       min_confidence=1.0, enabled=True))
            changed += 1
        elif row.deleted_at is not None or (row.en or "") != en \
                or (row.description or "") != desc \
                or list(row.head_types or []) != head_types \
                or list(row.tail_types or []) != tail_types or not bool(row.enabled):
            row.deleted_at = None
            row.en = en
            row.description = desc
            row.head_types = head_types
            row.tail_types = tail_types
            row.enabled = True
            changed += 1

    keep_names = {
        "entity": {e[0] for e in _ENTITIES},
        "relation": {r[0] for r in _RELATIONS},
        "derived": {d["name"] for d in DERIVED_RELATIONS},
    }
    now = datetime.now(timezone.utc)
    rows = (await session.execute(select(ExtractionDict))).scalars().all()
    for r in rows:
        keep = r.name in keep_names.get(r.category, set())
        if r.deleted_at is None and not keep:
            r.deleted_at = now
            changed += 1

    await session.commit()
    return changed


async def seed_extraction_dict(session: AsyncSession) -> int:
    """补列后按种子全集同步(空表即全量写入;旧库补齐/刷新/软删)。幂等。"""
    await _ensure_mergeable_column(session)
    await _ensure_relation_type_columns(session)
    return await _sync_seed(session)
