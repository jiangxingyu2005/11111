"""text 目录真实数据驱动的消歧效果评测 + 评分。

数据源(text/ 目录真实实体名):
  - ems_test.txt / ems_test.html(同一雷达站信号参数):雷达 / 对空搜索雷达 / 雷达站 /
    干扰机 / 侦察设备 / 指挥所 / 舰艇 / 2.4GHz / 850MHz / 3.2GHz / QPSK / FM / BPSK。
  - cat_弹道导弹-战术.csv:9K720 Iskander-M(BM03)、MGM-140 ATACMS(BM04)—— 两条不同导弹。
  - test_eq.html:歼-20(战斗机)。

评测三层:
  A. 确定性-别名归并(真实 PG):同义异名靠别名归并(正例)
  B. 确定性-类型作用域 + 负例不误并(真实 PG):异名异义/异类型不误并(负例)
  C. 向量-语义归并(真实 Milvus,合成向量模拟 BGE):无别名时同义异名靠语义归并

评分:精确率 P / 召回率 R / F1(正例=应归并,负例=不应归并)。
"""

from __future__ import annotations

import asyncio
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sqlalchemy import delete

from ...models import EntityRegistryEntry
from .. import db as gdb
from .. import resolve
from ..ids import canonical_entity_id
from ..milvus_client import AsyncMilvusClient
from ..models import Entity
from ...utils.database import SessionLocal, init_db

DIM = 1024

# ---- ground truth(从 text 数据提取,模拟理想 LLM 抽取带别名)----
CANON = ("雷达", "装备")  # 雷达系统 canonical
POSITIVE = [  # (name, aliases, type) 应归并到「雷达」
    ("对空搜索雷达", ["雷达"], "装备"),
    ("雷达站", ["雷达"], "装备"),
]
NEGATIVE = [  # (name, aliases, type) 不应归并(各自独立)
    ("干扰机", [], "装备"),
    ("侦察设备", [], "装备"),
    ("指挥所", [], "装备"),
    ("舰艇", [], "装备"),
    ("9K720 Iskander-M", [], "弹道导弹"),
    ("MGM-140 ATACMS", [], "弹道导弹"),
    ("歼-20", [], "战斗机"),
    ("2.4GHz", [], "频率"),
    ("850MHz", [], "频率"),
]


class WordEmbedder:
    """词表驱动的合成向量(模拟 BGE):雷达组语义相近,其余正交。

    仅用于「有语义向量时消歧能做什么」的演示;真实语义需装 BGE/配 openai。
    """

    _RADAR = {"雷达", "雷达站", "对空搜索雷达", "警戒雷达", "搜索雷达"}

    def embed_one(self, text: str) -> list[float]:
        v = [0.0] * DIM
        norm = text.strip()
        if norm in self._RADAR:
            v[0], v[1] = 1.0, 0.02
        elif "雷达" in norm:
            v[0], v[1] = 0.9, 0.436
        else:
            v[abs(hash(norm)) % DIM] = 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


async def main() -> int:
    await init_db()
    milvus = AsyncMilvusClient()
    word = WordEmbedder()
    await milvus.create_entity_collection()

    radar_id = canonical_entity_id(*CANON)
    lines: list[str] = []

    async with SessionLocal() as session:
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()

        # ---- 场景 A:确定性-别名归并(真实 PG)----
        await resolve.persist(session, [Entity(entity_id=radar_id, name=CANON[0],
                                               entity_type=CANON[1])])
        await session.commit()

        pos_entities = [Entity(entity_id=canonical_entity_id(n, t), name=n, entity_type=t,
                               aliases=a) for (n, a, t) in POSITIVE]
        neg_entities = [Entity(entity_id=canonical_entity_id(n, t), name=n, entity_type=t,
                               aliases=a) for (n, a, t) in NEGATIVE]
        _, _, remap = await resolve.align(session, pos_entities + neg_entities, [])

        tp_a = sum(1 for (n, a, t) in POSITIVE
                   if remap.get(canonical_entity_id(n, t)) == radar_id)
        fp_a = sum(1 for (n, a, t) in NEGATIVE
                   if remap.get(canonical_entity_id(n, t)) not in (None, canonical_entity_id(n, t)))
        fn_a = len(POSITIVE) - tp_a
        p_a, r_a, f_a = _f1(tp_a, fp_a, fn_a)
        lines.append(f"A 确定性-别名归并: TP={tp_a} FP={fp_a} FN={fn_a} "
                     f"→ P={p_a:.2f} R={r_a:.2f} F1={f_a:.2f}")

        # ---- 场景 B:类型作用域 + 负例(同名字符串、异类型不误并)----
        # 辐射源「2.4GHz」(频段值)与频率「2.4GHz」同名异类型,不应互并
        src_id = canonical_entity_id("2.4GHz", "辐射源")
        await resolve.persist(session, [Entity(entity_id=src_id, name="2.4GHz",
                                               entity_type="辐射源",
                                               properties={"freq_band": "2.4GHz"})])
        await session.commit()
        _, _, remap_b = await resolve.align(
            session, [Entity(entity_id=canonical_entity_id("2.4GHz", "频率"),
                             name="2.4GHz", entity_type="频率",
                             properties={"value": "2.4GHz"})], [])
        cross_merged = remap_b.get(canonical_entity_id("2.4GHz", "频率")) == src_id
        lines.append(f"B 类型作用域(频率 2.4GHz 不并入辐射源 2.4GHz): "
                     f"{'PASS' if not cross_merged else 'FAIL'}")

        # ---- 场景 C:向量-语义归并(真实 Milvus,合成向量模拟 BGE)----
        await milvus.delete_entity_vector(radar_id)
        await milvus.upsert_entity_vector(radar_id, CANON[0], CANON[1],
                                          word.embed_one(CANON[0]))
        tp_c = 0
        for (n, a, t) in POSITIVE:
            got = await resolve.resolve_vector(milvus, word, None, n, t)
            if got == radar_id:
                tp_c += 1
        # 负例:干扰机 / 导弹 语义不近,不应命中
        fp_c = 0
        for (n, a, t) in NEGATIVE:
            got = await resolve.resolve_vector(milvus, word, None, n, t)
            if got == radar_id:
                fp_c += 1
        fn_c = len(POSITIVE) - tp_c
        p_c, r_c, f_c = _f1(tp_c, fp_c, fn_c)
        lines.append(f"C 向量-语义归并(模拟 BGE): TP={tp_c} FP={fp_c} FN={fn_c} "
                     f"→ P={p_c:.2f} R={r_c:.2f} F1={f_c:.2f}")

        # ---- 场景 D:无别名 + mock 向量(现状)漏归并演示 ----
        bare = [Entity(entity_id=canonical_entity_id(n, t), name=n, entity_type=t)
                for (n, a, t) in POSITIVE]
        _, _, remap_d = await resolve.align(session, bare, [])
        tp_d = sum(1 for (n, a, t) in POSITIVE
                   if remap_d.get(canonical_entity_id(n, t)) == radar_id)
        lines.append(f"D 现状(无别名+mock 向量)异名归并: {tp_d}/{len(POSITIVE)} "
                     f"(={tp_d}/{len(POSITIVE)} 归并;漏并说明需上游别名或 BGE)")

        # 清理
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()
        for (n, a, t) in POSITIVE + NEGATIVE:
            await milvus.delete_entity_vector(canonical_entity_id(n, t))
        await milvus.delete_entity_vector(radar_id)
        await milvus.delete_entity_vector(src_id)

    print("\n".join(lines))
    print("-" * 60)
    print(f"确定性层(有别名) F1={f_a:.2f} | 向量层(模拟语义) F1={f_c:.2f} | "
          f"负例零误并 P={p_a:.2f}(A)/{p_c:.2f}(C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
