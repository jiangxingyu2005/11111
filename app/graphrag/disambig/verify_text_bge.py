"""text 目录真实数据 + 真实 BGE(bge-large-zh-v1.5)向量的消歧评测与评分。

与 verify_text.py(合成向量 WordEmbedder)对应,本脚本用真实 EmbeddingClient 读 .env
(provider=bge),在真实语义向量下复测四层:
  A. 确定性-别名归并(真实 PG)—— 与向量无关,回归兜底。
  B. 确定性-类型作用域 + 负例不误并(真实 PG)。
  C. 向量-语义归并(真实 Milvus + 真实 BGE)—— 无别名时同义异名靠语义归并。
  D. 端到端 align(无别名 + 真实 BGE)—— 现状 mock 向量漏并的异名,真实语义能否救回。

前置:.env 已切 GRAPHRAG_EMBEDDING_PROVIDER=bge 且模型已本地缓存;PG/Milvus 已启动。
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
from .. import resolve
from ..embeddings import EmbeddingClient
from ..ids import canonical_entity_id
from ..milvus_client import AsyncMilvusClient
from ..models import Entity
from ..seed import seed_extraction_dict
from ...utils.database import SessionLocal, init_db

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


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


async def main() -> int:
    await init_db()
    milvus = AsyncMilvusClient()
    embeddings = EmbeddingClient()  # 真实 bge(读 .env,模型懒加载)
    await milvus.create_entity_collection()

    radar_id = canonical_entity_id(*CANON)
    lines: list[str] = []

    # ---- 相似度分布(雷达 vs 正/负例,展示阈值 margin)----
    radar_vec = embeddings.embed_one(CANON[0])
    pos_scores = [_cos(radar_vec, embeddings.embed_one(n)) for (n, _, _) in POSITIVE]
    neg_scores = [_cos(radar_vec, embeddings.embed_one(n)) for (n, _, _) in NEGATIVE]
    lines.append(f"[分布] 正例相似度 {[f'{s:.3f}' for s in pos_scores]} "
                 f"| 负例 {[f'{s:.3f}' for s in neg_scores]}")
    lines.append(f"[分布] gap = pos_min {min(pos_scores):.3f} vs neg_max {max(neg_scores):.3f}")

    async with SessionLocal() as session:
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()

        await seed_extraction_dict(session)  # 建列 + 标注 semantic_mergeable(幂等)

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

        # ---- 场景 B:类型作用域 + 负例(异类型同名不误并)----
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

        # ---- 场景 C:向量-语义归并(真实 Milvus + 真实 BGE)----
        await milvus.delete_entity_vector(radar_id)
        await milvus.upsert_entity_vector(radar_id, CANON[0], CANON[1], radar_vec)
        tp_c = 0
        for (n, a, t) in POSITIVE:
            if await resolve.resolve_vector(milvus, embeddings, None, n, t) == radar_id:
                tp_c += 1
        fp_c = 0
        for (n, a, t) in NEGATIVE:
            if await resolve.resolve_vector(milvus, embeddings, None, n, t) == radar_id:
                fp_c += 1
        fn_c = len(POSITIVE) - tp_c
        p_c, r_c, f_c = _f1(tp_c, fp_c, fn_c)
        lines.append(f"C 向量-语义归并(真实 BGE): TP={tp_c} FP={fp_c} FN={fn_c} "
                     f"→ P={p_c:.2f} R={r_c:.2f} F1={f_c:.2f}")

        # ---- 场景 D:无别名 + 真实 BGE 端到端 align ----
        await resolve.persist(session, [Entity(entity_id=radar_id, name=CANON[0],
                                               entity_type=CANON[1])], milvus, embeddings)
        await session.commit()
        bare = [Entity(entity_id=canonical_entity_id(n, t), name=n, entity_type=t)
                for (n, a, t) in POSITIVE]
        _, _, remap_d = await resolve.align(session, bare, [], milvus, embeddings, None)
        tp_d = sum(1 for (n, a, t) in POSITIVE
                   if remap_d.get(canonical_entity_id(n, t)) == radar_id)
        lines.append(f"D 无别名 + 真实 BGE 异名归并: {tp_d}/{len(POSITIVE)}")

        # ---- 清理 ----
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()
        for (n, a, t) in POSITIVE + NEGATIVE:
            await milvus.delete_entity_vector(canonical_entity_id(n, t))
        await milvus.delete_entity_vector(radar_id)
        await milvus.delete_entity_vector(src_id)

    print("\n".join(lines))
    print("-" * 60)
    print(f"确定性层(有别名) F1={f_a:.2f} | 向量语义层(真实 BGE) F1={f_c:.2f} | "
          f"无别名端到端 {tp_d}/{len(POSITIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
