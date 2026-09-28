"""消歧集成验证(真实 PG):证明确定性消歧在入库链路(align+persist)对真实数据生效。

运行:
    cd qingxi && python -m app.graphrag.disambig.verify_pg

前置:PG 已启动(engine 连 postgres),启动时 create_all 自动建 entity_registry 表。
每次运行先清空 entity_registry,可重复执行。
"""

from __future__ import annotations

import asyncio

from sqlalchemy import delete

from ...models import EntityRegistryEntry
from .. import db as gdb
from .. import resolve
from ..ids import canonical_entity_id
from ..models import Entity
from ...utils.database import SessionLocal, init_db


async def main() -> int:
    await init_db()  # create_all(自动建 entity_registry)

    results: list[tuple[str, bool]] = []

    def check(label: str, cond: bool) -> None:
        results.append((label, bool(cond)))

    async with SessionLocal() as session:
        # 清空注册表(干净起点,可重复运行)
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()

        # ---- 文档 1:实体「雷达」(装备)首次入库 ----
        radar_id = canonical_entity_id("雷达", "装备")
        await resolve.persist(session, [Entity(entity_id=radar_id, name="雷达", entity_type="装备")])
        await session.commit()

        # ---- 文档 2:「雷达站」别名「雷达」→ 跨文档归并 ----
        radar_station_id = canonical_entity_id("雷达站", "装备")
        e2 = Entity(entity_id=radar_station_id, name="雷达站", entity_type="装备", aliases=["雷达"])
        aligned, _, remap = await resolve.align(session, [e2], [])
        check("① 别名跨文档归并:雷达站 → 雷达", remap.get(radar_station_id) == radar_id
              and aligned[0].entity_id == radar_id)

        await resolve.persist(session, aligned)
        await session.commit()
        rows = await gdb.select_where(session, EntityRegistryEntry, entity_id=radar_id)
        check("② 别名并集持久化:注册表含「雷达站」", bool(rows) and "雷达站" in rows[0]["aliases"])
        check("③ name 首现者不变:仍为「雷达」", bool(rows) and rows[0]["name"] == "雷达")

        # ---- 文档 3:异名同指纹(辐射源甲/乙,容差) ----
        props_a = {"freq_band": "2.4GHz", "pulse_width": 1.0, "prf": 1000}
        a_id = canonical_entity_id("辐射源甲", "辐射源")
        await resolve.persist(session, [Entity(entity_id=a_id, name="辐射源甲",
                                               entity_type="辐射源", properties=props_a)])
        await session.commit()
        b_id = canonical_entity_id("辐射源乙", "辐射源")
        eb = Entity(entity_id=b_id, name="辐射源乙", entity_type="辐射源",
                    properties={"freq_band": "2.4 GHz", "pulse_width": 1.2, "prf": 1040})
        _, _, remap_b = await resolve.align(session, [eb], [])
        check("④ 指纹跨文档归并(容差):辐射源乙 → 甲", remap_b.get(b_id) == a_id)

        # ---- 类型作用域:同名同指纹、不同 type 不互并 ----
        freq_id = canonical_entity_id("2.4GHz", "频率")
        await resolve.persist(session, [Entity(entity_id=freq_id, name="2.4GHz",
                                               entity_type="频率", properties={"value": "2.4GHz"})])
        src_id = canonical_entity_id("2.4GHz", "辐射源")
        await resolve.persist(session, [Entity(entity_id=src_id, name="2.4GHz",
                                               entity_type="辐射源", properties={"freq_band": "2.4GHz"})])
        await session.commit()
        _, _, remap_f = await resolve.align(
            session, [Entity(entity_id=canonical_entity_id("2.4GHz", "频率"),
                             name="2.4GHz", entity_type="频率", properties={"value": "2.4GHz"})], [])
        check("⑤ 类型作用域:频率 2.4GHz 归自身,不并入辐射源",
              remap_f.get(freq_id) is None or remap_f.get(freq_id) == freq_id)

        # ---- 幂等:重跑不产生新实体、仍归同 id ----
        e2b = Entity(entity_id=radar_station_id, name="雷达站", entity_type="装备", aliases=["雷达"])
        _, _, remap2 = await resolve.align(session, [e2b], [])
        check("⑥ 幂等:重跑仍归同 id", remap2.get(radar_station_id) == radar_id)

        all_rows = await gdb.select_all(session, EntityRegistryEntry)
        active = [r for r in all_rows if r["status"] == "active"]
        # 活跃实体 = 雷达 / 辐射源甲 / 频率 2.4GHz / 辐射源 2.4GHz(共 4;雷达站、辐射源乙 已归并不独立存在)
        check("⑦ 注册表 active 实体数 = 4", len(active) == 4)

    passed = sum(1 for _, ok in results if ok)
    for label, ok in results:
        print(("  PASS  " if ok else "  FAIL  ") + label)
    print("-" * 60)
    print(f"{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
