"""消歧向量层 + 全量对账集成验证(真实 PG + Milvus + Neo4j)。

运行:
    cd qingxi && python -m app.graphrag.disambig.verify_resolve

前置:PG/Milvus/Neo4j 均已启动;每次运行清理自身测试数据,可重复执行。
用合成向量模拟 BGE(雷达/雷达站 语义相近),证明:
  A. 向量层 resolve_vector 端到端(类型作用域 + 阈值);
  B. global_resolve 由向量命中驱动 DSU 归并 + 四层回填(PG/Neo4j/Milvus)+ 幂等。
"""

from __future__ import annotations

import asyncio
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台无法编码 ① 等字符
except Exception:
    pass

from sqlalchemy import delete

from ...models import EntityRegistryEntry, ParentDocument
from .. import db as gdb
from .. import dict as dict_store
from .. import resolve
from ..ids import canonical_entity_id
from ..milvus_client import AsyncMilvusClient
from ..models import Entity
from ..neo4j_client import Neo4jClient
from ..seed import seed_extraction_dict
from ...utils.database import SessionLocal, init_db

DIM = 1024
TAIL_ID = "test_resolve_tail_entity"


class FakeEmbedder:
    """合成向量:雷达/雷达站 语义相近(cos≈0.999),电台异类(cos≈0.5);其余稀疏确定性。"""

    def embed_one(self, text: str) -> list[float]:
        v = [0.0] * DIM
        if "雷达站" in text:
            v[0], v[1] = 0.999, 0.04
        elif "雷达" in text:
            v[0] = 1.0
        elif "电台" in text:
            v[0], v[1] = 0.5, 0.866
        else:
            v[abs(hash(text)) % DIM] = 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v


async def main() -> int:
    await init_db()

    milvus = AsyncMilvusClient()
    fake = FakeEmbedder()
    results: list[tuple[str, bool]] = []

    def check(label: str, cond: bool) -> None:
        results.append((label, bool(cond)))

    # 建实体 collection(幂等)
    await milvus.create_entity_collection()

    # ---- Part A:向量层 resolve_vector 端到端(合成向量)----
    radar_id = canonical_entity_id("雷达", "装备")
    station_id = canonical_entity_id("雷达站", "装备")
    await milvus.delete_entity_vector(radar_id)
    await milvus.delete_entity_vector(station_id)
    await milvus.upsert_entity_vector(radar_id, "雷达", "装备", fake.embed_one("雷达"))

    got = await resolve.resolve_vector(milvus, fake, None, "雷达站", "装备")
    check("A① 向量语义归并:雷达站 → 雷达", got == radar_id)
    got2 = await resolve.resolve_vector(milvus, fake, None, "电台", "装备")
    check("A② 异类不并:电台 → None", got2 is None)
    got3 = await resolve.resolve_vector(milvus, fake, None, "雷达站", "辐射源")
    check("A③ 类型作用域:辐射源下「雷达站」→ None", got3 is None)

    # ---- Part B:global_resolve 由向量命中驱动 + 四层回填 ----
    async with SessionLocal() as session:
        await session.execute(delete(EntityRegistryEntry))
        await session.commit()

        # 抽取字典种子(建列 + 标注 semantic_mergeable;幂等)
        await seed_extraction_dict(session)

        # ---- Part A.2:参数值类型不进向量层(方案①)----
        mergeable = await dict_store.load_mergeable_types(session)
        check("A④ 可向量化类型含装备、排除频率",
              "装备" in mergeable and "频率" not in mergeable)
        freq_id = canonical_entity_id("2.4GHz", "频率")
        await milvus.upsert_entity_vector(freq_id, "2.4GHz", "频率", fake.embed_one("2.4GHz"))
        got_freq = await resolve.resolve_vector(milvus, fake, None, "2.4GHz", "频率", mergeable)
        check("A⑤ 频率类型不进向量层:resolve_vector → None", got_freq is None)

        # 预置两个重复实体(无别名,靠向量相似归并):雷达 先入库
        await resolve.persist(session, [Entity(entity_id=radar_id, name="雷达", entity_type="装备")])
        await resolve.persist(session, [Entity(entity_id=station_id, name="雷达站", entity_type="装备")])
        await session.commit()
        await milvus.upsert_entity_vector(radar_id, "雷达", "装备", fake.embed_one("雷达"))
        await milvus.upsert_entity_vector(station_id, "雷达站", "装备", fake.embed_one("雷达站"))

        # 父块绑两个实体(kg_entity_id 逗号串)
        parent_id = "test_resolve_parent"
        await gdb.upsert(session, ParentDocument, {
            "parent_id": parent_id, "doc_id": "test_doc", "title": "t", "content": "c",
            "kg_entity_id": f"{radar_id},{station_id}",
        }, "parent_id")
        await session.commit()

        # Milvus ems_chunks 预置一个子块,kg_entity_ids 含两个 id
        chunk_id = f"{parent_id}#1"
        await milvus.upsert([{
            "child_id": chunk_id, "content": "test", "parent_id": parent_id,
            "doc_id": "test_doc", "kb_id": "", "enabled": True, "parent_enabled": True,
            "kg_entity_ids": [radar_id, station_id], "content_vector": fake.embed_one("test"),
        }])

        async with Neo4jClient() as neo4j:
            await neo4j.merge_entity({"entity_id": radar_id, "name": "雷达", "entity_type": "装备", "properties": {}})
            await neo4j.merge_entity({"entity_id": station_id, "name": "雷达站", "entity_type": "装备", "properties": {}})
            await neo4j.merge_entity({"entity_id": TAIL_ID, "name": "目标", "entity_type": "装备", "properties": {}})
            await neo4j.merge_segment({"parent_id": parent_id, "doc_id": "test_doc", "title": "t"})
            await neo4j.merge_binds(radar_id, parent_id, "primary", 0.9)
            await neo4j.merge_binds(station_id, parent_id, "primary", 0.9)
            await neo4j.merge_relation({"head_entity_id": station_id, "tail_entity_id": TAIL_ID,
                                        "rel_type": "CO_LINK", "confidence": 0.8})

            # 全量对账(向量命中驱动)
            report = await resolve.global_resolve(session, neo4j, milvus, fake, None, dry_run=False)
        await session.commit()

        merged_pairs = {(m["old_id"], m["new_id"]) for m in report["merged"]}
        check("B① 向量驱动归并:雷达站 → 雷达", (station_id, radar_id) in merged_pairs)
        check("B② 关系迁移计数 ≥1", report["relations_moved"] >= 1)
        check("B③ BINDS 迁移计数 ≥1", report["binds_moved"] >= 1)
        check("B④ 父块回填计数 ≥1", report["parents_updated"] >= 1)
        check("B⑤ chunks 回填计数 ≥1", report["chunks_updated"] >= 1)

        radar_rows = await gdb.select_where(session, EntityRegistryEntry, entity_id=radar_id)
        station_rows = await gdb.select_where(session, EntityRegistryEntry, entity_id=station_id)
        check("B⑥ station 标记 merged", bool(station_rows)
              and station_rows[0]["status"] == "merged"
              and station_rows[0]["merged_into"] == radar_id)
        check("B⑦ radar 吸收别名「雷达站」", bool(radar_rows)
              and "雷达站" in (radar_rows[0]["aliases"] or []))

        parent_rows = await gdb.select_where(session, ParentDocument, parent_id=parent_id)
        kg = parent_rows[0]["kg_entity_id"] if parent_rows else ""
        check("B⑧ 父块 kg_entity_id 去重重映射", kg == radar_id)

        async with Neo4jClient() as neo4j:
            ids = {e["entity_id"] for e in await neo4j.list_all_entities()}
            check("B⑨ Neo4j station 已删、radar/目标保留",
                  station_id not in ids and radar_id in ids and TAIL_ID in ids)
            binds = await neo4j.entity_ids_of_parent(parent_id)
            check("B⑩ 父块 BINDS 归 radar", station_id not in binds and radar_id in binds)

            # 幂等:二次对账 merged 空
            report2 = await resolve.global_resolve(session, neo4j, milvus, fake, None, dry_run=False)
        await session.commit()
        check("B⑪ 幂等:二次对账 merged=[]", report2["merged"] == [])

        # ---- 清理测试数据 ----
        await session.execute(delete(EntityRegistryEntry))
        await gdb.delete(session, ParentDocument, parent_id, "parent_id")
        await session.commit()
        async with Neo4jClient() as neo4j:
            for eid in (radar_id, station_id, TAIL_ID):
                await neo4j.run("MATCH (e:Entity {entity_id: $id}) DETACH DELETE e", id=eid)
            await neo4j.detach_delete_segment(parent_id)
        await milvus.delete([chunk_id])
        await milvus.delete_entity_vector(radar_id)
        await milvus.delete_entity_vector(station_id)
        await milvus.delete_entity_vector(freq_id)

    passed = sum(1 for _, ok in results if ok)
    for label, ok in results:
        print(("  PASS  " if ok else "  FAIL  ") + label)
    print("-" * 60)
    print(f"{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
