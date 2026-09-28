"""消歧模块离线验证:三阶段匹配 + 入库对齐 + 全图批量对账。

运行:
    cd qingxi
    python -m app.graphrag.disambig.demo

不依赖 PG/Neo4j/Milvus/LLM,全部内存 + 假 embedder,可离线跑通。
"""

from __future__ import annotations

from ..ids import canonical_entity_id
from .engine import NeverJudge, ResolverConfig, align, global_resolve, persist, resolve_one
from .stores import DocIndexStore, EntityRecord, EntityRegistryStore, EntityVectorIndex, GraphStore


class FakeEmbedder:
    """假语义向量:让「雷达↔雷达站」相似(≥0.95)、「对空雷达」灰区(≈0.90)、「电台」不似(≈0.50)。"""

    _TABLE = {
        "雷达":    [1.000, 0.000],
        "雷达站":  [0.999, 0.040],   # cos≈0.9993
        "对空雷达": [0.900, 0.436],   # cos≈0.900
        "电台":    [0.500, 0.866],   # cos≈0.500
    }

    def __call__(self, name):
        return self._TABLE.get(name, [0.0, 1.0])


def _eid(name, etype):
    return canonical_entity_id(name, etype)


def _rec(eid, name, etype, aliases=None, props=None, fp=None):
    return EntityRecord(entity_id=eid, name=name, entity_type=etype,
                        aliases=list(aliases or []), properties=dict(props or {}),
                        fingerprint=fp)


def main():
    results = []

    def check(label, cond):
        results.append((label, bool(cond)))

    cfg = ResolverConfig(sim_high=0.95, sim_low=0.85, topk=5, llm_gate=False)
    judge = NeverJudge()
    embedder = FakeEmbedder()

    # ===== 场景 1:入库在线对齐 —— 别名跨文档归并 =====
    registry = EntityRegistryStore()
    vindex = EntityVectorIndex()
    e_radar = _rec(_eid("雷达", "装备"), "雷达", "装备")
    persist(registry, vindex, embedder, [e_radar])
    new = [{"entity_id": _eid("雷达站", "装备"), "name": "雷达站",
            "entity_type": "装备", "aliases": ["雷达"], "properties": {}}]
    _merged, _rels, remap1 = align(cfg, registry, vindex, embedder, judge, new, [])
    check("① 别名归并:『雷达站』→『雷达』", remap1[_eid("雷达站", "装备")] == e_radar.entity_id)

    # ===== 场景 2:属性指纹归并(异名同指纹) =====
    registry2 = EntityRegistryStore()
    vindex2 = EntityVectorIndex()
    a = _rec(_eid("辐射源甲", "辐射源"), "辐射源甲", "辐射源",
             props={"freq_band": "2.4GHz", "pulse_width": 1.0, "prf": 1000})
    persist(registry2, vindex2, embedder, [a])
    hit2 = resolve_one(cfg, registry2, vindex2, embedder, judge,
                       "辐射源乙", "辐射源", [],
                       {"freq_band": "2.4 GHz", "pulse_width": 1.2, "prf": 1040})
    check("① 指纹归并:『辐射源乙』→『辐射源甲』(容差)", hit2 == a.entity_id)

    # ===== 场景 3:类型作用域 —— 同指纹不同 type 不并 =====
    hit3 = resolve_one(cfg, registry2, vindex2, embedder, judge,
                       "2.4GHz", "频率", [], {"value": "2.4GHz"})
    check("类型作用域:频率『2.4GHz』不并入辐射源", hit3 is None)

    # ===== 场景 4:向量归并(异名同义,无别名/指纹) =====
    registry4 = EntityRegistryStore()
    vindex4 = EntityVectorIndex()
    radar4 = _rec(_eid("雷达", "装备"), "雷达", "装备")
    persist(registry4, vindex4, embedder, [radar4])
    hit4 = resolve_one(cfg, registry4, vindex4, embedder, judge,
                       "雷达站", "装备", [], {})
    check("② 向量归并:『雷达站』(≥0.95)→『雷达』", hit4 == radar4.entity_id)

    # ===== 场景 5:灰区 —— mock 裁判不并 =====
    hit5 = resolve_one(cfg, registry4, vindex4, embedder, judge,
                       "对空雷达", "装备", [], {})
    check("③ 灰区:『对空雷达』(0.90)mock 不并", hit5 is None)

    # ===== 场景 6:全图批量对账(四层回填,幂等) =====
    registry6 = EntityRegistryStore()
    vindex6 = EntityVectorIndex()
    graph6 = GraphStore()
    docs6 = DocIndexStore()
    x = _rec(_eid("某雷达", "装备"), "某雷达", "装备", aliases=["X波段雷达"])
    y = _rec(_eid("X波段雷达", "装备"), "X波段雷达", "装备")
    persist(registry6, vindex6, embedder, [x, y])
    graph6.bind(x.entity_id, "p1", "primary", 1.0)
    graph6.bind(y.entity_id, "p2", "primary", 0.9)
    graph6.relate(x.entity_id, y.entity_id, "CO_LINK")
    docs6.set_parent("p1", [x.entity_id])
    docs6.set_parent("p2", [y.entity_id])
    docs6.set_chunk("p1#1", [x.entity_id])
    docs6.set_chunk("p2#1", [y.entity_id])

    report = global_resolve(cfg, registry6, vindex6, embedder, judge, graph6, docs6)
    canon = x.entity_id  # x 先注册 → canonical
    check("对账:恰好 1 条归并", len(report["merged"]) == 1)
    check("对账:canonical 取最早注册", report["merged"][0]["new_id"] == canon
          and report["merged"][0]["old_id"] == y.entity_id)
    check("对账:BINDS 迁移(canonical←p2)", graph6.binds.get((canon, "p2")) == "primary"
          and (y.entity_id, "p2") not in graph6.binds)
    check("对账:自环关系丢弃", graph6.relations == [])
    check("对账:旧节点已删", graph6.entities == {canon})
    check("对账:PG kg_entity_id 重映射", docs6.parent_entities["p2"] == [canon])
    check("对账:Milvus kg_entity_ids 重映射", docs6.chunk_entities["p2#1"] == [canon])
    check("对账:注册表旧行标记 merged",
          registry6.entries[y.entity_id].status == "merged"
          and registry6.entries[y.entity_id].merged_into == canon)
    check("对账:别名被吸收进 canonical", "x波段雷达" in registry6.entries[canon].aliases)

    report2 = global_resolve(cfg, registry6, vindex6, embedder, judge, graph6, docs6)
    check("对账幂等:再跑 merged 为空", report2["merged"] == [])

    # ===== 场景 7:dry_run 只出计划、不动数据 =====
    registry7 = EntityRegistryStore()
    vindex7 = EntityVectorIndex()
    graph7 = GraphStore()
    docs7 = DocIndexStore()
    p = _rec(_eid("雷达", "装备"), "雷达", "装备", aliases=["雷达站"])
    q = _rec(_eid("雷达站", "装备"), "雷达站", "装备")
    persist(registry7, vindex7, embedder, [p, q])
    graph7.bind(p.entity_id, "p1")
    graph7.bind(q.entity_id, "p2")
    dry = global_resolve(cfg, registry7, vindex7, embedder, judge, graph7, docs7, dry_run=True)
    check("dry_run:返回计划且不落库", dry["dry_run"] is True and len(dry["merged"]) == 1
          and registry7.entries[q.entity_id].status == "active")

    # ===== 汇总 =====
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for label, ok in results:
        print(("  PASS  " if ok else "  FAIL  ") + label)
    print("-" * 60)
    print(f"{passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
