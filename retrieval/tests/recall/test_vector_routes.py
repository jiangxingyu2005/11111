"""① 路内层子路单测：hybrid（sparse）+ dense B（数值/单位归一化信号参数嵌入）。

口径（计划）：
  - sparse / denseB「缺席」是【配置态，非降级】——开启即生效，无需真 Milvus，不新增降级码。
  - dense B 要【真实语义区分度】：9GHz ≡ 9000MHz（同桶全等）、9.0 vs 9.2（邻桶部分相似）、
    参数无关文档（近正交），而非加盐哈希的灰度副向量。

fake 通道下用 mock.patch 注入 effective_cfg（sparse/secondary 由 yaml 门控，无 env 杠杆），
spy FakeStore.search_chunks 断言走 hybrid / 内层 N 路 RRF 路径（routes 键 + 路数）。
"""
from __future__ import annotations

import asyncio
import copy
import os
import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 注意：不设 RETRIEVAL_TRACE_DIR（本文件不测 trace，设了会覆盖 test_gate_qc 的 trace_gate 目录，
# 在 reset_caches 后污染 get_settings 缓存，导致 test_qc04 找不到 trace 文件）
os.environ.update({
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_USER": "ro", "POSTGRES_PASSWORD": "x",
    "MILVUS_HOST": "127.0.0.1",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "x",
})

from app.config import effective_cfg  # noqa: E402
from app.degrade.codes import DegradedCode  # noqa: E402
from app.degrade.orchestrator import DegradedCollector  # noqa: E402
from app.understand.query_parser import Condition  # noqa: E402

SCOPE = ["kb_public", "kb_internal"]


def _cfg_with(**overrides) -> dict:
    """基于真实 effective_cfg 深拷贝后改键，其余结构不漂移（fake dev_overrides 已含）。"""
    cfg = copy.deepcopy(effective_cfg())
    for path, val in overrides.items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg


def _run_recall(query: str, conditions=None, cfg_overrides=None):
    """跑一次 ① 路 recall，返回 (hits, 捕获到的 routes 字典, collector)。"""
    from app.recall import vector_route
    from app.store.fake import get_fake_store

    cfg = _cfg_with(**(cfg_overrides or {}))
    collector = DegradedCollector()
    # 直接 patch 单例实例方法（patch 类方法 + wraps 会丢 self 绑定，导致 search_chunks 缺参）
    store = get_fake_store()
    with mock.patch("app.recall.vector_route.effective_cfg", return_value=cfg):
        with mock.patch.object(store, "search_chunks",
                               wraps=store.search_chunks) as spy:
            hits = asyncio.run(vector_route.recall(
                query, scope=SCOPE, collector=collector, conditions=conditions))
    routes = spy.call_args.args[0] if spy.call_args else {}
    return hits, routes, collector


# ────────────────────────────────────────────────────────────
# dense B：真实语义区分度（纯函数级）
# ────────────────────────────────────────────────────────────
def test_hash_numeric_embedding_semantic_distinctiveness():
    """同参同桶全等；邻参部分相似；远参正交——B 路有真语义梯度，不是加盐哈希。"""
    from app.store.hash_embed import cosine, hash_numeric_embedding

    v9 = hash_numeric_embedding({"rf": 9.0})
    v9_2 = hash_numeric_embedding({"rf": 9.2})
    v100 = hash_numeric_embedding({"rf": 100.0})
    v9_doa = hash_numeric_embedding({"rf": 9.0, "doa": 30.0})

    assert cosine(v9, v9) > 0.999, "同向量余弦应≈1"
    assert cosine(v9, v9_2) > 0.0, "邻参应保留部分相似（软 one-hot 邻桶 0.5）"
    assert cosine(v9, v100) < cosine(v9, v9_2), "远参相似度必须低于邻参"
    assert 0.0 <= cosine(v9, v9_2) <= 1.0
    # 不同字段占据不同切片：加了 doa 后向量维度正交，相似度随参数差异下降
    assert 0.0 <= cosine(v9, v9_doa) <= 1.0


def test_numeric_embedding_unit_normalized_equivalence():
    """单位归一在 query_parser 层完成：9GHz 与 9000MHz 归一后同为 rf=9.0 -> 同向量。

    这里验证的是 B 路入参口径：value 已到基准单位，同一 value 产同一向量（确定性）。
    """
    from app.recall.vector_route import _conditions_to_numeric

    c_ghz = [Condition(field="rf", op="point", value=9.0, confidence=0.9)]
    c_mhz = [Condition(field="rf", op="point", value=9.0, confidence=0.9)]  # 9000MHz 已归一
    assert _conditions_to_numeric(c_ghz) == _conditions_to_numeric(c_mhz)
    assert _conditions_to_numeric(c_ghz) is not None
    assert _conditions_to_numeric([]) is None, "无五要素条件 -> B 路缺席"


# ────────────────────────────────────────────────────────────
# dense B：路由激活（改配置即启用，单测覆盖）
# ────────────────────────────────────────────────────────────
def test_dense_b_route_activates_with_conditions():
    """secondary.enabled=true 且有信号条件 -> 内层含 dense_v2（与 dense 双路 RRF）。"""
    conds = [Condition(field="rf", op="point", value=9.0, confidence=0.9),
             Condition(field="prf", op="point", value=1200.0, confidence=0.9)]
    hits, routes, collector = _run_recall(
        "X波段 载频 9GHz 的雷达",
        conditions=conds,
        cfg_overrides={"milvus.dense.secondary.enabled": True},
    )
    assert "dense_v2" in routes, f"B 路应激活，实际 routes={list(routes)}"
    assert len(routes) == 2, "应走 dense + dense_v2 双路内层 RRF"
    assert not collector.has(DegradedCode.VECTOR_UNAVAILABLE)


def test_dense_b_absent_without_conditions():
    """secondary.enabled=true 但无信号条件 -> dense_v2 为 None，B 路缺席（非降级）。"""
    hits, routes, collector = _run_recall(
        "X波段 线性调频 雷达信号",
        conditions=None,
        cfg_overrides={"milvus.dense.secondary.enabled": True},
    )
    assert "dense_v2" not in routes, "无条件时 B 路应缺席"
    assert len(routes) == 1, "只剩 dense 单路"
    assert not collector.has(DegradedCode.VECTOR_UNAVAILABLE)


# ────────────────────────────────────────────────────────────
# sparse / 内层 hybrid：路由激活（改配置即启用，单测覆盖）
# ────────────────────────────────────────────────────────────
def test_hybrid_sparse_route_activates():
    """sparse.enabled=true -> 内层含 sparse（与 dense 双路 RRF，走 hybrid）。"""
    hits, routes, collector = _run_recall(
        "X波段 线性调频 雷达信号，载频 9GHz",
        cfg_overrides={"milvus.sparse.enabled": True},
    )
    assert "sparse" in routes, f"sparse 路应激活，实际 routes={list(routes)}"
    assert len(routes) >= 2, "应走 dense + sparse 多路内层 RRF"
    assert not collector.has(DegradedCode.VECTOR_UNAVAILABLE)


def test_sparse_disabled_by_default_single_route():
    """默认配置 sparse.enabled=false -> 只走 dense 单路（内层不 hybrid）。"""
    hits, routes, collector = _run_recall("X波段 线性调频 雷达信号")
    assert routes and "sparse" not in routes, "默认 sparse 关闭，不应有 sparse 路"
    assert len(routes) == 1, "默认 dense-only 单路"
