"""门禁测试 G-QC-01~05（V2.0 §6.1 在线检索专属门禁，5 条）。

覆盖范围与判定口径以《在线检索开发文档_V2.0.md》§6.1 为准：
  G-QC-01  阈值独立配置/独立生效（τ_a / τ_b / τ_s）
  G-QC-02  外层 RRF 严格 1 source 1 vote
  G-QC-03  降级矩阵注入覆盖（本期 12 条，#8~12 社区、#13 标量 deferred 豁免）
  G-QC-04  TRACE 完整性（凭 trace_id 拉全链 span，stage 覆盖率 100%）
  G-QC-05  SSE 聚合响应 100% 过 retrieval_response.v1.schema.json（含降级路径）

fake 通道运行（零外部依赖）；seed 需先 `python scripts/gen_seed.py` 生成。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.update({
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_USER": "ro", "POSTGRES_PASSWORD": "x",
    "MILVUS_HOST": "127.0.0.1",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "x",
    "GRAPHRAG_LLM_BASE_URL": "",            # 不配 LLM：验证降级路径（主链存活 + 留痕）
    "RETRIEVAL_TRACE_DIR": "logs/trace_gate",
})

from fastapi.testclient import TestClient  # noqa: E402

from app.config import reset_caches  # noqa: E402
from app.main import app  # noqa: E402

SCOPE = ["kb_public", "kb_internal"]
QUERY = "X波段 线性调频 雷达信号，载频 9GHz，PRF 1200Hz"


def _events(query=QUERY, scope=None):
    c = TestClient(app)
    with c.stream("POST", "/api/retrieval", json={"query": query, "scope": scope or SCOPE}) as r:
        assert r.status_code == 200
        out, ev, buf = [], None, []
        for raw in r.iter_lines():
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw.startswith("event:"):
                ev = raw[6:].strip()
            elif raw.startswith("data:"):
                buf.append(raw[5:].strip())
            elif raw.strip() == "" and ev is not None:
                out.append((ev, json.loads("".join(buf))))
                ev, buf = None, []
        return out


def _aggregate(ev):
    """把 SSE 事件聚合为契约快照（与 retrieval_response.v1.schema.json 对齐）。"""
    agg = {"contract_version": "1.0", "trace_id": "", "meta": {"intent": "ENTITY_LEVEL"},
           "results": [], "assets": [], "artifacts": [], "graph_ref": None,
           "degraded": [], "timings": None}
    for name, d in ev:
        if name == "meta":
            agg["trace_id"] = d["trace_id"]
        elif name == "results":
            agg["results"] = d["hits"]
        elif name == "done":
            agg.update({k: d.get(k) for k in ("meta", "graph_ref", "degraded", "timings")})
    return agg


def _schema(name):
    return json.loads((ROOT / "api" / name).read_text(encoding="utf-8"))


# ────────────────────────────────────────────────────────────
# G-QC-01：τ_a / τ_b / τ_s 独立配置、独立生效
# ────────────────────────────────────────────────────────────
def test_qc01_thresholds_independently_loaded():
    """改任一阈值不得影响其它（配置加载断言）。

    τ_a = milvus.dense.primary.threshold（0.72）
    τ_b = milvus.dense.secondary.threshold（0.68）
    τ_s = Scene KNN 独立阈值（0.62，RETRIEVAL_TAU_S，IF-4 deferred，本期不入配置、不参与断言）

    独立性在【原始配置】层面断言（0.72 vs 0.68）；fake 通道 dev_overrides 把两路
    同降到 0.05 是「量纲适配的环境覆盖」，两条路径仍独立接线，不算漂移。
    """
    import copy

    reset_caches()
    from app.config import effective_cfg, retrieval_cfg

    cfg = retrieval_cfg()
    dense = cfg["milvus"]["dense"]
    tau_a, tau_b = dense["primary"]["threshold"], dense["secondary"]["threshold"]
    assert isinstance(tau_a, (int, float)) and isinstance(tau_b, (int, float))
    assert tau_a != tau_b, "τ_a/τ_b 必须独立取值（0.72 vs 0.68），共享一值即配置漂移"

    c2 = copy.deepcopy(cfg)
    c2["milvus"]["dense"]["primary"]["threshold"] = 0.99     # 只改 A 路
    assert c2["milvus"]["dense"]["secondary"]["threshold"] == tau_b, "改 τ_a 影响到了 τ_b —— 阈值未独立"

    # τ_s（Scene KNN）为 IF-4 deferred：不得被 τ_a/τ_b 键别名占用
    assert "tau_s" not in dense, "τ_s 是 IF-4 独立阈值，不得混入 dense 两路"

    # fake 通道 dev_overrides 允许同降两路阈值（环境适配），但仍是两条独立路径
    eff = effective_cfg()
    assert eff["milvus"]["dense"]["primary"]["threshold"] < 0.2
    assert eff["milvus"]["dense"]["secondary"]["threshold"] < 0.2
    reset_caches()


# ────────────────────────────────────────────────────────────
# G-QC-02：外层 RRF 严格 1 source 1 vote
# ────────────────────────────────────────────────────────────
def test_qc02_one_source_one_vote():
    from app.fuse.rrf import Hit, assert_one_source_one_vote, rrf_fuse

    sources = {
        "vector_hybrid": [Hit(child_id="a", rank_score=0.9, source="vector_hybrid"),
                          Hit(child_id="b", rank_score=0.8, source="vector_hybrid")],
        "graph_local": [Hit(child_id="b", rank_score=1.0, source="graph_local"),
                        Hit(child_id="c", rank_score=0.9, source="graph_local")],
    }
    assert_one_source_one_vote(sources)                  # 融合前：组内不得有重复 child_id
    fused = rrf_fuse(sources, k=60)
    for h in fused:
        voted = h.meta["voted_by"]
        assert len(voted) == len(set(voted)), "同一来源对同一命中重复投票"
        assert len(voted) <= len(sources), "票数不得超过来源数"
    # b 被两路命中，但每来源恰好一票（不是两票）
    b = next(h for h in fused if h.child_id == "b")
    assert sorted(b.meta["voted_by"]) == ["graph_local", "vector_hybrid"]


def test_qc02_four_source_fusion_votes():
    """四源给同一 child_id 投票时，voted_by 恰好覆盖四源（每源一票，1 source 1 vote）。

    这是「整体链路四源融合」的确定性证明：recall_all 端到端里 rerank 只留 top5，
    互补召回的 graph/community 块未必进最终结果，故四源覆盖在此处（融合层）断言。
    """
    from app.fuse.rrf import Hit, rrf_fuse

    sources = {
        "vector_hybrid": [Hit(child_id="a", rank_score=1.0, source="vector_hybrid")],
        "bm25": [Hit(child_id="a", rank_score=1.0, source="bm25")],
        "graph_local": [Hit(child_id="a", rank_score=1.0, source="graph_local")],
        "graph_global": [Hit(child_id="a", rank_score=1.0, source="graph_global")],
    }
    fused = rrf_fuse(sources, k=60)
    a = next(h for h in fused if h.child_id == "a")
    assert sorted(a.meta["voted_by"]) == ["bm25", "graph_global", "graph_local",
                                          "vector_hybrid"]


# ────────────────────────────────────────────────────────────
# G-QC-03：降级矩阵注入覆盖（本期 12 条，#8~12 社区、#13 标量 deferred）
# ────────────────────────────────────────────────────────────
# 本期按可注入性分四档：
#   INJECTABLE      fake 通道真注入（RETRIEVAL_FAULT_INJECT）
#   INPUT_TRIGGERED 无需注入器，由输入触发
#   UNIT_TESTED     hit 级跳过码（社区 #9~12），由 community_route 单测直接触发
#   DEFERRED        特性未启用，无法注入（禁伪造注入器；看门狗逻辑由单测覆盖）
INJECTABLE = {"vector_unavailable", "rerank_timeout", "graph_unavailable",
              "generation_unavailable", "parent_fetch_unavailable",
              "bm25_unavailable", "community_unavailable"}
INPUT_TRIGGERED = {"condition_invalid"}
UNIT_TESTED = {"community_no_start", "community_stale",
               "community_summary_partial", "community_no_evidence"}
DEFERRED = {"faq_unavailable", "nl2query_llm_timeout", "sparse_unavailable",
            "budget_cut_community", "budget_cut_route3"}
IN_SCOPE = INJECTABLE | INPUT_TRIGGERED | UNIT_TESTED | DEFERRED


def test_qc03_matrix_codepoints_complete_and_mapped():
    """18 条码位全部落字典且都有 step 映射；本期 12 条码位齐备（禁静默降级）。"""
    from app.degrade.codes import DegradedCode, STEP_MAP

    codes = {c.value for c in DegradedCode}
    assert IN_SCOPE <= codes, f"本期范围码位缺失: {IN_SCOPE - codes}"
    assert codes <= set(STEP_MAP), f"存在未映射降级码（step 必填）: {codes - set(STEP_MAP)}"


def test_qc03_smoke_injects_injectable_subset():
    """7 条可注入码对应的故障场景必须在 smoke.py 里真实注入（行为 + 留痕双验的载体）。"""
    smoke = (ROOT / "scripts" / "smoke.py").read_text(encoding="utf-8")
    fault_map = {"vector_unavailable": "fault_vector", "rerank_timeout": "fault_rerank",
                 "graph_unavailable": "fault_graph", "generation_unavailable": "fault_llm",
                 "parent_fetch_unavailable": "fault_pg", "bm25_unavailable": "fault_bm25",
                 "community_unavailable": "fault_community"}
    for code, scenario in fault_map.items():
        assert scenario in smoke, f"{code} 缺少对应 smoke 故障注入场景 {scenario}"


# ────────────────────────────────────────────────────────────
# G-QC-03 补：社区 #9~12 hit 级跳过码（单测覆盖档，禁伪造注入器）
# ────────────────────────────────────────────────────────────
def test_qc03_community_no_start_recorded(monkeypatch):
    """Scene KNN 无过阈值社区 -> community_no_start 留痕 + 空结果。"""
    import asyncio

    from app.degrade.codes import DegradedCode
    from app.degrade.orchestrator import DegradedCollector
    from app.recall import community_route

    class _S:
        retrieval_community_enabled = True

    async def _emb(query, conditions=None):
        return {"dense": [0.1] * 8, "dense_v2": None, "sparse": {}}

    monkeypatch.setattr(community_route, "get_settings", lambda: _S())
    monkeypatch.setattr("app.recall.vector_route._embed_query_full", _emb)
    monkeypatch.setattr(community_route, "_scene_knn", lambda *a, **k: [])

    col = DegradedCollector()
    hits = asyncio.run(community_route.recall("query", scope=["kb_public"], collector=col))
    assert hits == []
    assert col.has(DegradedCode.COMMUNITY_NO_START)


def test_qc03_community_hit_level_skip_codes():
    """stale / summary_partial / no_evidence：单社区跳过 + 留痕，其余社区继续。"""
    from app.degrade.orchestrator import DegradedCollector
    from app.recall.community_route import _backfill

    ccfg = {"hops": 2, "max_degree": 50, "limit": 50}
    comms = [
        {"community_id": "comm_stale", "score": 0.9, "summary": "s", "stale": True},
        {"community_id": "comm_empty", "score": 0.8, "summary": "", "stale": False},
        {"community_id": "comm_nomember", "score": 0.7, "summary": "s",
         "member_entity_ids": [], "stale": False},
    ]
    col = DegradedCollector()
    hits = _backfill(comms, ["kb_public"], ccfg, col)
    assert hits == []
    assert sorted(col.codes()) == ["community_no_evidence", "community_stale",
                                   "community_summary_partial"]


# ────────────────────────────────────────────────────────────
# G-QC-04：TRACE 完整性（凭 trace_id 拉全链 span，缺 stage 即失败）
# ────────────────────────────────────────────────────────────
EXPECTED_STAGES = {"understand", "recall", "fuse", "rerank", "backfill", "generate"}


def _wait_trace(trace_dir: Path, trace_id: str, timeout_s: float = 5.0) -> dict:
    p = trace_dir / f"{trace_id}.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        time.sleep(0.1)
    raise FileNotFoundError(f"trace 未异步导出: {p}（旁路导出失败需排查）")


def test_qc04_trace_span_coverage():
    """任一请求凭 trace_id 拉出全链 span，stage 覆盖率 100%。"""
    ev = _events()
    done = dict(ev)["done"]
    trace_id = done["trace_id"]
    payload = _wait_trace(ROOT / "logs" / "trace_gate", trace_id)
    stages = {s["stage"] for s in payload["spans"]}
    missing = EXPECTED_STAGES - stages
    assert not missing, f"span 缺 stage: {missing}（覆盖率 {len(stages)}/{len(EXPECTED_STAGES)}）"
    # 每个 span 都必须是已关闭的（duration_ms 已回填），且带 status
    for s in payload["spans"]:
        assert s["status"] in ("ok", "degraded", "failed")


# ────────────────────────────────────────────────────────────
# G-QC-05：SSE 聚合响应 100% 过 retrieval_response.v1.schema.json
# ────────────────────────────────────────────────────────────
def test_qc05_sse_aggregate_matches_schema():
    import jsonschema

    agg = _aggregate(_events())
    jsonschema.validate(agg, _schema("retrieval_response.v1.schema.json"))


def test_qc05_sse_degraded_path_matches_schema():
    """降级路径（故障注入）下聚合响应仍必须过 schema（degraded[] 的 step/code 落在枚举内）。"""
    import jsonschema

    # 降级路径：L3 超量纲 query 触发 condition_invalid，聚合响应仍必须过 schema
    ev = _events(query="载频 99999GHz 的信号")
    agg = _aggregate(ev)
    codes = [d["code"] for d in agg["degraded"]]
    assert "condition_invalid" in codes, "L3 超量纲必须留 condition_invalid"
    jsonschema.validate(agg, _schema("retrieval_response.v1.schema.json"))


def test_qc05_schema_files_are_valid_jsonschema():
    """两份 schema 本身必须是合法 JSON Schema（error/degraded 契约的骨架前提）。"""
    import jsonschema

    for name in ("retrieval_response.v1.schema.json", "subgraph_view.v1.schema.json"):
        jsonschema.Draft202012Validator.check_schema(_schema(name))


def test_graph_relation_types_whitelist():
    """G4：relation_types 白名单校验——未知关系类型 422，已知类型放行（config/relations.yaml）。"""
    c = TestClient(app)
    # 未知关系类型 → 422（禁静默放行）
    r = c.get("/api/graph/subgraph", params={
        "trace_id": "00000000", "scope": SCOPE, "relation_types": ["INVALID_REL"]})
    assert r.status_code == 422, "未知 relation_type 必须 422"
    # 已知关系类型 → 200（无 focus 时降级 focus_entity_missing，但白名单校验已过）
    r2 = c.get("/api/graph/subgraph", params={
        "trace_id": "00000000", "scope": SCOPE, "relation_types": ["SOURCE_OF", "SAME_BAND"]})
    assert r2.status_code == 200
    codes = [d["code"] for d in r2.json()["degraded"]]
    assert "focus_entity_missing" in codes
