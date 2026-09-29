"""契约测试（fake 后端，端到端）：SSE 序列 + 响应契约 + 图谱契约。

与 scripts/smoke.py 同断言，但走 pytest（CI 友好）。
运行：pytest tests/contract/test_pipeline_fake.py -v
"""
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.update({
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_USER": "ro", "POSTGRES_PASSWORD": "x",
    "MILVUS_HOST": "127.0.0.1",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "x",
    "GRAPHRAG_LLM_BASE_URL": "",            # 不配 LLM：验证降级路径（不出正文但主链存活）
    "RETRIEVAL_TRACE_DIR": "logs/trace_test",
})

from fastapi.testclient import TestClient  # noqa: E402

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


def _schema(name):
    return json.loads((ROOT / "api" / name).read_text(encoding="utf-8"))


def test_sse_sequence():
    ev = _events()
    seq = [e for e, _ in ev]
    assert seq[0] == "meta" and seq[-1] == "done"
    assert "results" in seq and "assets" in seq


def test_results_have_evidence_and_parent_content():
    ev = dict(_events())["results"]["hits"]
    assert ev, "results 不应为空"
    for r in ev:
        assert r["evidence"]["child_id"] == r["child_id"]
        assert r["evidence"]["span"], "fake seed 必须带真实 span"
        assert r["content"], "QRY-06 父块回捞必须有正文"


def test_aggregated_response_matches_schema():
    import jsonschema

    ev = _events()
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
    jsonschema.validate(agg, _schema("retrieval_response.v1.schema.json"))


def test_graph_ref_and_subgraph_roundtrip():
    import jsonschema

    ev = _events()
    done = dict(ev)["done"]
    gr = done["graph_ref"]
    assert gr["available"] and gr["focus_entities"]
    tid = done["trace_id"]
    g = TestClient(app).get("/api/graph/subgraph",
                            params={"trace_id": tid, "scope": SCOPE}).json()
    jsonschema.validate(g, _schema("subgraph_view.v1.schema.json"))
    assert g["nodes"] and g["edges"]
    assert all(e["evidence"]["child_id"] for e in g["edges"])   # G-API-02


def test_scope_required_r4():
    r = TestClient(app).post("/api/retrieval", json={"query": QUERY})
    assert r.status_code == 422, "scope 必填，缺失必须 422（权限红线 R4）"


def test_scope_isolation():
    ev = dict(_events(scope=["kb_public"]))["results"]["hits"]
    assert {r["metadata"]["kb_id"] for r in ev} <= {"kb_public"}


def test_no_llm_then_generation_unavailable():
    """未配 LLM：必须留 generation_unavailable，且主链仍以 done 正常结束。"""
    ev = _events()
    codes = [d["code"] for d in dict(ev)["done"]["degraded"]]
    assert "generation_unavailable" in codes
    assert [e for e, _ in ev][-1] == "done"
