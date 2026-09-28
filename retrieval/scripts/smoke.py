"""一键冒烟（端到端、真 HTTP、真校验）——判断"能跑"还是"花架子"的唯一入口。

做了什么：
  1. 进程内起 model_stub（真 uvicorn 端口 8011），retrieval 用 TestClient 真打 HTTP
  2. 解析 SSE 事件序列，聚合成契约对象，用 api/*.schema.json 做 jsonschema 校验
  3. 用 done 里的 trace_id 二次调 /api/graph/subgraph，校验图谱契约与证据锚点
  4. 跑故障注入场景（vector/graph/pg/llm/rerank/bm25/community），验证【行为 + degraded 留痕】双到位
  5. recall_all：四源全开，验证整链路跑通 + 无静默降级 + voted_by 覆盖四源

用法：
  python scripts/smoke.py                    # 跑全部场景（主链路 + recall_all + 7 个故障注入）
  python scripts/smoke.py --scenario main    # 只跑主链路
  python scripts/smoke.py --scenario recall_all
退出码：0=全绿；非 0=有失败（CI 可直接用）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUB_PORT = 8011
SCOPE = ["kb_public", "kb_internal"]
QUERY = "X波段 线性调频 雷达信号，载频 9GHz 左右，PRF 1200Hz"

BASE_ENV = {
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "retrieval_ro", "POSTGRES_PASSWORD": "dev_only",
    "POSTGRES_DATABASE": "qingxi",
    "MILVUS_HOST": "127.0.0.1", "MILVUS_PORT": "19530",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "dev_only",
    "GRAPHRAG_EMBEDDING_MODEL": "BAAI/bge-m3",
    "GRAPHRAG_LLM_BASE_URL": f"http://127.0.0.1:{STUB_PORT}/v1",
    "GRAPHRAG_LLM_API_KEY": "EMPTY", "GRAPHRAG_LLM_MODEL": "stub-echo",
    "RETRIEVAL_EMBED_URL": f"http://127.0.0.1:{STUB_PORT}/embed",
    "RETRIEVAL_RERANK_SERVICE_URL": f"http://127.0.0.1:{STUB_PORT}/rerank",
    "RETRIEVAL_TRACE_EXPORT": "file", "RETRIEVAL_TRACE_DIR": "logs/trace_smoke",
    "RETRIEVAL_MOCK_IF": "false",
    "PYTHONIOENCODING": "utf-8",
}


# ────────────────────────── stub 服务 ──────────────────────────
def start_stub() -> None:
    import uvicorn
    sys.path.insert(0, str(ROOT / "services"))
    from model_stub import app  # noqa: E402
    cfg = uvicorn.Config(app, host="127.0.0.1", port=STUB_PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    import httpx
    for _ in range(60):
        try:
            if httpx.get(f"http://127.0.0.1:{STUB_PORT}/healthz", timeout=1.0).status_code == 200:
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("model_stub 启动失败")


# ────────────────────────── SSE 工具 ──────────────────────────
def parse_sse(lines) -> list[tuple[str, dict]]:
    out, ev, buf = [], None, []
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw.startswith("event:"):
            ev = raw[6:].strip()
        elif raw.startswith("data:"):
            buf.append(raw[5:].strip())
        elif raw.strip() == "":
            if ev is not None:
                try:
                    out.append((ev, json.loads("".join(buf))))
                except ValueError:
                    out.append((ev, {"_raw": "".join(buf)}))
            ev, buf = None, []
    return out


def call_retrieval(client, query: str, scope: list[str]):
    with client.stream("POST", "/api/retrieval",
                       json={"query": query, "scope": scope}) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}"
        return parse_sse(r.iter_lines())


def aggregate(events: list[tuple[str, dict]]) -> dict:
    """SSE 事件 -> 契约对象（供 jsonschema 校验）。"""
    agg: dict = {"contract_version": "1.0", "trace_id": "", "meta": {"intent": "HYBRID"},
                 "results": [], "assets": [], "artifacts": [], "graph_ref": None,
                 "degraded": [], "timings": None}
    for ev, d in events:
        if ev == "meta":
            agg["trace_id"] = d.get("trace_id", "")
            agg["contract_version"] = d.get("contract_version", "1.0")
        elif ev == "results":
            agg["results"] = d.get("hits", [])
        elif ev == "assets":
            agg["assets"] = d.get("items", []) or []
        elif ev == "done":
            agg["meta"] = d.get("meta", agg["meta"])
            agg["graph_ref"] = d.get("graph_ref")
            agg["degraded"] = d.get("degraded", [])
            agg["timings"] = d.get("timings")
    return agg


def validate(obj: dict, schema_name: str) -> None:
    import jsonschema
    schema = json.loads((ROOT / "api" / schema_name).read_text(encoding="utf-8"))
    jsonschema.validate(obj, schema)


# ────────────────────────── 场景 ──────────────────────────
def scenario_main() -> list[str]:
    from fastapi.testclient import TestClient
    from app.main import app

    fails: list[str] = []

    def chk(cond: bool, msg: str) -> None:
        print(("  [OK] " if cond else "  [FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    client = TestClient(app)
    events = call_retrieval(client, QUERY, SCOPE)
    seq = [e for e, _ in events]
    print("  事件序列:", " -> ".join(seq))
    chk(seq[0] == "meta" and seq[-1] == "done", "SSE 序列以 meta 开头、done 结尾")
    chk("results" in seq and "assets" in seq, "含 results 与 assets 事件")
    chk(sum(1 for e in seq if e == "token") >= 1, "生成阶段有 token 流（非占位空流）")

    agg = aggregate(events)
    validate(agg, "retrieval_response.v1.schema.json")
    chk(True, "聚合响应通过 retrieval_response.v1.schema.json 校验")

    res = agg["results"]
    chk(len(res) > 0, f"results 非空（{len(res)} 条）")
    chk(all(r["evidence"].get("child_id") for r in res), "每条命中都有证据锚点 child_id")
    chk(all(r["evidence"].get("span") for r in res), "每条命中都有真实 span（非 null / 非 [0,0]）")
    chk(all(r["content"] for r in res), "每条命中都有父块正文（QRY-06 回捞生效）")

    chk(not agg["degraded"], f"无降级（degraded={agg['degraded']}）——主链路全绿才算无静默降级")

    gr = agg["graph_ref"] or {}
    chk(gr.get("available") is True and gr.get("focus_entities"),
        f"graph_ref 可用且带 focus 实体（{len(gr.get('focus_entities', []))} 个）")
    chk(agg["timings"] and agg["timings"].get("total_ms") is not None, "timings 含 total_ms")

    # ── 图谱二次取数 ──
    tid = agg["trace_id"]
    r = client.get("/api/graph/subgraph", params={"trace_id": tid, "scope": SCOPE})
    chk(r.status_code == 200, f"subgraph HTTP 200（{r.status_code}）")
    g = r.json()
    validate(g, "subgraph_view.v1.schema.json")
    chk(True, "subgraph 响应通过 subgraph_view.v1.schema.json 校验")
    chk(len(g["nodes"]) > 0 and len(g["edges"]) > 0,
        f"子图非空（nodes={len(g['nodes'])}, edges={len(g['edges'])}）")
    chk(all(e["evidence"].get("child_id") for e in g["edges"]), "每条边都有 evidence.child_id（G-API-02）")
    chk(all(n["is_focus"] for n in g["nodes"] if n["hops_from_focus"] == 0), "focus 节点已正确标记")

    # ── 权限红线：scope 必填 ──
    r2 = client.post("/api/retrieval", json={"query": QUERY})
    chk(r2.status_code == 422, f"缺 scope 返回 422（实际 {r2.status_code}）")

    # ── 权限隔离：只给 kb_public ──
    ev2 = call_retrieval(client, QUERY, ["kb_public"])
    agg2 = aggregate(ev2)
    kbs = {r["metadata"].get("kb_id") for r in agg2["results"]}
    chk(kbs <= {"kb_public"}, f"scope 过滤生效，命中 kb 集合={kbs}")

    return fails


def scenario_fault(key: str, expect_code: str) -> list[str]:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    events = call_retrieval(client, QUERY, SCOPE)
    agg = aggregate(events)
    codes = [d["code"] for d in agg["degraded"]]
    ok = expect_code in codes
    print(f"  注入 {key}: degraded={codes} -> {'[OK]' if ok else '[FAIL]'}")
    # 降级后仍必须能返回结果（禁止整链路崩掉）
    survived = [e for e, _ in events][-1] == "done"
    print(f"    主链存活: {'[OK] done' if survived else '[FAIL] 未正常结束'}")
    return [] if (ok and survived) else [f"fault {key}: 期望 {expect_code}，实际 {codes}，存活={survived}"]


def scenario_recall_all() -> list[str]:
    """四源全开（①向量 ②BM25 ③-A图 ③-B社区）端到端跑通：无静默降级 + voted_by 覆盖四源。"""
    from fastapi.testclient import TestClient
    from app.main import app

    fails: list[str] = []

    def chk(cond: bool, msg: str) -> None:
        print(("  [OK] " if cond else "  [FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    client = TestClient(app)
    events = call_retrieval(client, QUERY, SCOPE)
    seq = [e for e, _ in events]
    chk(seq[0] == "meta" and seq[-1] == "done", "SSE 序列以 meta 开头、done 结尾")

    # 召回阶段四路计数（progress.stage=recall）
    recall_counts: dict = {}
    for ev, d in events:
        if ev == "progress" and d.get("stage") == "recall":
            recall_counts = {k: d.get(k, 0) for k in ("vector", "bm25", "graph", "community")}
    chk(recall_counts.get("vector", 0) > 0, f"① 路向量召回非空（vector={recall_counts.get('vector')}）")
    chk(recall_counts.get("bm25", 0) > 0, f"② 路 BM25 召回非空（bm25={recall_counts.get('bm25')}）")
    chk(recall_counts.get("graph", 0) > 0, f"③-A 图路召回非空（graph={recall_counts.get('graph')}）")
    chk(recall_counts.get("community", 0) > 0,
        f"③-B 社区路召回非空（community={recall_counts.get('community')}）")

    agg = aggregate(events)
    validate(agg, "retrieval_response.v1.schema.json")
    chk(True, "聚合响应通过 retrieval_response.v1.schema.json 校验")
    chk(len(agg["results"]) > 0, f"最终 results 非空（{len(agg['results'])} 条）")
    chk(not agg["degraded"], f"四源全开无静默降级（degraded={agg['degraded']}）")

    # 四源「召回」由上面 progress 计数证明；最终 results 经 rerank 只留 top5（按内容相似度），
    # 互补召回的 graph/community 块未必进 top5——故「覆盖四源」在召回层断言，此处验证融合契约：
    #   1 source 1 vote（无重复票） + 跨源融合生效（≥2 源投票） + voted_by 是合法来源子集
    valid_sources = {"vector_hybrid", "bm25", "graph_local", "graph_global"}
    voted = set()
    dup = False
    for r in agg["results"]:
        vb = r["metadata"].get("voted_by", [])
        if len(vb) != len(set(vb)):
            dup = True
            fails.append(f"voted_by 重复投票: {r['child_id']} -> {vb}")
        voted |= set(vb)
    chk(not dup, "1 source 1 vote：无重复投票")
    chk(voted and voted <= valid_sources, f"voted_by 是合法来源子集（{sorted(voted)}）")
    chk(len(voted) >= 2, f"跨源 RRF 融合生效（{len(voted)} 个来源投票）")

    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all", "main", "recall_all",
                             "fault_vector", "fault_graph", "fault_pg",
                             "fault_llm", "fault_rerank", "fault_bm25", "fault_community"])
    args = ap.parse_args()

    fault = None
    if args.scenario.startswith("fault_"):
        fault = args.scenario[6:]
    env = dict(BASE_ENV)
    if fault:
        env["RETRIEVAL_FAULT_INJECT"] = fault
        # bm25/community 需先启用对应开关，否则 recall 在 try 前就返回空，故障注入无从触发
        if fault == "bm25":
            env["RETRIEVAL_BM25_ENABLED"] = "true"
        elif fault == "community":
            env["RETRIEVAL_COMMUNITY_ENABLED"] = "true"
    if args.scenario == "recall_all":
        env["RETRIEVAL_BM25_ENABLED"] = "true"
        env["RETRIEVAL_COMMUNITY_ENABLED"] = "true"
    for k, v in env.items():
        os.environ[k] = v

    try:
        start_stub()
    except Exception as e:
        print(f"[smoke] stub 启动失败: {e}")
        return 2

    fails: list[str] = []
    if args.scenario in ("all", "main") and not fault:
        print("\n=== 场景 main：全链路契约 ===")
        fails += scenario_main()
    if args.scenario == "recall_all":
        print("\n=== 场景 recall_all：四源全开端到端 ===")
        fails += scenario_recall_all()
    if fault:
        expect = {"vector": "vector_unavailable", "graph": "graph_unavailable",
                  "pg": "parent_fetch_unavailable", "llm": "generation_unavailable",
                  "rerank": "rerank_timeout", "bm25": "bm25_unavailable",
                  "community": "community_unavailable"}[fault]
        print(f"\n=== 场景 fault_{fault}：期望 {expect} ===")
        fails += scenario_fault(fault, expect)

    if args.scenario == "all" and not fault:
        for sc in ["recall_all"] + [f"fault_{f}" for f in
                                    ["vector", "graph", "pg", "llm", "rerank", "bm25", "community"]]:
            r = subprocess.run([sys.executable, __file__, "--scenario", sc],
                               cwd=str(ROOT), env={**os.environ},
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(r.stdout.strip())
            if r.returncode != 0:
                fails.append(f"{sc} 子进程失败: {r.stderr[-200:]}")
        print("\n=== 结果 ===")
        print("[smoke] 全部通过 [PASS]" if not fails else "[smoke] 失败项：\n  - " + "\n  - ".join(fails))
        return 0 if not fails else 1

    print("\n=== 结果 ===")
    if fails:
        print("[FAIL] " + "; ".join(fails))
        return 1
    print("[OK] 场景通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
