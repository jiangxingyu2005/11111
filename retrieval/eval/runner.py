"""两种运行模式：进程内直连（in_process）与 HTTP 回放（http，EV-1）。

统一产出 per-query 记录（顺序与输入 queries 一一对应）：
  {
    "q": str, "type": str,
    "gold": {child_id: grade},           # grade 2=核心 / 1=相关
    "final": [{"child_id","parent_id","rerank_score","snippet"}],  # 重排后 top5
    "fused": [child_id, ...],             # 外层 RRF top50（http 模式为空，EV-3 未实现）
    "lat_ms": float,                      # in_process=检索段；http=服务 total_ms
    "first_token_ms": float|None,         # http 才有（未接生成则为 None）
    "degraded": [str, ...],               # 降级码（in_process 暂不采集）
  }
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_fake_env(seed_dir: Path) -> None:
    """in_process 模式：强制 fake 通道 + date seed（须在 import app.config 之前）。"""
    os.environ["RETRIEVAL_BACKEND"] = "fake"
    os.environ["RETRIEVAL_SEED_DIR"] = str(seed_dir)
    os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
    os.environ.setdefault("POSTGRES_USER", "retrieval_ro")
    os.environ.setdefault("POSTGRES_PASSWORD", "dev_only")
    os.environ.setdefault("POSTGRES_DATABASE", "qingxi")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("MILVUS_PORT", "19530")
    os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
    os.environ.setdefault("NEO4J_USER", "ro")
    os.environ.setdefault("NEO4J_PASSWORD", "dev_only")
    os.environ.setdefault("GRAPHRAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    os.environ.setdefault("RETRIEVAL_FAULT_INJECT", "")


async def _run_one_in_process(q: str, scope: list[str]) -> dict:
    """单 query 走真实代码路径 parse → 四路召回 → RRF → rerank。"""
    import app.config as C  # noqa: E402
    from app.degrade.orchestrator import DegradedCollector
    from app.fuse.rerank import rerank
    from app.fuse.rrf import rrf_fuse
    from app.recall import bm25_route, community_route, graph_route, vector_route
    from app.understand.query_parser import parse

    coll = DegradedCollector()
    cfg = C.effective_cfg()
    pq = parse(q, coll)
    vec, bm, gr, cm = await asyncio.gather(
        vector_route.recall(q, scope=scope, collector=coll, conditions=pq.conditions),
        bm25_route.recall(q, scope=scope, collector=coll),
        graph_route.recall(q, scope=scope, collector=coll),
        community_route.recall(q, scope=scope, collector=coll),
    )
    fused = rrf_fuse({"vector_hybrid": vec, "bm25": bm, "graph_local": gr, "graph_global": cm},
                     k=int(cfg.get("fusion", {}).get("k", 60)),
                     top_k=int(cfg.get("fusion", {}).get("top_k", 50)))
    fused_ids = [h.child_id for h in fused]
    top = await rerank(q, fused, coll)
    final = [{"child_id": h.child_id, "parent_id": h.parent_id,
              "rerank_score": h.meta.get("rerank_score"),
              "snippet": (h.content or "")[:80]} for h in top]
    return {"fused": fused_ids, "final": final}


async def _run_all_in_process(queries: list[dict], scope: list[str]) -> list[dict]:
    results = []
    for i, item in enumerate(queries):
        t0 = time.perf_counter()
        r = await _run_one_in_process(item["q"], scope)
        lat_ms = (time.perf_counter() - t0) * 1000
        results.append({"q": item["q"], "type": item["type"], "gold": item["gold"],
                        "fused": r["fused"], "final": r["final"], "lat_ms": lat_ms,
                        "first_token_ms": None, "degraded": []})
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(queries)}", flush=True)
    return results


def run_in_process(queries: list[dict], scope: list[str], seed_dir: Path,
                   warm: bool = True) -> list[dict]:
    """进程内直连：完整召回口径 recall@5/10/20 + 检索段延迟。结果顺序与输入一致。"""
    _setup_fake_env(seed_dir)
    import app.config as C  # noqa: E402
    C.reset_caches()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[警告] 未设 PYTHONHASHSEED=0，in_process 结果可能不可复现。"
              "建议：PYTHONHASHSEED=0 python -m eval --mode in_process", file=sys.stderr)

    from app.store.fake import reset_fake_store
    reset_fake_store()

    # 预热：装载单例 + 让路由缓存热起来（不计入响应时间统计）
    if warm and queries:
        asyncio.run(_run_one_in_process(queries[0]["q"], scope))
    return asyncio.run(_run_all_in_process(queries, scope))


def _post_one(client, target_url: str, query: str, scope: list[str]) -> tuple[list[dict], dict, list]:
    """POST /api/retrieval（eval_mode=true），解析 SSE 的 results/done 事件。"""
    url = target_url.rstrip("/") + "/api/retrieval"
    payload = {"query": query, "scope": scope, "eval_mode": True}
    hits: list[dict] = []
    timings: dict = {}
    degraded: list = []
    event = None
    with client.stream("POST", url, json=payload) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            if raw.startswith("event:"):
                event = raw[6:].strip()
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if event == "results":
                hits = obj.get("hits", [])
                degraded = obj.get("degraded", [])
            elif event == "done":
                timings = obj.get("timings", {})
                degraded = obj.get("degraded", degraded)
            elif event == "error":
                degraded = [{"code": "error", "detail": obj.get("message", "")}]
            # meta/progress/assets/token 不影响评分，跳过
    codes = [d.get("code") if isinstance(d, dict) else str(d) for d in degraded]
    return hits, timings, codes


def run_http(queries: list[dict], scope: list[str], target_url: str) -> list[dict]:
    """HTTP 回放：端到端 total_ms + recall@5（recall@10/20 需 EV-3 中间结果，未实现）。"""
    import httpx
    results = []
    with httpx.Client(timeout=120.0) as client:
        for i, item in enumerate(queries):
            t0 = time.perf_counter()
            hits, timings, degraded = _post_one(client, target_url, item["q"], scope)
            wall_ms = (time.perf_counter() - t0) * 1000
            final = [{"child_id": h.get("child_id"), "parent_id": h.get("parent_id"),
                      "rerank_score": (h.get("score") or {}).get("rerank"),
                      "snippet": (h.get("content") or "")[:80]} for h in hits]
            results.append({
                "q": item["q"], "type": item["type"], "gold": item["gold"],
                "fused": [], "final": final,
                "lat_ms": timings.get("total_ms") or wall_ms,
                "first_token_ms": timings.get("first_token_ms"),
                "degraded": degraded,
            })
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(queries)}", flush=True)
    return results
