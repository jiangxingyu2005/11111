"""POST /api/eval/benchmark —— 在线测评模块（接在在线检索服务后面）。

跑内置 golden 测评集（默认 data/seed_date，6 类客观 gold），返回：
  - 召回率：crecall / recall / cprec / MRR / nDCG（test / val / 分类型）
  - 响应时间：mean / p50 / p90 / p95 / p99 / max / 超预算
  - 逐 query Top5 匹配度：rerank_score（0~1）+ grade（2 核心命中 / 1 相关 / 0 未命中）

实现完全复用 eval/ 包：golden.py（gold 集）、metrics.py（指标）、report.py（汇总）、
runner.py::run_in_process（进程内 parse→四路召回→RRF→rerank）。

边界：
  - benchmark 的 gold child_id 来自 seed_dir（默认 data/seed_date），故测评必须跑在同一份
    seed 上；run_in_process 内部会 force RETRIEVAL_BACKEND=fake + RETRIEVAL_SEED_DIR=seed_dir。
  - 跑完在 finally 中还原 env / 缓存 / fake store，不污染在线服务的后续请求。
  - 全量（max_per_type=200 ≈ 584 条）进程内约 1~2 分钟；前端默认用小样本保证秒级反馈。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]          # retrieval/
DEFAULT_SEED_DIR = ROOT / "data" / "seed_date"


class BenchmarkRequest(BaseModel):
    seed_dir: str = str(DEFAULT_SEED_DIR)
    max_per_type: int = Field(default=200, ge=1, le=2000)
    ks: list[int] = Field(default=[5, 10, 20])
    scope: list[str] = Field(default=["kb_public"])


@router.post("/api/eval/benchmark")
async def benchmark(req: BenchmarkRequest) -> dict:
    from eval.golden import build_eval_set, load_seed, split_82
    from eval.report import _by_type, _hit_rate, _lean, summarize
    from eval.runner import run_in_process
    import app.config as C
    from app.store.fake import reset_fake_store

    seed_dir = Path(req.seed_dir)
    if not (seed_dir / "chunks.json").exists():
        raise HTTPException(status_code=400, detail=f"seed 缺失: {seed_dir}")

    # 1. 构造 gold 集（6 类客观 gold，8:2 分层）
    chunks, _entities, records = load_seed(seed_dir)
    evalset = build_eval_set(records, chunks, req.max_per_type)
    test, val = split_82(evalset)
    combined = test + val

    # 2. 进程内跑全链路（run_in_process 内部 asyncio.run，须在无事件循环的线程里执行）
    saved_backend = os.environ.get("RETRIEVAL_BACKEND")
    saved_seed = os.environ.get("RETRIEVAL_SEED_DIR")
    try:
        results = await asyncio.to_thread(run_in_process, combined, req.scope, seed_dir)
    finally:
        # 还原 env + 缓存 + fake store，避免污染在线服务后续请求
        for key, val in (("RETRIEVAL_BACKEND", saved_backend),
                         ("RETRIEVAL_SEED_DIR", saved_seed)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        C.reset_caches()
        reset_fake_store()

    # 3. 汇总（results 与 combined 顺序一致：test 在前、val 在后）
    test_results = results[:len(test)]
    val_results = results[len(test):]
    per_query = _lean(test_results + val_results)

    return {
        "meta": {
            "seed_dir": str(seed_dir),
            "n_chunks": len(chunks),
            "n_queries": len(evalset),
            "max_per_type": req.max_per_type,
            "ks": req.ks,
            "backend": "fake(hash embed, in-process)",
        },
        "test": summarize(test_results, req.ks),
        "val": summarize(val_results, req.ks),
        "by_type": _by_type(test_results + val_results, req.ks),
        "hit_rate@5": {"test": _hit_rate(test_results), "val": _hit_rate(val_results)},
        "per_query": per_query,
    }
