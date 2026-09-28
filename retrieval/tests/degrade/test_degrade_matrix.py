"""降级编排测试（G-QC-03：行为 + degraded 留痕，双验才及格）。

用 RETRIEVAL_FAULT_INJECT 真注入故障，验证：
  ① 对应环节真的被剥离（行为）
  ② degraded[] 里有对应 code（留痕，禁静默降级）
  ③ 主链不崩（可以降级成傻子，但不能崩掉）
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CASES = [
    ("vector", "vector_unavailable", {}),
    ("graph", "graph_unavailable", {}),
    ("pg", "parent_fetch_unavailable", {}),
    ("llm", "generation_unavailable", {}),
    ("rerank", "rerank_timeout", {}),
    ("bm25", "bm25_unavailable", {"RETRIEVAL_BM25_ENABLED": "true"}),
    ("community", "community_unavailable", {"RETRIEVAL_COMMUNITY_ENABLED": "true"}),
]


@pytest.mark.parametrize("fault,code,extra", CASES)
def test_fault_injection_records_degraded(fault, code, extra):
    """每个故障场景跑独立子进程（env 变了必须换进程，否则 Settings/单例缓存串味）。

    bm25/community 需先启用对应开关（否则 recall 在 try 前就返回空，故障注入无从触发）。
    """
    script = (
        "import json,os,sys;"
        "sys.path.insert(0, r'{root}');"
        "from fastapi.testclient import TestClient;"
        "from app.main import app;"
        "c=TestClient(app);"
        "r=c.post('/api/retrieval', json={{'query':'X波段 线性调频 雷达','scope':['kb_public','kb_internal']}});"
        "assert r.status_code==200, r.status_code;"
        "ev=[l for l in r.text.splitlines()];"
        "done=[l for l in ev if l.startswith('data:') and 'graph_ref' in l][-1];"
        "d=json.loads(done[5:]);"
        "codes=[x['code'] for x in d['degraded']];"
        "print('DEGRADED='+json.dumps(codes));"
        "assert '{code}' in codes, codes;"
        "assert d['trace_id']"
    ).format(root=ROOT, code=code)
    env = dict(os.environ)
    env.update({
        "RETRIEVAL_BACKEND": "fake", "RETRIEVAL_SEED_DIR": "data/seed",
        "RETRIEVAL_FAULT_INJECT": fault,
        "POSTGRES_HOST": "127.0.0.1", "POSTGRES_USER": "ro", "POSTGRES_PASSWORD": "x",
        "MILVUS_HOST": "127.0.0.1",
        "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "x",
        "GRAPHRAG_LLM_BASE_URL": "", "RETRIEVAL_TRACE_DIR": "logs/trace_test",
    })
    env.update(extra)
    r = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"{fault} 注入失败: {r.stdout[-400:]}\n{r.stderr[-400:]}"


def test_degraded_shape_matches_contract():
    """step 必须落在契约枚举里（曾因直接透传内部 stage 'backfill' 而校验失败）。"""
    sys.path.insert(0, str(ROOT))
    from app.degrade.codes import DegradedCode, STEP_MAP, step_of

    allowed = {"faq", "query_parse", "vector", "sparse", "graph", "scalar_filter",
               "rerank", "pg_fetch", "asset", "generate"}
    for c in DegradedCode:
        assert step_of(c.value) in allowed, f"{c.value} -> {step_of(c.value)} 不在枚举内"
    assert set(STEP_MAP) >= {c.value for c in DegradedCode}, "存在未映射的降级码"


def test_budget_watchdog_cuts_by_level():
    """看门狗必须单调：耗时越长，等级越狠（L4 正常 -> L0 兜底）。

    取样点刻意避开阈值本身：3.6/4.6/5.1/5.3 而不是 3.5/4.5/5.0/5.2。
    踩在边界上会因为"浮点累加 + 真实时间流逝几十微秒"随机越界成下一档，
    变成时好时坏的 flaky 测试——那比没有测试更糟。
    """
    from app.degrade.orchestrator import BudgetWatchdog

    w = BudgetWatchdog()
    assert w.current_level().value == "L4"
    w._t0 -= 3.6                              # e≈3.6 > 3.5 -> L3（砍重排）
    assert w.current_level().value == "L3"
    w._t0 -= 1.0                              # e≈4.6 > 4.5 -> L2（砍图路）
    assert w.current_level().value == "L2"
    w._t0 -= 0.5                              # e≈5.1 > 5.0 -> L1（砍 NL2Query+稀疏）
    assert w.current_level().value == "L1"
    assert w.over_budget() is False           # 5.1 < 5.2，尚未超预算
    w._t0 -= 0.2                              # e≈5.3 > 5.2 -> L0（PG 全文兜底）
    assert w.current_level().value == "L0"
    assert w.over_budget() is True
