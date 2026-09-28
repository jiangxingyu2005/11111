"""契约冒烟（mock 模式）：SSE 事件序列 + QRY-02 L1/L3 + 降级留痕。

运行：RETRIEVAL_MOCK_IF=true pytest tests/contract/test_sse_smoke.py
"""
import os

os.environ.setdefault("RETRIEVAL_MOCK_IF", "true")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_USER", "ro")
os.environ.setdefault("POSTGRES_PASSWORD", "x")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
os.environ.setdefault("NEO4J_USER", "ro")
os.environ.setdefault("NEO4J_PASSWORD", "x")

from app.degrade.orchestrator import DegradedCollector  # noqa: E402
from app.understand.query_parser import parse  # noqa: E402


def test_l1_range_parse():
    c = DegradedCollector()
    pq = parse("找载频 8~12GHz 的信号", c)
    assert pq.conditions, "L1 应命中 rf range"
    cond = pq.conditions[0]
    assert cond.field == "rf" and cond.op == "range"
    assert cond.value == [8.0, 12.0]
    assert cond.strength == "soft"  # 过渡期一律软语义


def test_l1_unit_convert():
    c = DegradedCollector()
    pq = parse("9.5GHz 的信号", c)
    assert pq.conditions[0].value == 9.5
    c2 = DegradedCollector()
    pq2 = parse("9500MHz 的信号", c2)
    assert abs(pq2.conditions[0].value - 9.5) < 1e-6  # MHz -> GHz 归一


def test_l3_reject_out_of_range():
    c = DegradedCollector()
    pq = parse("载频 99999GHz 的信号", c)  # 超量纲 -> V-03 丢弃
    assert not pq.conditions
    assert any(i["code"] == "condition_invalid" for i in c.items)  # 降级留痕
