"""压缩流水线测试：token 触发 / 事实兜底 / 封顶 5 次 / fail-closed（fake 后端，无 LLM）。

用 FakeSessionStore + FakeMemoryBackend 直测 maybe_compress，monkeypatch 掉
recursive_summarize（fake 通道无 LLM，摘要靠替身返回固定串）。
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.memory import compressor
from app.memory.backend import FakeMemoryBackend
from app.memory.compressor import estimate_tokens, maybe_compress
from app.memory.session_store import FakeSessionStore


def test_estimate_tokens_roughly_counts():
    assert estimate_tokens("") == 0
    assert estimate_tokens("电磁信号") == 4          # CJK 按字
    assert estimate_tokens("hello world") == 2       # 英文按词
    assert estimate_tokens("雷达 radar") == 3        # 2 CJK + 1 词


def test_no_compress_below_threshold():
    async def run():
        store, mem = FakeSessionStore(), FakeMemoryBackend()
        conv = await store.get_or_create_conversation("u1", "s1")
        cfg = {"window_size": 10, "budget_tokens": 100000, "trigger_ratio": 0.8,
               "max_compressions": 5}
        for i in range(3):
            await store.append_message(conv["id"], "user", f"问题{i}", estimate_tokens(f"问题{i}"))
        before = len(await store.load_window(conv["id"], 100))
        await maybe_compress(store, mem, conv, cfg)
        assert len(await store.load_window(conv["id"], 100)) == before   # 未到阈值不裁剪
        assert conv["summary"] == ""                                     # 不摘要
        assert await mem.get_all("u1") == []                             # 不抽事实
    asyncio.run(run())


def test_compress_extracts_facts_then_trims(monkeypatch):
    async def fake_summarize(prev, turns):
        return "压缩摘要"

    async def run():
        monkeypatch.setattr(compressor, "recursive_summarize", fake_summarize)
        store, mem = FakeSessionStore(), FakeMemoryBackend()
        conv = await store.get_or_create_conversation("u1", "s1")
        cfg = {"window_size": 2, "budget_tokens": 100, "trigger_ratio": 0.8,
               "max_compressions": 5}
        long = "电" * 50                                    # 50 token/条
        for _ in range(6):                                  # 300 token > 80 触发
            await store.append_message(conv["id"], "user", long, estimate_tokens(long))
        await maybe_compress(store, mem, conv, cfg)
        assert len(await store.load_window(conv["id"], 100)) <= 2   # 裁剪到最新 window_size 条
        assert conv["summary"] == "压缩摘要"
        assert conv["compress_count"] == 1
        assert await mem.get_all("u1") != []                 # 丢弃前事实已进 mem0
    asyncio.run(run())


def test_cap_after_max_compressions(monkeypatch):
    calls = {"n": 0}

    async def fake_summarize(prev, turns):
        calls["n"] += 1
        return "摘要"

    async def run():
        monkeypatch.setattr(compressor, "recursive_summarize", fake_summarize)
        store, mem = FakeSessionStore(), FakeMemoryBackend()
        conv = await store.get_or_create_conversation("u1", "s1")
        cfg = {"window_size": 2, "budget_tokens": 100, "trigger_ratio": 0.8,
               "max_compressions": 2}
        long = "电" * 50
        for _ in range(3):                                  # 触发 3 轮，只允许摘要 2 次
            for _ in range(6):
                await store.append_message(conv["id"], "user", long, estimate_tokens(long))
            await maybe_compress(store, mem, conv, cfg)
        assert calls["n"] == 2                              # 压满 2 次后不再摘要
        assert conv["compress_count"] == 2
        assert len(await store.load_window(conv["id"], 100)) <= 2
        assert await mem.get_all("u1") != []                # 事实持续兜底（压满后仍抽）
    asyncio.run(run())


def test_fail_closed_memory_add_error_preserves_window():
    class Boom(FakeMemoryBackend):
        async def add(self, user_id, messages):
            raise RuntimeError("mem0 down")

    async def run():
        store, mem = FakeSessionStore(), Boom()
        conv = await store.get_or_create_conversation("u1", "s1")
        cfg = {"window_size": 2, "budget_tokens": 100, "trigger_ratio": 0.8,
               "max_compressions": 5}
        long = "电" * 50
        for _ in range(6):
            await store.append_message(conv["id"], "user", long, estimate_tokens(long))
        await maybe_compress(store, mem, conv, cfg)
        # 事实抽取失败 -> 整体放弃：不裁剪、不摘要，原文保留
        assert len(await store.load_window(conv["id"], 100)) == 6
        assert conv["summary"] == ""
    asyncio.run(run())


def test_compress_when_window_below_size_but_over_budget(monkeypatch):
    """窗口条数未满 window_size、但单条超长压爆 token 阈值时，也要真正挤出最旧 1 条并裁剪。

    回归：曾因 trim 用 window_size（> 当前条数）导致 overflow 没被删、每轮重复触发。
    """
    async def fake_summarize(prev, turns):
        return "摘要"

    async def run():
        monkeypatch.setattr(compressor, "recursive_summarize", fake_summarize)
        store, mem = FakeSessionStore(), FakeMemoryBackend()
        conv = await store.get_or_create_conversation("u1", "s1")
        cfg = {"window_size": 10, "budget_tokens": 100, "trigger_ratio": 0.8,
               "max_compressions": 5}
        await store.append_message(conv["id"], "user", "短问题", estimate_tokens("短问题"))
        await store.append_message(conv["id"], "assistant", "电" * 100, 100)  # 100 token > 80 触发
        await maybe_compress(store, mem, conv, cfg)
        assert len(await store.load_window(conv["id"], 100)) == 1   # 真正挤掉最旧 1 条
        assert await mem.get_all("u1") != []                        # 丢弃前事实已抽进 mem0
        assert conv["summary"] == "摘要"
    asyncio.run(run())
