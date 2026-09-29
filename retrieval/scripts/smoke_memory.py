"""记忆模块 smoke 验证：fake 后端 + 模拟数据，跑通多轮/隔离/压缩/事实检索全链路。

与 smoke.py 的区别：记忆写侧是 fire-and-forget 后台任务，HTTP 层难以同步断言，
故这里直调 memory_service 的两个真实入口（load_context 读 / commit_turn 写）。
fake 通道无 LLM，摘要用进程内假替身返回固定串；其余（窗口/隔离/压缩/事实抽取/检索）全走真实代码。

用法：python scripts/smoke_memory.py
退出码：0=全绿；非 0=有失败（CI 可直接用）
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_BASE_ENV = {
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "MEMORY_ENABLED": "true",
    "MEMORY_AUTH_MODE": "passthrough",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "retrieval_ro", "POSTGRES_PASSWORD": "dev_only",
    "POSTGRES_DATABASE": "qingxi",
    "MILVUS_HOST": "127.0.0.1", "MILVUS_PORT": "19530",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "dev_only",
    "GRAPHRAG_EMBEDDING_MODEL": "BAAI/bge-m3",
    "PYTHONIOENCODING": "utf-8",
}
for k, v in _BASE_ENV.items():
    os.environ[k] = v

from app.config import reset_caches  # noqa: E402

reset_caches()
from app.memory import compressor, service as ms  # noqa: E402
from app.memory.backend import get_memory_backend  # noqa: E402
from app.memory.session_store import get_session_store  # noqa: E402


async def _fake_summarize(prev: str, turns: list[dict]) -> str:
    """fake 通道无 LLM，用进程内替身模拟「递归摘要」（真实实现走本地 Ollama /chat/completions）。"""
    digest = "、".join(t.get("content", "")[:10] for t in turns[:2])
    return f"{prev} ▸ {digest}" if prev else f"摘要·{digest}"


async def _run() -> list[str]:
    compressor.recursive_summarize = _fake_summarize  # 假 LLM 替身
    fails: list[str] = []

    def chk(cond: bool, msg: str) -> None:
        print(("  [OK] " if cond else "  [FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    # [1] 多轮写入 + 窗口累积 + 装载
    await ms.commit_turn("u1", "s1", "有哪些 X 波段雷达信号类型？", "主要有线性调频和脉冲多普勒两类。")
    await ms.commit_turn("u1", "s1", "线性调频的带宽典型值是多少？", "典型 100MHz 到 1GHz。")
    ctx = await ms.load_context("u1", "s1", "刚才说的带宽范围再确认下")
    chk(len(ctx["window"]) == 4, f"多轮装载：window={len(ctx['window'])} 条（2 轮 × user+assistant）")
    chk(ctx["summary"] == "", "未到 token 阈值：summary 为空（不提前摘要）")

    # [2] 多用户隔离
    ctx2 = await ms.load_context("u2", "s9", "完全无关的问题")
    chk(len(ctx2["window"]) == 0, f"多用户隔离：u2 window={len(ctx2['window'])} 条（应=0，与 u1 隔离）")

    # [3] 压缩触发（长消息压爆 token 阈值 4096×0.8=3276）
    param = "该雷达信号中心频率9.4GHz、带宽500MHz、脉宽10us、重频2kHz、峰值功率100kW、天线增益35dB。"
    long_ans = param * 110  # 110×33 ≈ 3630 token，压爆 3276 阈值
    await ms.commit_turn("u1", "s1", "把参数详细列出来", long_ans)
    conv = await get_session_store().get_or_create_conversation("u1", "s1")
    win = await get_session_store().load_window(conv["id"], 1000)
    chk(conv["compress_count"] >= 1, f"token 触发压缩：compress_count={conv['compress_count']}")
    chk(bool(conv["summary"]), f"递归摘要已生成：{conv['summary'][:30]!r}...")
    chk(len(win) <= 10, f"窗口已裁剪：剩余 {len(win)} 条（≤ window_size=10）")

    # [4] 长期事实抽取 + 检索
    mem = get_memory_backend()
    facts = await mem.get_all("u1")
    chk(bool(facts), f"丢弃原文前事实已抽进 mem0：共 {len(facts)} 条")
    for f in facts[:3]:
        print(f"        · {f[:44]}")
    hits = await mem.search("雷达信号", "u1", k=3)
    chk(bool(hits), f"长期事实可检索：query='雷达信号' 命中 {len(hits)} 条")

    # [5] 压缩后三层装载齐备（注入 prompt 的三层来源）
    ctx3 = await ms.load_context("u1", "s1", "雷达信号有哪些类型")
    chk(bool(ctx3["summary"]) and bool(ctx3["window"]) and bool(ctx3["memories"]),
        "三层装载齐备：summary/window/memories 均非空")

    return fails


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fails = asyncio.run(_run())
    print("\n=== 结果 ===")
    if fails:
        print("[FAIL] " + "; ".join(fails))
        return 1
    print("[smoke_memory] 全部通过 [PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
