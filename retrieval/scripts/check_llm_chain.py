"""E2 报告生成链路验收：证明「SSE 真流式 + 证据 [n] 拼接」这条链路是通的。

验什么（四项硬断言）：
  1. 上游可达性  —— openai 后端连不上就直接 FAIL，不许"降级了还报绿"
  2. 真流式      —— 必须收到 >=3 个 chunk；整段一次返回说明根本没流起来
  3. 证据进得去  —— 输出里出现证据编号 [1]（build_prompt 编号与 results 顺序一致）
  4. 标注不可混  —— echo 通道必须带"【stub 生成"标注；openai 通道绝不能带

为什么单独一个脚本：
  model_stub 的 openai 分支曾经是坏的——用 httpx.post 收流式请求，
  上游回 SSE、r.json() 必然抛错，上层 catch 后记 generation_unavailable。
  表现为"配了真 LLM 却永远降级"，日志只有一句 warning。所以必须有可执行断言。

用法：
  # 1) CPU 小模型（llama.cpp server / ollama，OpenAI 兼容）
  python scripts/check_llm_chain.py --backend openai --upstream http://127.0.0.1:8080/v1

  # 2) 无模型自检：只验证 harness 本身好使（echo 模板流）
  python scripts/check_llm_chain.py --backend echo

  # 3) 生产 vLLM（有 GPU 的机器上）
  python scripts/check_llm_chain.py --backend openai --upstream http://gpu-box:9000/v1 \
      --model qwen2.5-14b-instruct

纪律：
  这是【链路验收】，不是报告质量验收。CPU 小模型能证明链路通，
  不能证明报告可信——后者要人工抽检（E2 完整验收口径），且明确标注为逻辑级替代。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUB_PORT = 8013
SEED = ROOT / "data" / "seed"
STUB_MARK = "【stub 生成"
MAX_NEW_TOKENS = 256          # main() 按 --max-new-tokens 覆盖


def start_stub(backend: str, upstream: str, model: str) -> None:
    os.environ["LLM_BACKEND"] = backend
    os.environ["EMBED_BACKEND"] = "hash"
    os.environ["RERANK_BACKEND"] = "hash"
    if backend == "hf":
        os.environ["HF_MODEL"] = model or "Qwen/Qwen2.5-0.5B-Instruct"
        os.environ["HF_MAX_NEW_TOKENS"] = str(MAX_NEW_TOKENS)
    if upstream:
        os.environ["LLM_UPSTREAM"] = upstream
    if model and backend == "openai":
        os.environ["LLM_UPSTREAM_MODEL"] = model

    import uvicorn
    sys.path.insert(0, str(ROOT / "services"))
    from model_stub import app  # noqa: E402
    cfg = uvicorn.Config(app, host="127.0.0.1", port=STUB_PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()

    import httpx
    for _ in range(60):
        try:
            if httpx.get(f"http://127.0.0.1:{STUB_PORT}/healthz", timeout=1.0).status_code == 200:
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("model_stub 启动失败")


def probe_upstream(upstream: str, model: str) -> tuple[bool, str]:
    """探活上游——必须做【真实 completion】，不能只查 /models。

    踩过的坑：ollama 某个残缺安装 /models 返回 200 且列出模型，
    但真正推理时 500（llama-server 二进制不存在）。只查 /models 会误判为可用，
    然后一路绿到生成环节才炸——正是"看似正常"的那类错误。
    """
    import httpx
    try:
        r = httpx.get(upstream.rstrip("/") + "/models", timeout=5.0)
        ids = [m.get("id") for m in (r.json().get("data") or [])] if r.status_code == 200 else []
    except Exception as e:
        return False, f"/models 不可达: {str(e)[:120]}"
    try:
        payload = {"messages": [{"role": "user", "content": "说一句话"}], "stream": False}
        if model:
            payload["model"] = model
        c = httpx.post(upstream.rstrip("/") + "/chat/completions", json=payload, timeout=120.0)
        if c.status_code != 200:
            return False, f"推理返回 HTTP {c.status_code}: {c.text[:160]}"
        txt = (c.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not txt.strip():
            return False, "推理返回空内容"
        return True, f"models={ids[:3]} 推理OK 样本={txt[:24]!r}"
    except Exception as e:
        return False, f"推理失败: {str(e)[:160]}"


def build_fake_hits(n: int = 5):
    """从 seed 直接造命中，避免依赖完整检索链路（本脚本只验生成侧）。"""
    from app.fuse.rrf import Hit
    chunks = json.loads((SEED / "chunks.json").read_text(encoding="utf-8"))
    parents = {p["parent_id"]: p
               for p in json.loads((SEED / "parents.json").read_text(encoding="utf-8"))}
    hits = [Hit(child_id=c["child_id"], parent_id=c["parent_id"], source="vector_hybrid",
                rank_score=1.0 - i * 0.1, content=c["content"],
                meta={"kb_id": c.get("kb_id", "")})
            for i, c in enumerate(chunks[:n])]
    return hits, parents


async def run_stream(query: str, hits, parents):
    from app.config import get_settings, reset_caches
    reset_caches()
    get_settings()
    from app.degrade.orchestrator import DegradedCollector
    from app.generate.report import stream_report

    col = DegradedCollector()
    toks, t0 = [], time.time()
    first = None
    async for tok in stream_report(query, hits, parents, col):
        if first is None:
            first = (time.time() - t0) * 1000
        toks.append(tok)
    return toks, first, (time.time() - t0) * 1000, col.as_list()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["echo", "openai", "hf"], default="echo")
    ap.add_argument("--hf-model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
                    help="hf 后端用的模型（CPU 建议 0.5B~3B；国内走 hf-mirror）")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080/v1",
                    help="OpenAI 兼容上游：llama.cpp server / ollama / vLLM")
    ap.add_argument("--model", default="", help="上游实际模型名（llama.cpp 常为 local-model）")
    ap.add_argument("--query", default="工作在X波段、采用线性调频的舰载火控雷达有哪些特征？")
    args = ap.parse_args()

    if not (SEED / "chunks.json").exists():
        raise SystemExit("seed 缺失 —— 先跑 python scripts/gen_seed.py")

    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens

    # 把 retrieval 指向进程内 stub。注意：这些必须显式设进 environ，
    # 否则 pydantic-settings 会读仓库根 .env 兜底（指向 8001），
    # 表现为"连错服务 -> 502 -> 记降级"，看着像模型挂了其实是配置没生效。
    os.environ.update({
        "RETRIEVAL_BACKEND": "fake",
        "RETRIEVAL_SEED_DIR": "data/seed",
        "GRAPHRAG_LLM_BASE_URL": f"http://127.0.0.1:{STUB_PORT}/v1",
        "GRAPHRAG_LLM_API_KEY": "EMPTY",
        "GRAPHRAG_LLM_MODEL": args.model or ("hf-local" if args.backend == "hf" else "stub-echo"),
        "RETRIEVAL_EMBED_URL": f"http://127.0.0.1:{STUB_PORT}/embed",
        "RETRIEVAL_RERANK_SERVICE_URL": f"http://127.0.0.1:{STUB_PORT}/rerank",
        "RETRIEVAL_TRACE_EXPORT": "file", "RETRIEVAL_TRACE_DIR": "logs/trace_llm",
        "PYTHONIOENCODING": "utf-8",
    })

    hf_note = f"  hf_model={args.hf_model}" if args.backend == "hf" else ""
    print(f"[E2] backend={args.backend}  "
          f"upstream={args.upstream if args.backend == 'openai' else '-'}{hf_note}")

    if args.backend == "openai":
        ok, info = probe_upstream(args.upstream, args.model)
        print(f"[E2] 上游探活：{'OK' if ok else 'FAIL'}  {info}")
        if not ok:
            print("[FAIL] 上游不可达。注意：探活做的是【真实 completion】，"
                  "不是只查 /models（后者会漏掉'列得出模型但推理 500'的残缺环境）")
            print("   起法二选一：")
            print("   llama.cpp: ./llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf "
                  "--port 8080 --host 127.0.0.1")
            print("   ollama   : ollama serve  &&  ollama pull qwen2.5:3b "
                  "（端点 http://127.0.0.1:11434/v1）")
            print("   若本机连推理栈都没有，改用进程内 HF 后端："
                  "--backend hf --hf-model Qwen/Qwen2.5-0.5B-Instruct")
            print("   连不上就必须报 FAIL —— 降级了还报绿比报错更危险")
            return 1

    # hf 后端吃本地路径，openai 后端吃上游模型名——别混用，否则会静默回落到默认 repo id
    start_stub(args.backend, args.upstream,
               args.hf_model if args.backend == "hf" else args.model)
    hits, parents = build_fake_hits()
    toks, first_ms, total_ms, degraded = asyncio.run(
        run_stream(args.query, hits, parents))
    text = "".join(toks)

    ft = f"{first_ms:.0f}ms" if first_ms is not None else "无(未收到任何 token)"
    print(f"[E2] chunks={len(toks)}  首token={ft}  总耗时={total_ms:.0f}ms  "
          f"字符数={len(text)}")
    print(f"[E2] degraded={[d['code'] for d in degraded]}")
    print("[E2] 输出前 120 字：" + text[:120].replace("\n", " / "))

    fails = []
    if len(toks) < 3:
        fails.append(f"只收到 {len(toks)} 个 chunk，不是真流式（整段返回说明流没起来）")
    if not text.strip():
        fails.append("输出为空")
    if "[1]" not in text:
        fails.append("输出未出现证据编号 [1] —— build_prompt 的证据没送进模型")
    if args.backend == "echo" and STUB_MARK not in text:
        fails.append("echo 通道缺少【stub 生成】标注 —— 可能被误当真实模型输出")
    if args.backend == "openai" and STUB_MARK in text:
        fails.append("openai 通道却出现 stub 标注 —— 说明请求没打到上游，仍走的 echo 分支")
    if degraded and any(d["code"] == "generation_unavailable" for d in degraded):
        fails.append("降级码 generation_unavailable —— 生成链路没真正跑通")

    if fails:
        print("[FAIL] " + "；".join(fails))
        return 1
    note = ("真模型链路已通；注意 CPU 首 token 含模型冷加载，"
            "生产必须用 GPU vLLM 才进得了 5.2s 预算"
            if args.backend in ("openai", "hf") else "echo 为模板流，仅自检 harness")
    print(f"[OK] 生成链路通过（{note}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
