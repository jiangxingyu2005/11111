"""接入自有模型的自检 CLI —— 先验证，再写进配置。

为什么需要：
    接入自有模型最常见的失败不是"地址不通"，而是**地址通了但字段对不上**：
    你按 OpenAI 形状去解析一个 TEI/Ollama/自研服务，每个请求都失败，
    而服务端日志只看到一堆 200（它觉得自己正常回了的）。
    所以必须跑一次真实调用，验证"取得到向量/分数/文本"。

用法：
    # 1) 自检当前配置（读 config/retrieval.yaml + env）
    python scripts/check_provider.py

    # 2) 拿到一个新服务地址，先验再配（不写文件、不重启）
    python scripts/check_provider.py --kind embed --url http://10.0.0.5:8080 --preset tei_embed
    python scripts/check_provider.py --kind rerank --url http://10.0.0.5:8080/rerank --preset tei_rerank
    python scripts/check_provider.py --kind llm --url http://10.0.0.5:9000/v1 --model qwen2.5-14b-instruct

    # 3) 自研服务：手写字段映射
    python scripts/check_provider.py --kind embed --url http://my/encode \
        --request-json '{"texts_key":"sentences"}' \
        --response-json '{"dense":"vectors.*"}'

退出码：0 全通过 / 1 有失败。CI 里直接拿来当接入门禁。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))

from providers import (  # noqa: E402
    BACKEND_FALLBACK, EmbedProvider, LLMProvider, RerankProvider,
    describe, load_endpoint, mask_url,
)


def _backend_of(kind: str, default: str) -> str:
    v = os.getenv(f"{kind.upper()}_BACKEND") or os.getenv(
        {"embed": "EMBED_BACKEND", "rerank": "RERANK_BACKEND", "llm": "LLM_BACKEND"}[kind])
    if v:
        return "openai" if v == "vllm" else v
    try:
        import yaml
        with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return ((y.get("models") or {}).get(kind) or {}).get("backend", default)
    except Exception:
        return default


def _model_id(kind: str) -> str:
    try:
        import yaml
        with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return ((y.get("models") or {}).get(kind) or {}).get("id", "")
    except Exception:
        return ""


def _expect_dim() -> int:
    try:
        import yaml
        with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return int((y.get("models") or {}).get("embed", {}).get("dim", 1024))
    except Exception:
        return 1024


async def probe_embed(over: dict) -> tuple[bool, str]:
    backend = over.get("backend") or _backend_of("embed", "hash")
    cfg = load_endpoint("embed", BACKEND_FALLBACK.get(backend, "local"))
    for k in ("url", "model", "api_key", "preset"):
        if over.get(k):
            setattr(cfg, k, over[k])
    if over.get("headers"):
        cfg.headers.update(over["headers"])
    if over.get("request"):
        cfg.request.update(over["request"])
    if over.get("response"):
        cfg.response.update(over["response"])
    if backend in ("openai", "generic") and not cfg.url:
        return False, f"backend={backend} 但未给 --url（也未配 endpoints.embed.url / EMBED_URL）"
    cfg.protocol = BACKEND_FALLBACK.get(backend, "local") if not cfg.url else cfg.protocol
    p = EmbedProvider(backend, cfg, over.get("model") or _model_id("embed"), _expect_dim())
    try:
        out = await p.embed(["接入自检：工作在 X 波段的舰载火控雷达"], "query")
        sparse = out["embeddings"][0]["sparse"] is not None
        msg = (f"dim={out['dim']}（声明 {p.expect_dim}）sparse={'有' if sparse else '无'} "
               f"{out['elapsed_ms']}ms  via {mask_url(cfg.url) or 'local:' + backend}")
        if not sparse:
            msg += " —— 该服务不返回稀疏权重，sparse 路需关（改 milvus.sparse.enabled=false）"
        return True, msg
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


async def probe_rerank(over: dict) -> tuple[bool, str]:
    backend = over.get("backend") or _backend_of("rerank", "hash")
    cfg = load_endpoint("rerank", BACKEND_FALLBACK.get(backend, "local"))
    for k in ("url", "model", "api_key", "preset"):
        if over.get(k):
            setattr(cfg, k, over[k])
    if over.get("headers"):
        cfg.headers.update(over["headers"])
    if over.get("request"):
        cfg.request.update(over["request"])
    if over.get("response"):
        cfg.response.update(over["response"])
    if backend == "generic" and not cfg.url:
        return False, "backend=generic 但未给 --url（也未配 endpoints.rerank.url / RERANK_URL）"
    cfg.protocol = BACKEND_FALLBACK.get(backend, "local") if not cfg.url else cfg.protocol
    p = RerankProvider(backend, cfg, over.get("model") or _model_id("rerank"))
    try:
        out = await p.rerank("X 波段火控雷达", ["9.5GHz 线性调频脉冲", "某型驱逐舰舷号"])
        return True, f"{len(out['scores'])} 条分数 示例={out['scores'][:2]}  " \
                     f"via {mask_url(cfg.url) or 'local:' + backend}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


async def probe_llm(over: dict) -> tuple[bool, str]:
    backend = over.get("backend") or _backend_of("llm", "echo")
    cfg = load_endpoint("llm", BACKEND_FALLBACK.get(backend, "local"))
    for k in ("url", "model", "api_key"):
        if over.get(k):
            setattr(cfg, k, over[k])
    if over.get("headers"):
        cfg.headers.update(over["headers"])
    if backend in ("echo", "hf"):
        return True, f"backend={backend} 为进程内实现，无需外部地址（生产请切 openai/generic）"
    if not cfg.url:
        return False, f"backend={backend} 但未给 --url（也未配 endpoints.llm.url / LLM_URL / LLM_UPSTREAM）"
    import time
    import httpx
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=min(cfg.timeout_s, 60.0)) as cli:
            r = await cli.post(cfg.url.rstrip("/") + "/chat/completions",
                               json={"model": cfg.model or _model_id("llm"),
                                     "messages": [{"role": "user", "content": "说一句话"}],
                                     "stream": False},
                               headers={"Authorization": f"Bearer {cfg.api_key or 'EMPTY'}"})
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        txt = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        # 只查连通会漏掉"通了但字段名不对"，这里必须验证真取得到内容
        if not txt.strip():
            return False, "HTTP 200 但取不到 choices[0].message.content —— 响应形状不是 OpenAI 兼容"
        return True, f"HTTP 200 样本={txt[:30]!r} {int((time.time()-t0)*1000)}ms via {mask_url(cfg.url)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


async def main_async(args) -> int:
    over = {"url": args.url, "preset": args.preset, "model": args.model,
            "api_key": args.api_key, "backend": args.backend,
            "headers": json.loads(args.headers_json) if args.headers_json else {},
            "request": json.loads(args.request_json) if args.request_json else {},
            "response": json.loads(args.response_json) if args.response_json else {}}

    print("=== 当前接入配置（自省，已脱敏）===")
    for kind in ("embed", "rerank", "llm"):
        cfg = load_endpoint(kind, "local")
        backend = _backend_of(kind, {"embed": "hash", "rerank": "hash", "llm": "echo"}[kind])
        print(f"  {kind:<7} backend={backend:<8} {json.dumps(describe(cfg), ensure_ascii=False)}")
    print()

    kinds = [args.kind] if args.kind else ["embed", "rerank", "llm"]
    fn = {"embed": probe_embed, "rerank": probe_rerank, "llm": probe_llm}
    failed = []
    for k in kinds:
        ok, msg = await fn[k](over if args.kind else {})
        print(f"[{'OK' if ok else 'FAIL'}] {k:<7} {msg}")
        if not ok:
            failed.append(k)

    if failed:
        print(f"\n[FAIL] 未通过：{failed}")
        print("提示：字段对不上就换 preset 或手写 --request-json/--response-json；"
              "可用 presets 见 config/retrieval.yaml:presets")
        return 1
    print("\n[OK] 接入自检通过 —— 可以把地址写进 config/retrieval.yaml:endpoints 了")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["embed", "rerank", "llm"], default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--preset", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--backend", default="")
    ap.add_argument("--headers-json", default="")
    ap.add_argument("--request-json", default="")
    ap.add_argument("--response-json", default="")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
