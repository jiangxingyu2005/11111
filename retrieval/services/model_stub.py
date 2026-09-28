"""模型服务（默认 8001）——对外暴露稳定的模型接入接口。

对外端点（形状固定，换任何模型都不变）：
    POST /embed                 {texts, type} -> {embeddings:[{dense, sparse}], dim, model}
    POST /rerank                {query, texts}-> {scores:[...], model}
    POST /v1/chat/completions   OpenAI 兼容，stream=true 时 SSE
    GET  /v1/models             探测上游可用模型
    GET  /v1/model-info         【自省】当前三类能力接的是什么、从哪来、能不能连通
    POST /v1/probe              【自检】不重启就能验证某个能力的接入配置对不对
    GET  /healthz               存活

接入自有模型：改 config/retrieval.yaml 的 models.<kind>.backend + endpoints.<kind>，
或直接用环境变量覆盖（EMBED_URL / RERANK_URL / LLM_URL ...），**不需要改本文件**。
详见 docs/接入自有模型.md 与 scripts/check_provider.py。

纪律：
  - 模型身份的唯一登记处是 config/retrieval.yaml:models；本文件只消费，不另立门户
  - 自省接口一律脱敏：api_key 与自定义头的值绝不出现在响应里
  - 拿不到 sparse 就返回 null，绝不伪造；嵌入维度与声明不符直接报错
启动：python services/model_stub.py --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))

from providers import (  # noqa: E402
    BACKEND_FALLBACK, EmbedProvider, EndpointCfg, LLMProvider, RerankProvider,
    describe, load_endpoint, mask_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("model_stub")

app = FastAPI(title="model-service", version="2.0.0")


# ────────────────────────── 后端解析 ──────────────────────────
def _resolve_backend(env_key: str, kind: str, default: str) -> tuple[str, str]:
    """env > yaml；返回 (值, 来源)。来源要能自省，否则排查时不知道配置到底从哪来。"""
    v = os.getenv(env_key)
    if v:
        return v, "env"
    try:
        import yaml
        with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        b = ((cfg.get("models") or {}).get(kind) or {}).get("backend")
        if b:
            return b, "yaml"
    except Exception as e:
        log.error("读 config/retrieval.yaml 失败（%s），后端回退默认 %s", e, default)
    return default, "default"


EMBED_BACKEND, _src_embed = _resolve_backend("EMBED_BACKEND", "embed", "hash")
RERANK_BACKEND, _src_rerank = _resolve_backend("RERANK_BACKEND", "rerank", "hash")
LLM_BACKEND, _src_llm = _resolve_backend("LLM_BACKEND", "llm", "echo")
if LLM_BACKEND == "vllm":                       # 历史别名
    LLM_BACKEND = "openai"

_cfg_cache: dict[str, tuple[str, EndpointCfg]] = {}


def _providers() -> tuple[EmbedProvider, RerankProvider, LLMProvider]:
    """按当前配置构造三个 Provider（配置可被 POST /v1/probe 临时改写后重建）。"""
    import yaml
    with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    m = y.get("models") or {}
    expect_dim = int((m.get("embed") or {}).get("dim", 1024))

    def mk(kind: str, backend: str) -> EndpointCfg:
        c = load_endpoint(kind, BACKEND_FALLBACK.get(backend, "local"))
        # backend 决定协议；只有在"确实要连外部"时才让 preset/显式 protocol 生效，
        # 避免本地后端被一个残留的 preset 悄悄改成去打 HTTP（那正是静默漂移）
        if c.protocol in ("openai", "generic") and c.url:
            return c
        c.protocol = BACKEND_FALLBACK.get(backend, "local")
        return c

    ec = mk("embed", EMBED_BACKEND)
    rc = mk("rerank", RERANK_BACKEND)
    lc = mk("llm", LLM_BACKEND)

    eid = os.getenv("EMBED_MODEL") or (m.get("embed") or {}).get("id", "BAAI/bge-m3")
    rid = os.getenv("RERANK_MODEL") or (m.get("rerank") or {}).get("id", "bge-reranker-v2-m3")
    lid = os.getenv("HF_MODEL") or (m.get("llm") or {}).get("id", "qwen2.5-14b-instruct")
    return (EmbedProvider(EMBED_BACKEND, ec, eid, expect_dim),
            RerankProvider(RERANK_BACKEND, rc, rid),
            LLMProvider(LLM_BACKEND, lc, lid))


def _get() -> tuple[EmbedProvider, RerankProvider, LLMProvider]:
    key = f"{EMBED_BACKEND}|{RERANK_BACKEND}|{LLM_BACKEND}"
    if key not in _cfg_cache:
        _cfg_cache[key] = _providers()
    return _cfg_cache[key]


# ────────────────────────── 请求模型 ──────────────────────────
class EmbedReq(BaseModel):
    texts: list[str]
    type: str = "doc"          # doc / query


class RerankReq(BaseModel):
    query: str
    texts: list[str]


class ChatReq(BaseModel):
    model: str = "stub"
    messages: list[dict]
    stream: bool = False
    temperature: float = 0.2


class ProbeReq(BaseModel):
    kind: str = "embed"        # embed | rerank | llm
    url: str = ""              # 留空则用当前配置；填了就临时测这个地址
    preset: str = ""
    model: str = ""
    api_key: str = ""
    headers: dict = {}
    request: dict = {}
    response: dict = {}


# ────────────────────────── 自省 / 存活 ──────────────────────────
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "embed": EMBED_BACKEND, "rerank": RERANK_BACKEND, "llm": LLM_BACKEND}


@app.get("/v1/model-info")
def model_info() -> dict:
    """自省：三类能力当前接的是什么、配置从哪来、地址脱敏、有没有配 key。

    存在的理由：排查"到底接没接对"必须一眼可查，靠翻日志太慢且容易看漏。
    """
    ep, rp, lp = _get()
    return {
        "service": "model-service", "version": "2.0.0",
        "embed": {"backend": EMBED_BACKEND, "model": ep.model_id,
                  "expect_dim": ep.expect_dim, "source": _src_embed, **describe(ep.cfg)},
        "rerank": {"backend": RERANK_BACKEND, "model": rp.model_id,
                   "source": _src_rerank, **describe(rp.cfg)},
        "llm": {"backend": LLM_BACKEND, "model": lp.model_id,
                "source": _src_llm, **describe(lp.cfg)},
        "hint": "改 config/retrieval.yaml 的 models/endpoints 或用 env 覆盖；"
                "接入后用 POST /v1/probe 自检，无需重启",
    }


# ────────────────────────── 自检 ──────────────────────────
@app.post("/v1/probe")
async def probe(req: ProbeReq) -> dict:
    """不重启验证接入配置对不对。

    验什么：连通性 + 协议/字段映射对不对 + 嵌入维度是否与声明一致 + 首字节延迟。
    为什么需要：接入自有模型最常见的失败是"URL 通了但字段对不上"，
    只查连通性会漏掉，表现为线上每个请求都失败。这里直接跑一次真实调用。
    """
    ep, rp, lp = _get()
    kind = req.kind
    if kind not in ("embed", "rerank", "llm"):
        return {"ok": False, "error": f"kind 必须是 embed|rerank|llm，收到 {kind}"}

    # 用请求里的覆盖值临时构造一份配置，不污染全局
    import copy
    base = {"embed": ep, "rerank": rp, "llm": lp}[kind]
    cfg = copy.deepcopy(base.cfg)
    if req.url:
        cfg.url = req.url
    if req.preset:
        cfg.preset = req.preset
    if req.model:
        cfg.model = req.model
    if req.api_key:
        cfg.api_key = req.api_key
    if req.headers:
        cfg.headers = {**cfg.headers, **req.headers}
    if req.request:
        cfg.request = {**cfg.request, **req.request}
    if req.response:
        cfg.response = {**cfg.response, **req.response}

    t0 = time.time()
    try:
        if kind == "embed":
            out = await EmbedProvider(base.backend, cfg, base.model_id,
                                      base.expect_dim if kind == "embed" else 0).embed(
                ["接入自检：电磁信号 X 波段"], "query")
            return {"ok": True, "kind": kind, "dim": out["dim"],
                    "expect_dim": base.expect_dim,
                    "sparse": out["embeddings"][0]["sparse"] is not None,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "endpoint": mask_url(cfg.url) or f"local:{base.backend}"}
        if kind == "rerank":
            out = await RerankProvider(base.backend, cfg, base.model_id).rerank(
                "X 波段火控雷达", ["9.5GHz 线性调频", " unrelated text "])
            return {"ok": True, "kind": kind, "n_scores": len(out["scores"]),
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "endpoint": mask_url(cfg.url) or f"local:{base.backend}"}
        # llm：非流式发一条最小请求，验通断与模型名
        if not cfg.url:
            return {"ok": False, "kind": "llm",
                    "error": f"backend={base.backend} 未配置 endpoints.llm.url（或 LLM_URL）"}
        import httpx
        async with httpx.AsyncClient(timeout=min(cfg.timeout_s, 60.0)) as cli:
            r = await cli.post(cfg.url.rstrip("/") + "/chat/completions",
                               json={"model": cfg.model or base.model_id,
                                     "messages": [{"role": "user", "content": "说一句话"}],
                                     "stream": False},
                               headers={"Authorization": f"Bearer {cfg.api_key or 'EMPTY'}"})
        return {"ok": r.status_code == 200, "kind": "llm", "http": r.status_code,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "endpoint": mask_url(cfg.url),
                "body": r.text[:200] if r.status_code != 200 else None}
    except Exception as e:
        return {"ok": False, "kind": kind, "error": f"{type(e).__name__}: {e}"[:400],
                "endpoint": mask_url(cfg.url) or f"local:{base.backend}",
                "elapsed_ms": int((time.time() - t0) * 1000)}


# ────────────────────────── 能力端点 ──────────────────────────
@app.post("/embed")
async def embed(req: EmbedReq) -> dict:
    ep, _r, _l = _get()
    return await ep.embed(req.texts, req.type or "doc")


@app.post("/rerank")
async def rerank(req: RerankReq) -> dict:
    _e, rp, _l = _get()
    return await rp.rerank(req.query, req.texts)


def _echo_answer(messages: list[dict]) -> str:
    """模板化报告：明确标注为 stub 产物，避免被误当真实模型输出（禁止编造红线）。"""
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    blocks: list[tuple[str, str]] = []
    cur = None
    for line in user.splitlines():
        if line.startswith("["):
            if cur:
                blocks.append(cur)
            cur = (line[1:line.find("]")], line[line.find("]") + 1:].strip())
        elif cur:
            cur = (cur[0], cur[1] + " " + line.strip())
    if cur:
        blocks.append(cur)
    q = user.split("用户问题：")[-1].split("\n")[0].strip() if "用户问题：" in user else ""

    lines = [f"【stub 生成 · 非真实模型输出】针对「{q}」的研判如下：", "", "一、相似信号列表"]
    for i, (_n, txt) in enumerate(blocks[:8], start=1):
        lines.append(f"[{i}] {txt[:70]}……")
    if not blocks:
        lines.append("（无命中证据，依据不足）")
    lines += [
        "",
        "二、关联路径：命中实体经图谱 1~2 跳关联，详见图谱页 focus 视图。",
        "三、规则触发详情：结构化条件按软语义参与排序，未做硬过滤（过渡期口径）。",
        "四、置信度：本段为 stub 模板输出，不代表模型判断；接入真实 LLM 后替换为证据支撑的结论。",
        "",
        "注：以上为流式链路演示文本，真实结论必须由 LLM 依据证据生成。",
    ]
    return "\n".join(lines)


@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    _e, _r, lp = _get()
    out = await lp.chat(req.model_dump())
    if out is not None:
        return out                                   # openai / hf / generic

    text = _echo_answer(req.messages)                # echo 兜底
    model = f"stub-echo"
    if not req.stream:
        return {"choices": [{"message": {"role": "assistant", "content": text},
                             "index": 0, "finish_reason": "stop"}],
                "model": model, "object": "chat.completion"}

    async def gen():
        for i in range(0, len(text), 6):
            yield _sse_chunk(model, text[i:i + 6])
            await asyncio.sleep(0.01)
        yield _sse_chunk(model, "")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse_chunk(model: str, content: str) -> str:
    import json
    body = {"choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}],
            "model": model, "object": "chat.completion.chunk"}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


@app.get("/v1/models")
async def list_models():
    """探测上游可用模型；local/echo 后端返回自身。"""
    _e, _r, lp = _get()
    if lp.cfg.protocol != "openai" or not lp.cfg.url:
        return {"object": "list", "data": [{"id": lp.model_id, "object": "model"}]}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.get(lp.cfg.url.rstrip("/") + "/models", headers=lp._headers())
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"object": "list", "data": [], "error": f"upstream unreachable: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    ep, rp, lp = _get()
    log.info("model-service starting port=%s", args.port)
    log.info("  embed  backend=%s model=%s dim_expect=%s protocol=%s url=%s src=%s",
             EMBED_BACKEND, ep.model_id, ep.expect_dim, ep.cfg.protocol,
             mask_url(ep.cfg.url) or "local", _src_embed)
    log.info("  rerank backend=%s model=%s protocol=%s url=%s src=%s",
             RERANK_BACKEND, rp.model_id, rp.cfg.protocol,
             mask_url(rp.cfg.url) or "local", _src_rerank)
    log.info("  llm    backend=%s model=%s protocol=%s url=%s src=%s",
             LLM_BACKEND, lp.model_id, lp.cfg.protocol,
             mask_url(lp.cfg.url) or "local", _src_llm)
    if EMBED_BACKEND == "hash":
        log.warning("embed=hash：链路可跑但无语义能力，验收/评测前必须切真模型")
    if LLM_BACKEND == "echo":
        log.warning("llm=echo：报告为模板文本并已显式标注，非模型输出")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
