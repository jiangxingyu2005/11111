"""E2 LLM 流式透传测试 —— 用本地测试替身验证 SSE 是真的逐块透传。

为什么要有这个测试：
  model_stub 的 openai 分支曾经用 httpx.post 收流式请求，上游回的是 SSE，
  r.json() 必然抛错 -> 500 -> 上层 catch 后记 generation_unavailable。
  表现为"配了真 LLM 却永远降级"，日志只有一句 warning。
  所以这里用【最小 OpenAI 兼容替身】验证：流式必须逐块到达，不能攒齐再解析。

  替身只模拟协议，不模拟智能——它证明的是链路，不是报告质量。
"""
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from httpx import Client as HTTPXClient

ROOT = Path(__file__).resolve().parents[2]   # tests/generate/ -> tests/ -> 仓库根
for p in (str(ROOT), str(ROOT / "services")):
    if p not in sys.path:
        sys.path.insert(0, p)

UPSTREAM_PORT = 8077
FULL = "电磁信号研判：依据[1]证据，该辐射源工作于X波段。"


def _upstream_app() -> FastAPI:
    """最小 OpenAI 兼容替身：真 SSE，逐字吐，末尾 [DONE]。"""
    app = FastAPI()

    @app.get("/models")
    def models():
        return {"object": "list", "data": [{"id": "test-double", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: dict):
        model = req.get("model", "test-double")
        if not req.get("stream"):        # 非流式：返回完整 JSON（真上游也是这个形状）
            return {"choices": [{"message": {"role": "assistant", "content": FULL},
                                 "index": 0, "finish_reason": "stop"}],
                    "model": model, "object": "chat.completion"}

        def gen():
            for ch in FULL:
                body = {"choices": [{"delta": {"content": ch}, "index": 0,
                                     "finish_reason": None}], "model": model}
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


@pytest.fixture(scope="module")
def upstream_url():
    """上游替身只起一次：重复绑端口会产生 uvicorn 启动失败的线程噪音。"""
    cfg = uvicorn.Config(_upstream_app(), host="127.0.0.1", port=UPSTREAM_PORT,
                         log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    url = f"http://127.0.0.1:{UPSTREAM_PORT}/v1"
    for _ in range(80):
        try:
            HTTPXClient(timeout=1.0).get(url.rstrip("/v1") + "/models")
            return url
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"测试替身起不来 port={UPSTREAM_PORT}")


def stub_with(backend: str, upstream: str = ""):
    """按指定后端重载 model_stub（后端是模块级常量，只能靠重载切换）。"""
    import os
    os.environ["LLM_BACKEND"] = backend
    os.environ["EMBED_BACKEND"] = "hash"
    os.environ["RERANK_BACKEND"] = "hash"
    if upstream:
        os.environ["LLM_UPSTREAM"] = upstream
    import model_stub
    return importlib.reload(model_stub)


def test_openai_backend_streams_chunks_not_one_blob(upstream_url):
    """核心断言：流式必须收到多个 chunk；整段返回说明根本没流起来。"""
    stub = stub_with("openai", upstream_url)
    chunks, done_seen = [], False
    with TestClient(stub.app) as cli:
        with cli.stream("POST", "/v1/chat/completions",
                        json={"model": "test-double", "stream": True,
                              "messages": [{"role": "user", "content": "hi"}]}) as r:
            assert r.status_code == 200, f"HTTP {r.status_code}"
            for line in r.iter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    done_seen = True
                    continue
                assert line.startswith("data: "), f"非 SSE 行: {line[:40]}"
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    chunks.append(delta["content"])

    assert len(chunks) >= 3, f"只收到 {len(chunks)} 个 chunk，不是真流式"
    assert "".join(chunks) == FULL, "透传内容被改写"
    assert done_seen, "缺少 data: [DONE] 结束标记"


def test_openai_non_stream_returns_json(upstream_url):
    """非流式请求不能被塞进 SSE 管道。"""
    stub = stub_with("openai", upstream_url)
    with TestClient(stub.app) as cli:
        r = cli.post("/v1/chat/completions",
                     json={"model": "test-double", "stream": False,
                           "messages": [{"role": "user", "content": "hi"}]})
    # 替身实现了非流式：要求原样拿到 JSON 而不是把 SSE 当 JSON 解析后崩掉
    assert r.status_code == 200, f"HTTP {r.status_code}"
    assert r.headers.get("content-type", "").startswith("application/json")
    assert r.json()["choices"][0]["message"]["content"] == FULL


def test_echo_backend_is_clearly_marked():
    """echo 通道必须自带 stub 标注，禁止被误当真实模型输出。"""
    stub = stub_with("echo")
    with TestClient(stub.app) as cli:
        r = cli.post("/v1/chat/completions",
                     json={"model": "x", "stream": False,
                           "messages": [{"role": "user", "content": "[1] 证据文本"}]})
    assert r.status_code == 200
    assert "【stub 生成" in r.json()["choices"][0]["message"]["content"]


def test_model_info_reports_backend_and_source(upstream_url):
    """启动后必须能一眼看出"当前到底是什么后端、配置从哪来、连的哪个地址"。

    排查"到底接没接对"靠翻日志太慢且容易看漏，所以必须有个自省接口。
    """
    stub = stub_with("openai", upstream_url)
    with TestClient(stub.app) as cli:
        info = cli.get("/v1/model-info").json()
    llm = info["llm"]
    assert llm["backend"] == "openai", llm
    assert llm["source"] == "env", "env 显式设置必须标记来源为 env"
    assert llm["protocol"] == "openai"
    assert "127.0.0.1" in llm["url"], f"地址应脱敏回显以便确认：{llm}"


def test_model_info_never_leaks_api_key(upstream_url):
    """自省接口绝不能回显密钥——这类接口常被贴到工单里。"""
    import os
    os.environ["LLM_API_KEY"] = "sk-should-not-appear"
    stub = stub_with("openai", upstream_url)
    with TestClient(stub.app) as cli:
        blob = cli.get("/v1/model-info").text
    assert "sk-should-not-appear" not in blob
    os.environ.pop("LLM_API_KEY", None)
