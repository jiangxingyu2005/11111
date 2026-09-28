"""Provider 层测试 —— 证明"填配置就能接自有模型"，且接错会被抓住。

重点不是"能发 HTTP"，而是：
  1. 不同服务形状（TEI / Ollama / OpenAI / SiliconFlow）靠字段映射都能解析
  2. 维度与声明不符必须报错（换模型忘了重建库 = 召回全是噪声）
  3. 自省接口必须脱敏（api_key 绝不能出现在响应里）
  4. 配置错（要连外部却没给 url）必须被拦在启动前
"""
import json
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from httpx import Client as HTTPXClient

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "services")):
    if p not in sys.path:
        sys.path.insert(0, p)

from providers import (  # noqa: E402
    EmbedProvider, EndpointCfg, RerankProvider, describe, dig, mask_url,
)

PORT = 8088
DIM = 1024


def _upstream() -> FastAPI:
    """四种真实服务形状的替身（只模拟协议，不模拟智能）。"""
    app = FastAPI()

    @app.post("/tei/embed")                       # TEI: {inputs:[...]} -> [[...]]
    async def tei_embed(req: dict):
        return [[0.1] * DIM for _ in req["inputs"]]

    @app.post("/ollama/api/embed")                # Ollama: {model,input} -> {embeddings:[[...]]}
    async def ollama_embed(req: dict):
        return {"embeddings": [[0.2] * DIM for _ in req["input"]]}

    @app.post("/openai/v1/embeddings")            # OpenAI: {model,input} -> {data:[{embedding}]}
    async def openai_embed(req: dict):
        n = len(req["input"])
        return {"data": [{"embedding": [0.3] * DIM, "index": i} for i in range(n)]}

    @app.post("/openai-bad/v1/embeddings")        # 维度对不上的服务
    async def openai_wrong(req: dict):
        return {"data": [{"embedding": [0.3] * 768, "index": i} for i in range(len(req["input"]))]}

    @app.post("/tei/rerank")                      # TEI: -> [{index, score}]
    async def tei_rerank(req: dict):
        return [{"index": i, "score": 0.9 - i * 0.1} for i in range(len(req["texts"]))]

    @app.post("/v1/rerank")                       # SiliconFlow: {query,documents} -> {results:[{relevance_score}]}
    async def sf_rerank(req: dict):
        return {"results": [{"index": i, "relevance_score": 0.8 - i * 0.1}
                            for i in range(len(req["documents"]))]}

    @app.post("/my/encode")                       # 自研：{sentences} -> {vectors:[[...]]}
    async def my_encode(req: dict):
        return {"vectors": [[0.4] * DIM for _ in req["sentences"]]}

    return app


@pytest.fixture(scope="module")
def base():
    cfg = uvicorn.Config(_upstream(), host="127.0.0.1", port=PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(80):
        try:
            HTTPXClient(timeout=1.0).get(url + "/openapi.json")
            return url
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("上游替身起不来")


def _cfg(url: str, protocol: str, **kw) -> EndpointCfg:
    return EndpointCfg(protocol=protocol, url=url, timeout_s=10.0, **kw)


# ────────────────────── dig() 字段路径 ──────────────────────
def test_dig_list_mapping():
    assert dig({"data": [{"embedding": [1, 2]}]}, "data.*.embedding") == [[1, 2]]
    assert dig({"embeddings": [[1, 2], [3, 4]]}, "embeddings.*") == [[1, 2], [3, 4]]
    assert dig({"results": [{"relevance_score": 0.5}]}, "results.*.relevance_score") == [0.5]
    assert dig({"a": {"b": 1}}, "a.b") == 1
    assert dig({"a": {"b": 1}}, "a.z") is None
    assert dig({"a": [1, 2, 3]}, "a.1") == 2


# ────────────────────── 各服务形状 ──────────────────────
async def _embed(protocol, url, **kw):
    p = EmbedProvider("generic" if protocol == "generic" else "openai",
                      _cfg(url, protocol, **kw), "test-model", DIM)
    return await p.embed(["甲", "乙"], "query")


def test_tei_shape_via_preset_mapping(base):
    import asyncio
    out = asyncio.run(_embed("generic", base + "/tei/embed",
                             request={"texts_key": "inputs"}, response={"dense": "*"}))
    assert out["dim"] == DIM and out["count"] == 2
    assert out["embeddings"][0]["dense"][0] == pytest.approx(0.1)


def test_ollama_shape_via_preset_mapping(base):
    import asyncio
    out = asyncio.run(_embed("generic", base + "/ollama/api/embed",
                             request={"texts_key": "input", "model_key": "model"},
                             response={"dense": "embeddings.*"}))
    assert out["count"] == 2 and out["embeddings"][0]["dense"][0] == pytest.approx(0.2)


def test_custom_service_shape(base):
    import asyncio
    out = asyncio.run(_embed("generic", base + "/my/encode",
                             request={"texts_key": "sentences"},
                             response={"dense": "vectors.*"}))
    assert out["count"] == 2 and out["embeddings"][0]["dense"][0] == pytest.approx(0.4)


def test_openai_shape_and_sparse_is_null_not_fake(base):
    """OpenAI 标准没有稀疏权重：必须如实给 null，不能伪造空字典。"""
    import asyncio
    # openai 协议下 Provider 会自动拼 /embeddings，所以这里只给到 /v1
    out = asyncio.run(_embed("openai", base + "/openai/v1", model="m"))
    assert out["count"] == 2 and out["dim"] == DIM
    assert out["embeddings"][0]["sparse"] is None


def test_dim_mismatch_is_rejected(base):
    """换模型忘了重建库 = 召回全是噪声，必须在接入那一刻就炸。"""
    import asyncio
    with pytest.raises(ValueError, match="维度"):
        asyncio.run(_embed("openai", base + "/openai-bad/v1", model="m"))


def test_rerank_tei_shape(base):
    import asyncio
    p = RerankProvider("generic", _cfg(base + "/tei/rerank", "generic",
                                       response={"scores": "*.score"}), "m")
    out = asyncio.run(p.rerank("q", ["a", "b", "c"]))
    assert out["scores"] == pytest.approx([0.9, 0.8, 0.7])


def test_rerank_siliconflow_shape(base):
    import asyncio
    p = RerankProvider("generic", _cfg(base + "/v1/rerank", "generic",
                                       request={"texts_key": "documents", "model_key": "model"},
                                       response={"scores": "results.*.relevance_score"}), "m")
    out = asyncio.run(p.rerank("q", ["a", "b"]))
    assert out["scores"] == pytest.approx([0.8, 0.7])


# ────────────────────── 脱敏与配置校验 ──────────────────────
def test_describe_never_leaks_secret():
    cfg = _cfg("https://api.example.com/v1?key=abc", "openai",
               api_key="sk-super-secret",
               headers={"Authorization": "Bearer sk-super-secret", "X-Trace": "1"})
    d = describe(cfg)
    blob = json.dumps(d, ensure_ascii=False)
    assert "sk-super-secret" not in blob, f"自省接口泄漏密钥：{blob}"
    assert "key=abc" not in blob, "URL 查询串必须脱敏"
    assert d["has_api_key"] is True           # 有无 key 要能看，值不能看
    assert d["header_keys"] == ["Authorization", "X-Trace"]


def test_mask_url_keeps_only_origin():
    assert mask_url("https://a.b:9000/v1/chat?k=1") == "https://a.b:9000/…"
    assert mask_url("") == ""


def test_missing_url_for_external_backend_is_rejected():
    """要连外部却没给地址 —— 必须在启动校验里就拦住。"""
    from app.config import validate_model_contract
    cfg = {"models": {"embed": {"id": "BAAI/bge-m3", "dim": 1024, "backend": "generic"},
                      "rerank": {"id": "bge-reranker-v2-m3", "backend": "hash"},
                      "llm": {"id": "q", "backend": "echo"}},
           "endpoints": {"embed": {"url": ""}}, "reranker": {"model": "bge-reranker-v2-m3"},
           "milvus": {"sparse": {"enabled": False}}}
    problems = validate_model_contract(cfg, "fake", False)
    assert any("endpoints.embed.url" in p for p in problems), problems
