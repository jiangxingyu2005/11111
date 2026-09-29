"""模型 Provider 层 —— 让「接入自己的模型」从改代码变成填配置。

为什么要有这一层：
    原先每个后端都是 model_stub.py 里的一个 if 分支。用户要接自有的
    嵌入 / 重排 / 生成服务，只能改代码、重新发版。现在统一成
    「声明式配置 + 三种协议」：接主流服务只需在 config/retrieval.yaml:endpoints
    填 url 和 preset，接自研服务再补一组字段路径。

三种协议：
    local    进程内加载（hash 占位 / FlagEmbedding / transformers）
    openai   OpenAI 兼容 HTTP（vLLM / Xinference / OneAPI / llama.cpp / ollama /v1）
    generic  任意 HTTP + 字段路径映射（TEI / SiliconFlow / 自研服务）

配置优先级：env > config/retrieval.yaml:endpoints > presets > 内置默认

纪律（与项目红线一致）：
    - 接进来的嵌入模型必须过 dim 校验：与 models.embed.dim 声明不符就拒绝，
      这是「换模型忘了重建库」这类静默漂移的最后一道闸
    - 拿不到 sparse 就如实返回 null，绝不伪造空字典（禁止降级到无来源）
    - 对外自省接口一律脱敏：api_key / headers 里的敏感值绝不出现在响应里
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("model_stub.providers")

# backend -> 协议（决定 Provider 怎么去连）。新增自有后端时只改这里 + 白名单。
BACKEND_FALLBACK = {"hash": "local", "bge_m3": "local", "bge": "local",
                    "hf": "local", "echo": "local",
                    "openai": "openai", "generic": "generic"}

SECRET_HINT = "***"
_MASK_KEYS = ("api_key", "apikey", "authorization", "token", "password")


# ────────────────────────── 字段路径取值 ──────────────────────────
def dig(obj, path: str | None):
    """按点号路径取值，`*` 表示对列表逐元素映射。

    dig({"data":[{"embedding":[1,2]}]}, "data.*.embedding") -> [[1,2]]
    dig({"embeddings":[[1,2],[3,4]]}, "embeddings.*")       -> [[1,2],[3,4]]
    """
    if not path:
        return None
    parts = path.split(".")

    def walk(cur, i: int):
        if cur is None or i >= len(parts):
            return cur
        p = parts[i]
        if p == "*":
            if not isinstance(cur, list):
                return None
            return [walk(x, i + 1) for x in cur]
        if isinstance(cur, dict):
            return walk(cur.get(p), i + 1)
        if isinstance(cur, list) and p.lstrip("-").isdigit():
            idx = int(p)
            return walk(cur[idx], i + 1) if -len(cur) <= idx < len(cur) else None
        return None

    return walk(obj, 0)


# ────────────────────────── 配置 ──────────────────────────
@dataclass
class EndpointCfg:
    """一个能力（embed / rerank / llm）的接入配置。"""

    protocol: str = "local"          # local | openai | generic
    preset: str = ""
    url: str = ""
    model: str = ""
    api_key: str = ""
    headers: dict = field(default_factory=dict)
    timeout_s: float = 30.0
    request: dict = field(default_factory=dict)     # generic 专用：请求体拼装
    response: dict = field(default_factory=dict)    # generic 专用：响应取值路径

    def merged_request(self) -> dict:
        return self.request or {}

    def merged_response(self) -> dict:
        return self.response or {}


def _load_yaml() -> dict:
    try:
        import yaml
        with (ROOT / "config" / "retrieval.yaml").open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.error("读 config/retrieval.yaml 失败（%s）——接入配置回退默认值", e)
        return {}


def _env_json(key: str) -> dict:
    raw = os.getenv(key, "")
    if not raw:
        return {}
    try:
        import json
        return json.loads(raw)
    except ValueError:
        log.error("%s 不是合法 JSON，忽略：%r", key, raw[:80])
        return {}


# env 覆盖表：(env 键, 赋值函数)
_ENV_MAP = {
    "protocol": lambda p: os.getenv(f"{p.upper()}_PROTOCOL", ""),
    "preset": lambda p: os.getenv(f"{p.upper()}_PRESET", ""),
    "url": lambda p: os.getenv(f"{p.upper()}_URL", ""),
    "model": lambda p: os.getenv(f"{p.upper()}_MODEL", ""),
    "api_key": lambda p: os.getenv(f"{p.upper()}_API_KEY", ""),
    "timeout_s": lambda p: os.getenv(f"{p.upper()}_TIMEOUT_S", ""),
}


def load_endpoint(kind: str, default_protocol: str = "local") -> EndpointCfg:
    """解析某个能力的接入配置：yaml -> preset -> env 覆盖。"""
    cfg_yaml = _load_yaml()
    node = dict((cfg_yaml.get("endpoints") or {}).get(kind) or {})
    preset_name = node.get("preset") or os.getenv(f"{kind.upper()}_PRESET", "")
    if preset_name:
        preset = dict((cfg_yaml.get("presets") or {}).get(preset_name) or {})
        if not preset:
            log.error("endpoints.%s.preset=%r 在 presets 里不存在", kind, preset_name)
        else:                                  # 预设打底，显式配置覆盖
            merged = {**preset, **{k: v for k, v in node.items() if v not in (None, "", {}, [])}}
            node = merged

    c = EndpointCfg(protocol=node.get("protocol", default_protocol),
                    preset=preset_name, url=node.get("url", "") or "",
                    model=node.get("model", "") or "",
                    api_key=node.get("api_key", "") or "",
                    headers=dict(node.get("headers") or {}),
                    timeout_s=float(node.get("timeout_s", 30.0)),
                    request=dict(node.get("request") or {}),
                    response=dict(node.get("response") or {}))

    # env 单项覆盖（ops 不改文件也能切）
    for attr, getter in _ENV_MAP.items():
        v = getter(kind)
        if v:
            setattr(c, attr, float(v) if attr == "timeout_s" else v)
    for key, env_key in (("headers", f"{kind.upper()}_HEADERS"),
                         ("request", f"{kind.upper()}_REQUEST"),
                         ("response", f"{kind.upper()}_RESPONSE")):
        extra = _env_json(env_key)
        if extra:
            setattr(c, key, {**getattr(c, key), **extra})

    # 历史 env 别名，避免已有部署改配置后失效
    if kind == "llm" and not c.url:
        c.url = os.getenv("LLM_UPSTREAM", "")
    if kind == "llm" and not c.model:
        c.model = os.getenv("LLM_UPSTREAM_MODEL", "")
    if kind == "llm" and not c.api_key:
        c.api_key = os.getenv("LLM_UPSTREAM_KEY", "")
    return c


def mask_url(url: str) -> str:
    """脱敏：只保留 scheme://host:port，路径与查询串一律不回显（可能带 key）。"""
    if not url:
        return ""
    head = url.split("?")[0]
    if "://" in head:
        scheme, rest = head.split("://", 1)
        host = rest.split("/")[0]
        return f"{scheme}://{host}/…"
    return head.split("/")[0] + "/…"


def describe(cfg: EndpointCfg) -> dict:
    """对外自省用：绝不回显 api_key / 自定义头的值。"""
    # 注意：这里的键不能叫 model —— 自省接口外层已有 model_id，
    # 同名字段会把"模型身份"覆盖成"端点指定名"，正好掩盖身份漂移
    return {"protocol": cfg.protocol, "preset": cfg.preset or None,
            "url": mask_url(cfg.url), "endpoint_model": cfg.model or None,
            "has_api_key": bool(cfg.api_key),
            "header_keys": sorted(cfg.headers.keys()),
            "timeout_s": cfg.timeout_s}


# ────────────────────────── 本地实现 ──────────────────────────
def _hash_embed(texts: list[str]) -> list[dict]:
    from app.store.hash_embed import hash_embedding, hash_sparse
    return [{"dense": hash_embedding(t), "sparse": hash_sparse(t)} for t in texts]


_m3_cache: dict[str, object] = {}


def _m3_embed(texts: list[str], kind: str, model_id: str) -> list[dict]:
    from FlagEmbedding import BGEM3FlagModel
    m = _m3_cache.get(model_id)
    if m is None:                      # CPU 加载 2.3GB 权重很慢，进程级缓存避免每请求重载
        m = BGEM3FlagModel(model_id, use_fp16=False)
        _m3_cache[model_id] = m
    out = (m.encode(texts, return_dense=True, return_sparse=True) if kind != "query"
           else m.encode_queries(texts, return_dense=True, return_sparse=True))
    return [{"dense": out["dense_vecs"][i].tolist(),
             "sparse": {str(k): float(v) for k, v in out["lexical_weights"][i].items()}}
            for i in range(len(texts))]


# ────────────────────────── Provider ──────────────────────────
class EmbedProvider:
    """嵌入 Provider。对外契约稳定：{embeddings:[{dense, sparse}], dim, model}。"""

    def __init__(self, backend: str, cfg: EndpointCfg, model_id: str, expect_dim: int):
        self.backend, self.cfg, self.model_id = backend, cfg, model_id
        self.expect_dim = expect_dim

    # —— 各协议实现 ——
    def _local(self, texts: list[str], kind: str) -> list[dict]:
        if self.backend == "bge_m3":
            return _m3_embed(texts, kind, self.model_id)
        return _hash_embed(texts)

    async def _openai(self, texts: list[str], kind: str) -> list[dict]:
        import httpx
        url = self.cfg.url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as cli:
            r = await cli.post(url, json={"model": self.cfg.model or self.model_id,
                                          "input": texts}, headers=headers)
            r.raise_for_status()
            vecs = dig(r.json(), "data.*.embedding")
        if vecs is None:
            raise ValueError(f"OpenAI 兼容响应里取不到 data.*.embedding：{str(r.text)[:200]}")
        # OpenAI 标准没有稀疏权重，如实返回 null，不伪造
        return [{"dense": [float(x) for x in v], "sparse": None} for v in vecs]

    async def _generic(self, texts: list[str], kind: str) -> list[dict]:
        import httpx
        req = self.cfg.request or {}
        body = {req.get("texts_key", "texts"): texts}
        body.update(req.get("extra") or {})
        if self.cfg.model and req.get("model_key"):
            body[req["model_key"]] = self.cfg.model
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as cli:
            r = await cli.post(self.cfg.url, json=body, headers=self.cfg.headers)
            r.raise_for_status()
            data = r.json()
        dense = dig(data, (self.cfg.response or {}).get("dense", ""))
        if dense is None:
            raise ValueError(f"generic 响应里按 {self.cfg.response.get('dense')!r} 取不到向量："
                             f"{str(data)[:200]}")
        sparse_path = (self.cfg.response or {}).get("sparse")
        sparse = dig(data, sparse_path) if sparse_path else None
        out = []
        for i, v in enumerate(dense):
            out.append({"dense": [float(x) for x in v],
                        "sparse": dict(sparse[i]) if sparse and i < len(sparse) and sparse[i]
                        else None})
        return out

    async def embed(self, texts: list[str], kind: str = "doc") -> dict:
        t0 = time.time()
        if self.cfg.protocol == "openai":
            embs = await self._openai(texts, kind)
        elif self.cfg.protocol == "generic":
            embs = await self._generic(texts, kind)
        else:
            embs = self._local(texts, kind)

        dim = len(embs[0]["dense"]) if embs else 0
        # 维度闸门：与 models.embed.dim 声明不符必须炸，不能静默把库查废
        if self.expect_dim and dim != self.expect_dim:
            raise ValueError(
                f"嵌入维度 {dim} 与 models.embed.dim={self.expect_dim} 不符 —— "
                f"换模型必须重建 Milvus collection 并重跑 seed，否则召回全是噪声")
        return {"embeddings": embs, "dim": dim, "model": self.model_id,
                "backend": self.backend, "protocol": self.cfg.protocol,
                "count": len(embs), "elapsed_ms": int((time.time() - t0) * 1000)}


_rerank_cache: dict[str, object] = {}


class RerankProvider:
    """重排 Provider。契约：{scores, model}，scores 与 texts 同序。"""

    def __init__(self, backend: str, cfg: EndpointCfg, model_id: str):
        self.backend, self.cfg, self.model_id = backend, cfg, model_id

    def _local(self, query: str, texts: list[str]) -> list[float]:
        if self.backend == "bge":
            from FlagEmbedding import FlagReranker
            r = _rerank_cache.get(self.model_id)
            if r is None:                    # CPU 加载重，进程级缓存避免每请求重载
                r = FlagReranker(self.model_id, use_fp16=False)  # CPU 无 CUDA，fp16 会炸
                _rerank_cache[self.model_id] = r
            return [float(x) for x in r.compute_score([[query, t] for t in texts], normalize=True)]
        from app.store.hash_embed import cosine, hash_embedding
        qv = hash_embedding(query)
        return [round(max(0.0, (cosine(qv, hash_embedding(t)) + 1.0) / 2.0) ** 3, 6)
                for t in texts]

    async def _generic(self, query: str, texts: list[str]) -> list[float]:
        import httpx
        req = self.cfg.request or {}
        body = {req.get("query_key", "query"): query,
                req.get("texts_key", "texts"): texts}
        body.update(req.get("extra") or {})
        if self.cfg.model and req.get("model_key"):
            body[req["model_key"]] = self.cfg.model
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as cli:
            r = await cli.post(self.cfg.url, json=body, headers=self.cfg.headers)
            r.raise_for_status()
            data = r.json()
        path = (self.cfg.response or {}).get("scores", "")
        raw = dig(data, path)
        if raw is None:
            raise ValueError(f"generic 响应里按 {path!r} 取不到分数：{str(data)[:200]}")
        # 兼容 [{"index":i,"score":s}] 与 [s] 两种形状
        if raw and isinstance(raw[0], dict):
            pairs = [(int(x.get("index", i)), float(x.get("score", x.get("relevance_score", 0.0))))
                     for i, x in enumerate(raw)]
            pairs.sort(key=lambda p: p[0])
            return [s for _i, s in pairs]
        return [float(x) for x in raw]

    async def rerank(self, query: str, texts: list[str]) -> dict:
        if self.cfg.protocol in ("generic", "openai"):
            scores = await self._generic(query, texts)
        else:
            scores = self._local(query, texts)
        if len(scores) != len(texts):
            raise ValueError(f"重排分数 {len(scores)} 条与输入 {len(texts)} 条不齐，无法对齐候选")
        return {"scores": scores, "model": self.model_id,
                "backend": self.backend, "protocol": self.cfg.protocol}


class LLMProvider:
    """生成 Provider。对外恒为 OpenAI 兼容形状（非流式 JSON / 流式 SSE）。"""

    def __init__(self, backend: str, cfg: EndpointCfg, model_id: str):
        self.backend, self.cfg, self.model_id = backend, cfg, model_id

    async def chat(self, payload: dict):
        """payload 已是 OpenAI 请求体；返回 JSON dict 或 StreamingResponse。"""
        if self.backend == "openai" or self.cfg.protocol == "openai":
            return await self._proxy(payload)
        if self.backend == "hf":
            return await self._hf(payload)
        if self.cfg.protocol == "generic" and self.cfg.url:
            return await self._proxy(payload)     # generic 也走透传，仅换 URL/头
        return None                                # 交给 echo 分支

    def warm(self) -> None:
        """预加载 hf 模型权重（CPU 读盘+初始化 10s+，避免首查超时）。非 hf 后端为 no-op。"""
        if self.backend == "hf":
            _load_hf(self.cfg.model or self.model_id)

    def _url(self) -> str:
        return self.cfg.url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict:
        h = dict(self.cfg.headers)
        if self.cfg.api_key:
            h.setdefault("Authorization", f"Bearer {self.cfg.api_key}")
        return h

    async def _proxy(self, payload: dict):
        """逐行 SSE 透传。

        【这里曾经是坏的】：旧实现用 httpx.post 收流式请求，上游回的是 SSE，
        r.json() 必然抛错 -> 500 -> 上层 catch 后记 generation_unavailable。
        表现为"配了真 LLM 却永远降级"。所以流式必须逐行透传，不能攒齐再解析。
        """
        import httpx
        from fastapi.responses import StreamingResponse
        if self.cfg.model:
            payload["model"] = self.cfg.model

        if not payload.get("stream"):
            async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as cli:
                r = await cli.post(self._url(), json=payload, headers=self._headers())
                r.raise_for_status()
                return r.json()

        async def gen():
            async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.cfg.timeout_s, read=self.cfg.timeout_s)) as cli:
                async with cli.stream("POST", self._url(), json=payload,
                                      headers=self._headers()) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line.strip():
                            yield line + "\n\n"      # 原样透传，含最后的 data: [DONE]

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def _hf(self, payload: dict):
        """进程内 transformers 真生成（CPU 小模型跑通链路用，非生产形态）。

        stream=False -> 返回 OpenAI 兼容 JSON dict（nl2query 结构化抽取用）；
        stream=True  -> StreamingResponse（报告生成用）。
        """
        import asyncio
        import queue
        import threading
        from fastapi.responses import StreamingResponse

        model_id = self.cfg.model or self.model_id
        tok, mdl = await asyncio.get_running_loop().run_in_executor(None, _load_hf, model_id)
        text = tok.apply_chat_template(payload["messages"], tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt")
        max_new = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
        gen_kw = dict(max_new_tokens=max_new, do_sample=False,
                      temperature=None, top_p=None)

        # 非流式：同步 generate，返回标准 chat.completion（L2 结构化 JSON 抽取用）
        if not payload.get("stream"):
            def _run():
                out = mdl.generate(**inputs, **gen_kw)
                in_len = inputs["input_ids"].shape[1]
                return tok.decode(out[0][in_len:], skip_special_tokens=True)

            content = await asyncio.get_running_loop().run_in_executor(None, _run)
            return {
                "id": f"hf-{abs(hash(model_id))}",
                "object": "chat.completion",
                "model": model_id,
                "choices": [{"message": {"role": "assistant", "content": content},
                             "index": 0, "finish_reason": "stop"}],
                "usage": {"total_tokens": len(tok.encode(content))},
            }

        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**inputs, streamer=streamer, **gen_kw)
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                mdl.generate(**kw)
            finally:
                q.put(None)

        def pump():
            for piece in streamer:
                q.put(piece)
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        threading.Thread(target=pump, daemon=True).start()

        async def gen():
            loop = asyncio.get_running_loop()
            while True:
                piece = await loop.run_in_executor(None, q.get)
                if piece is None:
                    break
                yield _sse_chunk(model_id, piece)
            yield _sse_chunk(model_id, "")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")


_hf_cache: dict[str, tuple] = {}


def _load_hf(model_id: str):
    if model_id not in _hf_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        # CPU 低内存推理：优先 bf16（3B 约 6GB，省一半内存）；OOM/不支持则回退 fp32
        try:
            import torch
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except Exception:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
        mdl.eval()
        _hf_cache[model_id] = (tok, mdl)
    return _hf_cache[model_id]


def _sse_chunk(model: str, content: str) -> str:
    import json
    body = {"choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}],
            "model": model, "object": "chat.completion.chunk"}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
