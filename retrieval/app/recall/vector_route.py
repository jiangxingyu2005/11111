"""① 路向量召回（M-Q1）——BGE-M3 全量定案版（2026-09-28 V3-D4）。

模型：BGE-M3（dense 1024 维 + sparse lexical_weights 同源产出），与离线侧同模型同空间。
内层子路（融合为一个 `vector_hybrid` 源进外层 RRF）：
  dense A  恒在（主向量 embedding）
  dense B  可选（副向量 embedding_v2，数值/单位归一化信号参数嵌入）
  sparse   可选（text_sparse，IF-1 扩列后启用）
双通道：
  real = 真 Milvus（dense-only 或库内 hybrid_search，取决于开关与列是否就绪）
  fake = 内存三库（data/seed/*.json + 确定性嵌入），零外部依赖跑通全链路
两条通道共用同一 recall() 签名与同一套配置，禁止出现第二套业务流程。

红线：
  - scope 为必填参数（R4），无默认值，过滤表达式必须带 scope
  - 任何异常 -> 空列表 + vector_unavailable（整路剥离，绝不抛给主链）
  - dense B / sparse 列缺失时不得假装内层 RRF：dense-only 是明确行为，不算降级
"""
from __future__ import annotations

import asyncio
import logging

from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from ..fuse.rrf import Hit

log = logging.getLogger("retrieval.recall.vector")


# ────────────────────────────────────────────────────────────
# 嵌入通道（BGE-M3 双模态 + dense B 数值副向量）
# ────────────────────────────────────────────────────────────

async def _embed_query_full(query: str, conditions: list | None = None) -> dict:
    """query 向量化。

    返回 {"dense": [float], "dense_v2": [float]|None, "sparse": {token_id: weight}|None}
    dense_v2 由【已归一化的结构化条件】确定性计算（不依赖模型，real/fake 同算法）；
    无五要素条件 -> None（B 路缺席）。
    fake 通道：dense/sparse 直接本地哈希嵌入，保证与 seed 建库同一空间。
    real/remote：端点只出 dense 时 sparse=None（自动走 dense-only）。
    real/local：BGEM3FlagModel 懒加载，dense+sparse 同源产出。
    """
    s = get_settings()
    dense_v2 = _conditions_to_numeric(conditions)
    if s.retrieval_backend == "fake":
        from ..store.hash_embed import hash_embedding, hash_sparse
        return {"dense": hash_embedding(query), "dense_v2": dense_v2,
                "sparse": hash_sparse(query)}

    if s.retrieval_embed_mode == "local":
        model = _load_m3_local(s.graphrag_embedding_model)
        out = model.encode_queries([query], return_dense=True, return_sparse=True)
        return {"dense": out["dense_vecs"][0].tolist(),
                "dense_v2": dense_v2,
                "sparse": dict(out["lexical_weights"][0])}
    import httpx
    async with httpx.AsyncClient(timeout=5.0) as cli:
        r = await cli.post(s.retrieval_embed_url, json={"texts": [query], "type": "query"})
        r.raise_for_status()
        body = r.json()
        emb = body["embeddings"][0]
        # 端点契约：允许只出 dense（list），或出 {"dense":..., "sparse":...}
        if isinstance(emb, dict):
            return {"dense": emb["dense"], "dense_v2": dense_v2,
                    "sparse": emb.get("sparse")}
        return {"dense": emb, "dense_v2": dense_v2, "sparse": None}


def _conditions_to_numeric(conditions: list | None) -> list[float] | None:
    """把已归一化的结构化条件转成 B 路数值嵌入参数 {rf,pw,prf,doa,snr}。

    点值取原值、区间取中点；无五要素条件 -> None（B 路缺席）。
    字段名与 query_parser 的 UNIT_BASE 对齐（rf/pw/prf/doa/snr），基准单位一致。
    """
    from ..store.hash_embed import hash_numeric_embedding
    params: dict[str, float] = {}
    for c in conditions or ():
        f = getattr(c, "field", None)
        if f not in ("rf", "pw", "prf", "doa", "snr"):
            continue
        v = getattr(c, "value", None)
        if isinstance(v, (list, tuple)) and len(v) == 2 \
                and all(isinstance(x, (int, float)) for x in v):
            v = (float(v[0]) + float(v[1])) / 2.0
        elif isinstance(v, (int, float)):
            v = float(v)
        else:
            continue
        params[f] = v
    if not params:
        return None
    return hash_numeric_embedding(params)


_m3_local = None


def _load_m3_local(model_name: str):
    """local 模式懒加载（进程级缓存；仅 RETRIEVAL_EMBED_MODE=local 时需要 FlagEmbedding）。"""
    global _m3_local
    if _m3_local is None:
        from FlagEmbedding import BGEM3FlagModel  # noqa: 懒加载
        _m3_local = BGEM3FlagModel(model_name, use_fp16=True)
    return _m3_local


def _is_zero(vec: list[float] | None) -> bool:
    return not vec or all(v == 0.0 for v in vec)


def _build_routes(emb: dict, cfg: dict) -> dict[str, dict]:
    """按配置决定激活哪些内层子路。dense A 恒在；dense B / sparse 按开关 + 数据就绪度。"""
    dense = cfg.get("milvus", {}).get("dense", {})
    primary = dense.get("primary", {})
    routes = {"dense": {
        "vector": emb["dense"],
        "tau": float(primary.get("threshold", 0.72)),
        "anns_field": primary.get("vector_field", "embedding"),
        "metric": primary.get("metric_type", "COSINE"),
        "top_k": int(primary.get("top_k", 100)),
    }}
    sec = dense.get("secondary", {})
    if sec.get("enabled", False) and not _is_zero(emb.get("dense_v2")):
        routes["dense_v2"] = {
            "vector": emb["dense_v2"],
            "tau": float(sec.get("threshold", 0.68)),
            "anns_field": sec.get("vector_field", "embedding_v2"),
            "metric": sec.get("metric_type", "COSINE"),
            "top_k": int(sec.get("top_k", 100)),
        }
    sp = cfg.get("milvus", {}).get("sparse", {})
    if sp.get("enabled", False) and emb.get("sparse") is not None:
        routes["sparse"] = {
            "vector": emb["sparse"],
            "tau": 0.0,
            "anns_field": sp.get("field", "text_sparse"),
            "metric": sp.get("metric_type", "BM25"),
            "top_k": int(sp.get("top_k", 100)),
        }
    return routes


# ────────────────────────────────────────────────────────────
# 召回主入口
# ────────────────────────────────────────────────────────────

async def recall(query: str, *, scope: list[str], collector: DegradedCollector,
                 conditions: list | None = None) -> list[Hit]:
    """① 路召回。scope 必填（无默认值，调用方不传直接 TypeError）。

    conditions：QRY-02 归一化条件，用于生成 B 路数值副向量；为 None 时 B 路缺席。
    内层融合：单子路 -> dense-only；多子路 -> 库内/内存 N 路 RRF。
    """
    cfg = effective_cfg()
    timeout_ms = cfg.get("timeout", {}).get("milvus_hybrid_ms", 800)
    try:
        emb = await _embed_query_full(query, conditions)
        routes = _build_routes(emb, cfg)
        search_fn = _milvus_hybrid_search if len(routes) > 1 else _milvus_search
        return await asyncio.wait_for(
            asyncio.to_thread(search_fn, routes, scope, cfg),
            timeout=timeout_ms / 1000.0,
        )
    except Exception as e:  # 整路剥离：超时/连接/嵌入失败统一处理
        log.warning("vector route failed: %s", e)
        collector.record(DegradedCode.VECTOR_UNAVAILABLE, str(e)[:200], "recall_vector")
        return []


def _fake_search(routes: dict[str, dict], scope: list[str], cfg: dict, hybrid: bool) -> list[Hit]:
    from ..store.fake import get_fake_store

    top_k = int(cfg["milvus"]["dense"]["primary"].get("top_k", 100))
    rows = get_fake_store().search_chunks(routes, top_k=top_k, scope=scope)
    # search_chunks 已做单路 τ 过滤 / 多路内层 RRF，此处不再叠加 τ
    return _to_hits(rows, tau=0.0)


def _milvus_search(routes: dict[str, dict], scope: list[str], cfg: dict) -> list[Hit]:
    """dense-only 检索（单内层子路）。"""
    if cfg.get("_backend") == "fake":
        return _fake_search(routes, scope, cfg, hybrid=False)
    from pymilvus import MilvusClient  # 懒加载

    s = get_settings()
    dense_cfg = cfg["milvus"]["dense"]["primary"]
    client = MilvusClient(uri=s.milvus_uri)
    scope_expr = "kb_id in [" + ",".join(f'"{x}"' for x in scope) + "]"
    res = client.search(
        collection_name=s.milvus_chunk_collection,
        data=[routes["dense"]["vector"]],
        anns_field=dense_cfg.get("vector_field", "embedding"),
        search_params={"metric_type": dense_cfg.get("metric_type", "COSINE")},
        limit=int(dense_cfg.get("top_k", 100)),
        filter=scope_expr,
        output_fields=["parent_id", "content", "kg_entity_ids", "kb_id", "span"],
    )
    return _to_hits(res[0], tau=float(dense_cfg.get("threshold", 0.72)))


def _milvus_hybrid_search(routes: dict[str, dict], scope: list[str], cfg: dict) -> list[Hit]:
    """多内层子路（dense/dense_v2/sparse 任 ≥2）库内 RRFRanker 混合检索。

    real：pymilvus hybrid_search(AnnSearchRequest xN, RRFRanker(k=60))
    fake：FakeStore 内部按 N 个排名序列做等价 RRF（语义对齐，非近似替代）
    列未就绪而开关被误开：回退 dense-only + 记日志（禁静默假装 hybrid）。
    """
    if cfg.get("_backend") == "fake":
        return _fake_search(routes, scope, cfg, hybrid=True)
    from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker  # 懒加载

    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    fields = client.describe_collection(
        collection_name=s.milvus_chunk_collection).get("fields", [])
    fnames = {f["name"] for f in fields}
    scope_expr = "kb_id in [" + ",".join(f'"{x}"' for x in scope) + "]"

    reqs = []
    for r in routes.values():
        if r["anns_field"] not in fnames:
            log.warning("hybrid enabled but %s missing; skip sub-route", r["anns_field"])
            continue
        reqs.append(AnnSearchRequest(
            data=[r["vector"]],
            anns_field=r["anns_field"],
            param={"metric_type": r["metric"]},
            limit=r["top_k"],
            expr=scope_expr,
        ))
    if len(reqs) < 2:
        log.warning("hybrid requested but <2 sub-routes ready; fallback dense-only")
        return _milvus_search({"dense": routes["dense"]}, scope, cfg)

    res = client.hybrid_search(
        collection_name=s.milvus_chunk_collection,
        reqs=reqs,
        ranker=RRFRanker(k=int(cfg.get("hybrid", {}).get("rrfranker", {}).get("k", 60))),
        limit=int(cfg.get("hybrid", {}).get("top_k_after_hybrid", 50)),
        output_fields=["parent_id", "content", "kg_entity_ids", "kb_id", "span"],
    )
    return _to_hits(res[0], tau=0.0)   # hybrid 排序已是 RRF 序，不再叠加 τ 过滤


def _to_hits(rows, tau: float) -> list[Hit]:
    hits: list[Hit] = []
    for row in rows:
        if row["distance"] < tau:
            continue
        ent = row.get("entity", {})
        hits.append(Hit(
            child_id=str(row["id"]),
            parent_id=str(ent.get("parent_id", "")),
            rank_score=float(row["distance"]),
            source="vector_hybrid",
            content=ent.get("content", "") or "",
            kg_entity_ids=ent.get("kg_entity_ids") or [],
            meta={"kb_id": ent.get("kb_id", ""),
                  "span": ent.get("span"),           # IF-2：离线给了就有，没给就是 None
                  "section": ent.get("section", "")},
        ))
    return hits
