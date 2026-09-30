"""retrieval 服务入口（FastAPI）。

启动：uvicorn app.main:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.eval import router as eval_router
from .api.graph import router as graph_router
from .api.retrieval import router as retrieval_router
from .api.session import router as session_router
from .config import get_settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="retrieval", version="3.1.0")
app.include_router(retrieval_router)
app.include_router(session_router)
app.include_router(graph_router)
app.include_router(eval_router)


def _warm_ollama(s) -> None:
    """预热 ollama LLM：把 qwen2.5:3b 冷加载成本从首查挪到启动期。

    冷加载 3B Q4 需 10~30s；不预热，首查的 L2(nl2query) 会撞 20s 预算超时、
    报告生成首 token 也慢。启动期发一条最小请求触发 ollama 把模型加载进内存。
    失败/超时不阻断启动（L2/生成届时按降级路径自愈）；连接拒绝会立刻失败，不会拖满 timeout。
    """
    if not s.graphrag_llm_base_url:
        return
    log = logging.getLogger("retrieval.main")
    try:
        from .llm import chat_sync
        chat_sync([{"role": "user", "content": "ping"}], max_tokens=1, timeout=120.0)
        log.info("ollama llm warmed: model=%s", s.graphrag_llm_model)
    except Exception as e:
        log.warning("ollama llm warmup failed (首查 L2/生成将走冷路径): %s", e)


def _keepalive_loop(s, interval_s: int = 180) -> None:
    """保活心跳：本地 CPU 部署空置 ~30min 后，ollama/model_stub/Milvus/Neo4j 的模型/集合/
    缓存会被卸载或被 OS 换出，首查撞全冷路径返空（实测 33min 空置后首查 vector/graph 全 0）。
    每 interval_s 轻量 ping 一遍，把内存按在 RAM。失败只记 debug、绝不阻断（主链有降级路径）。
    daemon 线程，随进程退出自动回收。ollama 主要靠 OLLAMA_KEEP_ALIVE=-1 常驻，这里 ping 兜底。
    """
    log = logging.getLogger("retrieval.main")

    def _tick() -> None:
        # 1) ollama LLM（-1 常驻下是极小 forward pass；非 -1 部署靠它防 5min 卸载）
        if s.graphrag_llm_base_url:
            try:
                from .llm import chat_sync
                chat_sync([{"role": "user", "content": "ping"}], max_tokens=1, timeout=60.0)
            except Exception:
                pass
        # 2) model_stub：BGE-M3 嵌入（OS 换出后会慢到击穿召回预算）
        if s.retrieval_embed_url:
            try:
                import httpx
                with httpx.Client(timeout=30.0) as cli:
                    cli.post(s.retrieval_embed_url, json={"texts": ["keepalive"], "type": "query"})
            except Exception:
                pass
        # 3) Milvus：load_collection 幂等，防空闲卸载
        # 4) Neo4j：RETURN 1 保 Bolt 连接 + Cypher 编译缓存
        if s.retrieval_backend == "real":
            try:
                from pymilvus import MilvusClient
                client = MilvusClient(uri=s.milvus_uri)
                for name in (s.milvus_chunk_collection, s.milvus_entity_collection):
                    if client.has_collection(name):
                        client.load_collection(name)
            except Exception:
                pass
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(s.neo4j_uri,
                                              auth=(s.neo4j_user, s.neo4j_password))
                try:
                    with driver.session() as sess:
                        sess.run("RETURN 1").consume()
                finally:
                    driver.close()
            except Exception:
                pass

    def _run() -> None:
        while True:
            time.sleep(interval_s)
            try:
                _tick()
                log.info("keepalive tick ok")
            except Exception as e:  # 兜底：绝不因保活崩线程
                log.debug("keepalive tick failed: %s", e)

    threading.Thread(target=_run, name="retrieval-keepalive", daemon=True).start()


@app.on_event("startup")
def _startup() -> None:
    s = get_settings()  # 缺必需变量在此炸掉（禁止带默认值静默启动）
    if s.retrieval_backend == "real":
        # 预热重依赖：pymilvus / neo4j 冷 import 需 1~2s，首请求会击穿
        # milvus_hybrid_ms(800) / neo4j_local_ms(600) 预算，导致 vector/graph 路首查超时返空。
        # 提前 import 进 sys.modules，后续函数内懒加载变 no-op。
        log = logging.getLogger("retrieval.main")
        try:
            import pymilvus  # noqa: F401
            import neo4j  # noqa: F401
            log.info("warmed heavy deps for real backend: pymilvus=%s neo4j=%s",
                     getattr(pymilvus, "__version__", "?"),
                     getattr(neo4j, "__version__", "?"))
            # 预热 Milvus 集合 load：空闲/容器重启后集合会被卸载，首查触发 load 需秒级，
            # 会击穿 milvus_hybrid_ms(800) 预算导致向量路首查超时返空。启动时强制 load。
            from pymilvus import MilvusClient
            client = MilvusClient(uri=s.milvus_uri)
            for name in (s.milvus_chunk_collection, s.milvus_entity_collection):
                if client.has_collection(name):
                    client.load_collection(name)
                    log.info("milvus collection loaded: %s", name)
            # 预热 Neo4j 连接：graph 路每次调用新建 driver，首查要付 Bolt 握手 +
            # Cypher 编译冷启动，会击穿 graph.budget_s(0.6s) 导致首查 graph 路超时返空。
            # 启动期跑一条最小查询预热连接 + 编译缓存（与 Milvus load 同理）。
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(s.neo4j_uri,
                                              auth=(s.neo4j_user, s.neo4j_password))
                try:
                    with driver.session() as sess:
                        sess.run("RETURN 1").consume()
                    log.info("neo4j warmed: uri=%s", s.neo4j_uri)
                finally:
                    driver.close()
            except Exception as e:
                log.warning("neo4j warmup failed (首查 graph 路可能超时返空): %s", e)
        except Exception as e:  # 预热失败不阻断启动，首请求仍可由降级路径自愈
            log.warning("warmup heavy deps failed: %s", e)

    # 预热 ollama LLM（与后端通道无关：只要配了 graphrag_llm_base_url 就预热）
    _warm_ollama(s)

    # 保活心跳：防本地 CPU 部署空置后模型/集合被卸载导致首查返空（每 3min 轻量 ping）
    _keepalive_loop(s)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "retrieval", "version": "3.1.0"}


# 前端单页（在线检索 + 测评）。挂载必须放在所有路由之后，Mount("/") 按前缀匹配
# 会拦截其后注册的路由，故置于文件末尾作为兜底；浏览器开 http://127.0.0.1:8100/ 即见页面。
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
          name="static")
