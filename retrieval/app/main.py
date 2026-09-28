"""retrieval 服务入口（FastAPI）。

启动：uvicorn app.main:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .api.graph import router as graph_router
from .api.retrieval import router as retrieval_router
from .config import get_settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="retrieval", version="3.1.0")
app.include_router(retrieval_router)
app.include_router(graph_router)


@app.on_event("startup")
def _startup() -> None:
    get_settings()  # 缺必需变量在此炸掉（禁止带默认值静默启动）


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "retrieval", "version": "3.1.0"}
