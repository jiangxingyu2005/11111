"""GP-11 ROUTE-01:离线/实时链路分流(ems route/router.py 移植,配置源改 .env)。

规则(可配置,走 .env 的 graphrag_route_file_size_mb / graphrag_route_batch_count):
  R1 默认小批量小文件 → realtime;
  R2 单文件 > file_size_mb MB → offline;
  R3 批次文件数 > batch_count → offline。
判定优先级:R2/R3 命中其一 → offline;否则 → realtime(R1)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import get


@dataclass
class RouteResult:
    lane: str          # realtime | offline
    rule: str          # R1_DEFAULT_REALTIME | R2_FILE_SIZE | R3_BATCH_COUNT
    tip: str


_TIPS = {
    "R1_DEFAULT_REALTIME": "文件已进入实时索引通道，稍候即可检索。",
    "R2_FILE_SIZE": "该文件体量较大，已加入后台全量索引构建任务，预计处理耗时较长，完成后将推送通知。",
    "R3_BATCH_COUNT": "检测到批量上传，已将其加入后台离线构建队列。",
}


def _size(f) -> int:
    if isinstance(f, dict):
        return int(f.get("size", 0) or 0)
    return int(getattr(f, "size", 0) or 0)


def route_pipeline(files: Sequence, cfg: Optional[dict] = None) -> RouteResult:
    """按 R2/R3 → R1 优先级判定链路。files 元素需提供 size(dict 或带 .size 对象)。"""
    cfg = cfg or get("route") or {}
    size_limit = int(cfg.get("file_size_mb", 10)) * 1024 * 1024
    count_limit = int(cfg.get("batch_count", 5))

    if any(_size(f) > size_limit for f in files):
        return RouteResult("offline", "R2_FILE_SIZE", _TIPS["R2_FILE_SIZE"])
    if len(files) > count_limit:
        return RouteResult("offline", "R3_BATCH_COUNT", _TIPS["R3_BATCH_COUNT"])
    return RouteResult("realtime", "R1_DEFAULT_REALTIME", _TIPS["R1_DEFAULT_REALTIME"])
