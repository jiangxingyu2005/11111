"""TRACE span 埋点（M-C2 发号 + M-D2 记录）骨架。

契约：stage/status/duration/hit_count/hit_parent_ids/degraded_codes；
异步旁路导出（file/kafka/http），写失败不阻塞主链（EV-2）。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings


def new_trace_id() -> str:
    return uuid.uuid4().hex


# ────────────────────────────────────────────────────────────
# trace_id 绑定（M-R4/R5）：检索请求 -> focus 实体 + scope
# 独立图谱页凭 trace_id 二次取数，避免前端把命中实体全量回传（泄漏 + 被篡改）
# 生产应换 Redis（TTL 30min）；此处为进程内存实现，接口不变（IF-7 Redis 本期推迟）
# ────────────────────────────────────────────────────────────
TRACE_TTL_S = 1800
_binding: dict[str, dict] = {}


def bind_trace(trace_id: str, focus_entities: list[str], scope: list[str],
               child_ids: list[str] | None = None) -> None:
    _purge()
    _binding[trace_id] = {"ts": time.time(), "focus_entities": list(focus_entities),
                          "scope": list(scope), "child_ids": list(child_ids or [])}


def get_binding(trace_id: str) -> dict | None:
    _purge()
    b = _binding.get(trace_id)
    return b if b else None


def _purge() -> None:
    now = time.time()
    for k in [k for k, v in _binding.items() if now - v["ts"] > TRACE_TTL_S]:
        _binding.pop(k, None)


@dataclass
class Span:
    stage: str                      # understand/recall_vector/recall_graph/fuse/rerank/backfill/generate
    status: str = "ok"              # ok / degraded / failed
    t0: float = field(default_factory=time.monotonic)
    duration_ms: float = 0.0
    hit_count: int = 0
    hit_parent_ids: list[str] = field(default_factory=list)
    degraded_codes: list[str] = field(default_factory=list)
    detail: str = ""

    def close(self) -> None:
        self.duration_ms = round((time.monotonic() - self.t0) * 1000, 1)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "hit_count": self.hit_count,
            "hit_parent_ids": self.hit_parent_ids,
            "degraded_codes": self.degraded_codes,
            "detail": self.detail,
        }


class TraceRecorder:
    """单请求 span 收集 + 异步导出。eval_mode=true 时单独打标（EV-5）。"""

    def __init__(self, trace_id: str, eval_mode: bool = False) -> None:
        self.trace_id = trace_id
        self.eval_mode = eval_mode
        self.spans: list[Span] = []

    def start(self, stage: str) -> Span:
        s = Span(stage=stage)
        self.spans.append(s)
        return s

    def to_payload(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "eval_mode": self.eval_mode,
            "ts": time.time(),
            "spans": [s.to_dict() for s in self.spans],
        }

    def export_async(self) -> None:
        """旁路导出：任何异常吞掉只记日志，绝不阻塞主链（写侧纪律）。"""
        settings = get_settings()
        if settings.retrieval_trace_export == "file":
            payload = self.to_payload()
            asyncio.get_event_loop().run_in_executor(
                None, self._write_file, payload, Path(settings.retrieval_trace_dir)
            )
        # kafka/http 导出：后续按 EV-2 接入，此处预留分支

    @staticmethod
    def _write_file(payload: dict, dir_: Path) -> None:
        try:
            dir_.mkdir(parents=True, exist_ok=True)
            with (dir_ / f"{payload['trace_id']}.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # 旁路失败不影响主链
