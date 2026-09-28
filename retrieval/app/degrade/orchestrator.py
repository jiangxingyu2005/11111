"""QRY-07 降级编排器（M-D1）骨架。

职责：
  1. 预算看门狗：按 config/retrieval.yaml degradation.cutoffs 把累计耗时映射到 L0~L4
  2. 留痕收集器：全链 degraded[] 统一收口（响应体 / TRACE / 前端角标三处同源）

红线：禁止静默降级——任何降级动作必须过 DegradedCollector.record()。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import effective_cfg
from .codes import BudgetLevel, DegradedCode, step_of


@dataclass
class DegradedCollector:
    """一次请求的降级留痕收集器。trace span 与响应体 degraded[] 都从这里取。

    输出形状严格对齐 api/retrieval_response.v1.schema.json：
        {"step": <枚举>, "code": str, "message": str|None, "at": ISO8601}
    内部保留 raw（stage/detail）供排障与 TRACE，但对外只吐契约形状。
    """

    items: list[dict] = field(default_factory=list)

    def record(self, code: DegradedCode, detail: str = "", stage: str = "") -> None:
        code_v = code.value if isinstance(code, DegradedCode) else str(code)
        self.items.append({
            "code": code_v,
            "stage": stage or step_of(code_v),
            "detail": detail,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def has(self, code: DegradedCode) -> bool:
        return any(i["code"] == code.value for i in self.items)

    def raw(self) -> list[dict]:
        return list(self.items)

    def as_list(self) -> list[dict]:
        """契约形状（响应体 / 前端角标 / TRACE 三处同源）。

        step 必须由 code 经 STEP_MAP 推导，不接受调用方随手传的字符串——
        否则会出现 'backfill' 之类不在枚举里的值，契约校验直接失败。
        """
        return [{"step": step_of(i["code"]), "code": i["code"],
                 "message": i["detail"] or None, "at": i["at"]} for i in self.items]

    def codes(self) -> list[str]:
        return [i["code"] for i in self.items]


class BudgetWatchdog:
    """预算看门狗：start() 后随时 current_level()，驱动砍路决策。"""

    def __init__(self, budget_ms: int | None = None) -> None:
        cfg = effective_cfg().get("degradation", {})
        self._budget_s = (budget_ms or int(cfg.get("budget_s", 5.2) * 1000)) / 1000.0
        # cutoffs 按 threshold_s 升序
        self._cutoffs = sorted(
            cfg.get("cutoffs", []), key=lambda c: c.get("threshold_s", 0)
        )
        self._t0 = time.monotonic()

    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    def current_level(self) -> BudgetLevel:
        """返回已超阈值中最严格的一档（elapsed 越大砍得越狠）。"""
        e = self.elapsed_s()
        level = BudgetLevel.L4
        for c in self._cutoffs:  # 升序遍历，最后一个被超过的 threshold 生效
            if e > c.get("threshold_s", 0):
                level = BudgetLevel(c.get("level", "L4"))
        return level

    def over_budget(self) -> bool:
        return self.elapsed_s() > self._budget_s
