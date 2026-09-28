"""QRY-02 三层查询理解（M-Q2）骨架。

L1 正则/词典（配置驱动 config/query_parser.yaml，本文件已实现）
L2 LLM（vLLM function calling，骨架：未配置端点即降级 nl2query_llm_timeout）
L3 校验器（白名单/量纲区间/区间自洽，本文件已实现核心检查）

输出契约：intent ∈ {ENTITY_LEVEL, GLOBAL, HYBRID, PURE_STRUCTURED} + conditions[]（五要素）
过渡语义（V2.0.1 §A.5）：④ 路后移期间 conditions 一律软语义，绝不做硬排除。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import query_parser_cfg
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector

# L2 output_schema 的 intent 枚举（yaml 里 STRUCTURED_ONLY 与 V2.0 的 PURE_STRUCTURED 同义，统一用后者）
INTENTS = {"ENTITY_LEVEL", "GLOBAL", "HYBRID", "PURE_STRUCTURED"}

# 字段 -> 归一基准单位（契约 conditions[].unit；与 query_parser.yaml unit_convert.base 一致）
UNIT_BASE = {"rf": "GHz", "pw": "us", "prf": "Hz", "doa": "deg", "snr": "dB"}

# L3 量纲合理区间（query_parser.yaml validator V-03，归一单位后）
RANGE_LIMITS = {
    "rf": (0.01, 300.0),        # GHz
    "pw": (0.01, 10000.0),      # us
    "prf": (1.0, 2_000_000.0),  # Hz
    "doa": (0.0, 360.0),        # deg
    "snr": (-40.0, 80.0),       # dB
}


# 内部 op -> 契约 op（api/retrieval_response.v1.schema.json 的枚举）
# point 用 eq（容差由 structured_filter.yaml tolerance 决定，不进本契约）
OP_TO_CONTRACT = {"point": "eq", "range": "between", "eq": "eq", "month": "between", "day": "between"}


@dataclass
class Condition:
    field: str
    op: str                    # range / point / eq（内部口径）
    value: object
    confidence: float
    raw: str = ""
    strength: str = "soft"     # 过渡期恒 soft（④ 路预留），不进契约（契约无此字段）

    def to_dict(self) -> dict:
        """契约形状：field/op/value/unit/raw/confidence（additionalProperties=false，禁加字段）。"""
        return {
            "field": self.field,
            "op": OP_TO_CONTRACT.get(self.op, "eq"),
            "value": self.value,
            "unit": UNIT_BASE.get(self.field),
            "raw": self.raw or None,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class ParsedQuery:
    intent: str = "ENTITY_LEVEL"
    conditions: list[Condition] = field(default_factory=list)
    rewritten: str = ""        # QRY-00 会话改写后的 query（本期单轮直通）
    l1_hit: bool = False


_UNIT_TO_BASE = {  # 从 query_parser.yaml unit_convert 摘出的最小实现（字段: {单位串: 倍率}）
    "rf": {"GHz": 1.0, "G": 1.0, "吉赫": 1.0, "MHz": 0.001, "M": 0.001, "兆赫": 0.001},
    "pw": {"us": 1.0, "μs": 1.0, "微秒": 1.0, "ms": 1000.0, "毫秒": 1000.0, "ns": 0.001, "纳秒": 0.001},
    "prf": {"Hz": 1.0, "赫兹": 1.0, "kHz": 1000.0, "K": 1000.0, "MHz": 1000000.0, "pps": 1.0},
    "doa": {"度": 1.0, "deg": 1.0, "°": 1.0},
}


def _l1_extract(query: str, collector: DegradedCollector) -> list[Condition]:
    """L1 正则快速通道：消费 config/query_parser.yaml 的 l1_rules。"""
    cfg = query_parser_cfg().get("l1_rules", {})
    if not cfg.get("enabled", False):
        return []
    out: list[tuple[tuple[int, int], Condition]] = []
    for rule in cfg.get("patterns", []):
        m = re.search(rule["regex"], query)
        if not m:
            continue
        o = rule["out"]
        fld, kind = o["field"], o["kind"]
        unit_tbl = _UNIT_TO_BASE.get(fld, {})
        try:
            if kind == "range":
                lo, hi = float(m.group(1)), float(m.group(2))
                unit = m.group(3) if m.lastindex and m.lastindex >= 3 else ""
                factor = unit_tbl.get(unit, 1.0)
                value = [lo * factor, hi * factor]
                op = "range"
            elif kind == "point":
                v = float(m.group(1))
                unit = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                if not unit:
                    collector.record(DegradedCode.UNIT_INFERRED, f"field={fld} bare number", "understand")
                value = v * unit_tbl.get(unit, 1.0)
                op = "point"
            else:
                continue  # toa 等时间类本期骨架暂不展开，交给 L2
            out.append((m.span(), Condition(field=fld, op=op, value=value,
                                            confidence=rule.get("confidence", 0.85),
                                            raw=m.group(0)[:32])))
        except (ValueError, IndexError):
            continue
    # 重叠吸收：span 被另一条完全包含的条件丢弃（如 range "8~12GHz" 吸收 point "12GHz"）
    kept = [
        (span, c) for span, c in out
        if not any(
            os_ != span and os_[0] <= span[0] and span[1] <= os_[1]
            for os_, _ in out
        )
    ]
    kept.sort(key=lambda x: x[0][0])
    return [c for _, c in kept]


def _l3_validate(conds: list[Condition], collector: DegradedCollector) -> list[Condition]:
    """L3 校验器：字段白名单 + 量纲区间 + 区间自洽。不过则丢弃（过渡期一律 soft，无 hard）。"""
    cfg = query_parser_cfg()
    allowed = set(
        (cfg.get("l2_llm", {}).get("output_schema", {})
         .get("properties", {}).get("conditions", {})
         .get("items", {}).get("properties", {}).get("field", {})
         .get("enum", []))
    ) or set(RANGE_LIMITS)
    ok: list[Condition] = []
    for c in conds:
        if c.field not in allowed:
            collector.record(DegradedCode.CONDITION_INVALID, f"V-01 field={c.field}", "understand")
            continue
        lo, hi = RANGE_LIMITS.get(c.field, (None, None))
        vals = c.value if isinstance(c.value, list) else [c.value]
        if lo is not None and any(not (isinstance(v, (int, float)) and lo <= v < hi) for v in vals):
            collector.record(DegradedCode.CONDITION_INVALID, f"V-03 {c.field}={c.value}", "understand")
            continue
        if isinstance(c.value, list) and len(c.value) == 2 and c.value[0] > c.value[1]:
            collector.record(DegradedCode.CONDITION_INVALID, f"V-04 {c.field}={c.value}", "understand")
            continue
        ok.append(c)
    return ok


def parse(query: str, collector: DegradedCollector) -> ParsedQuery:
    """三层解析主入口（骨架版：L1+L3 真实，L2 骨架）。"""
    pq = ParsedQuery(rewritten=query)

    # L1
    conds = _l1_extract(query, collector)
    pq.l1_hit = bool(conds)

    # L2（骨架）：L1 未命中且有 LLM 端点时才调；本期未部署 -> 记降级继续
    if not pq.l1_hit:
        # TODO(M-Q2): 接 vLLM function calling（复用 qingxi llm.py 的调用模式，非检索逻辑）
        collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT, "L2 not deployed (skeleton)", "understand")

    # L3
    pq.conditions = _l3_validate(conds, collector)

    # 意图初判（骨架启发式；正式版由 L2 输出）
    if pq.conditions and len(query) <= 40:
        pq.intent = "PURE_STRUCTURED"
    else:
        pq.intent = "ENTITY_LEVEL"
    return pq
