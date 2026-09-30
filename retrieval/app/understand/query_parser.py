"""QRY-02 三层查询理解（M-Q2）骨架。

L1 正则/词典（配置驱动 config/query_parser.yaml，本文件已实现）
L2 LLM（vLLM function calling，骨架：未配置端点即降级 nl2query_llm_timeout）
L3 校验器（白名单/量纲区间/区间自洽，本文件已实现核心检查）

输出契约：intent ∈ {ENTITY_LEVEL, GLOBAL, HYBRID, PURE_STRUCTURED} + conditions[]（五要素）
过渡语义（V2.0.1 §A.5）：④ 路后移期间 conditions 一律软语义，绝不做硬排除。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import get_settings, query_parser_cfg
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector

log = logging.getLogger("retrieval.understand.query_parser")

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
    l2_needed: bool = False    # L1 未命中 -> 需要异步 L2 抽取（由调用方决定是否触发）


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


def _heuristic_intent(query: str, conds: list[Condition]) -> str:
    """意图启发式兜底（L2 未接/失败时用）：有条件 + 短问 -> 纯结构化，否则实体级。"""
    if conds and len(query) <= 40:
        return "PURE_STRUCTURED"
    return "ENTITY_LEVEL"


def parse(query: str, collector: DegradedCollector) -> ParsedQuery:
    """三层解析主入口（L1+L3 同步快通道；L2 由调用方异步触发，见 parse_l2）。"""
    pq = ParsedQuery(rewritten=query)

    # L1
    conds = _l1_extract(query, collector)
    pq.l1_hit = bool(conds)
    pq.l2_needed = not pq.l1_hit

    # L3
    pq.conditions = _l3_validate(conds, collector)
    pq.intent = _heuristic_intent(query, pq.conditions)
    return pq


# L2 的 LLM 请求超时（CPU 3B 模型生成 JSON 慢，给足；真正 deadline 由调用方 wait_for 兜底）
_L2_TIMEOUT_S = 25.0


def _build_l2_messages(query: str) -> list[dict]:
    """按 config/query_parser.yaml 的 l2_llm.output_schema + hard_rules 拼 prompt。

    3B 小模型对"只描述 schema"的遵循很差（会自作主张输出 {rf:[...]} 这种平铺格式），
    必须给一个【具体示例】让它照着抄格式。示例的字段/取值都在 schema 枚举内。
    """
    cfg = query_parser_cfg()
    l2 = cfg.get("l2_llm") or {}
    props = (l2.get("output_schema") or {}).get("properties") or {}
    cond_props = (props.get("conditions") or {}).get("items") or {} or {}
    cond_props = cond_props.get("properties") or {}
    field_enum = (cond_props.get("field") or {}).get("enum", [])
    op_enum = (cond_props.get("op") or {}).get("enum", [])
    bands = cfg.get("band_table") or []
    band_lines = "\n".join(
        f"  {', '.join(b.get('alias', []))} → [{b['rf_ghz'][0]}, {b['rf_ghz'][1]}]"
        for b in bands)
    hard = "\n".join(f"{i}. {r}" for i, r in enumerate(l2.get("hard_rules") or [], 1))
    system = (
        "你是电磁信号侦察领域的查询理解助手，把用户问题解析成结构化检索条件，"
        "只输出一个合法 JSON 对象，不要任何解释文字或 markdown 代码块。\n"
        f"字段 field 取值：{', '.join(field_enum)}（rf载频GHz / pw脉宽us / prf重频Hz / "
        f"doa方位deg / toa时间 / snr信噪比dB / modulation调制 / platform_type平台 / country国家）\n"
        f"算子 op 取值：{', '.join(op_enum)}（range→[min,max] 区间；point→单值；eq→枚举字面值）\n"
        f"波段展开表（命中波段必须输出 range 区间，禁止中心频点）：\n{band_lines}\n"
        "硬性规则：\n" + hard + "\n"
        "输出格式示例（严格照抄这个结构）：\n"
        '用户问题「X波段的舰载雷达」→ '
        '{"intent":"ENTITY_LEVEL","conditions":[{"field":"rf","op":"range","value":[8.0,12.0],'
        '"unit":"GHz","raw":"X波段","confidence":0.9}]}\n'
        "若用户没提任何参数，conditions 为空数组 []。"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"用户问题：{query}"}]


def _extract_json(text: str) -> dict | None:
    """从模型输出里鲁棒提取 JSON（容忍首尾多余文字 / ```json 围栏）。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.lstrip("`")
        nl = t.find("\n")
        if nl != -1 and t[:nl].strip().lower() in ("json",):
            t = t[nl + 1:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except ValueError:
        return None


def _conditions_from_json(items: list, collector: DegradedCollector) -> list[Condition]:
    """L2 输出的 conditions[] -> Condition 列表，并经 L3 校验器兜底（抽风输出直接丢弃）。"""
    conds: list[Condition] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        fld, op, val = it.get("field"), it.get("op"), it.get("value")
        if fld is None or op is None or val is None:
            continue
        try:
            conf = float(it.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        conds.append(Condition(field=str(fld), op=str(op), value=val,
                               confidence=conf, raw=str(it.get("raw", ""))[:32],
                               strength="soft"))
    return _l3_validate(conds, collector)


async def parse_l2(query: str, collector: DegradedCollector) -> ParsedQuery | None:
    """L2 真实现：调本地 LLM 抽取结构化条件。任何失败/超时 -> 记降级并返回 None（只用 L1）。"""
    s = get_settings()
    if not s.graphrag_llm_base_url:
        collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT,
                         "L2 endpoint not configured", "understand")
        return None
    try:
        from ..llm import chat
        messages = _build_l2_messages(query)
        content = await chat(messages, temperature=0.0, timeout=_L2_TIMEOUT_S)
        data = _extract_json(content)
        if not data:
            collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT,
                             f"L2 non-JSON output: {content[:80]!r}", "understand")
            return None
        conds = _conditions_from_json(data.get("conditions") or [], collector)
        pq = ParsedQuery(rewritten=query, l1_hit=False, l2_needed=False)
        pq.conditions = conds
        intent = str(data.get("intent", "")).upper()
        pq.intent = ("PURE_STRUCTURED" if intent == "STRUCTURED_ONLY"
                     else (intent if intent in INTENTS else _heuristic_intent(query, conds)))
        return pq
    except Exception as e:
        log.warning("L2 nl2query failed: %s", e)
        collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT, str(e)[:200], "understand")
        return None
