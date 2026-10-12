"""检索后规则求值:只把命中规则的短 hint 交给 LLM,整张规则表不进 Prompt。"""
from __future__ import annotations

from typing import Any


_OPS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clause(ctx: dict, clause: dict) -> bool:
    field = str(clause.get("field") or "")
    op = str(clause.get("op") or "")
    want = clause.get("value")
    got = ctx.get(field)
    if got is None:
        return False
    if op == "contains":
        return str(want) in str(got)
    fn = _OPS.get(op)
    if fn is None:
        return False
    if op in ("lt", "lte", "gt", "gte"):
        a, b = _num(got), _num(want)
        if a is None or b is None:
            return False
        return fn(a, b)
    return fn(str(got), str(want))


def _cond(ctx: dict, condition: dict) -> bool:
    if not isinstance(condition, dict):
        return False
    alls = condition.get("all") or []
    anys = condition.get("any") or []
    if not alls and not anys:
        return False
    if alls and not all(_clause(ctx, c) if isinstance(c, dict) else False for c in alls):
        return False
    if anys and not any(_clause(ctx, c) if isinstance(c, dict) else False for c in anys):
        return False
    return True


def _match_stats(ctx: dict, condition: dict) -> tuple[bool, float]:
    """返回 (是否触发, 条件命中比例)。

    all: 全命中才算触发, 命中比例 = 命中子句 / 总子句
    any: 命中一条即触发, 命中比例 = 命中子句 / 总子句
    两者都有: all 必须全过, any 算加分。
    """
    if not isinstance(condition, dict):
        return False, 0.0
    alls = [c for c in (condition.get("all") or []) if isinstance(c, dict)]
    anys = [c for c in (condition.get("any") or []) if isinstance(c, dict)]
    if not alls and not anys:
        return False, 0.0
    all_hit = sum(1 for c in alls if _clause(ctx, c))
    any_hit = sum(1 for c in anys if _clause(ctx, c))
    total = len(alls) + len(anys)
    ratio = (all_hit + any_hit) / total if total else 0.0
    triggered = (not alls or all_hit == len(alls)) and (not anys or any_hit > 0)
    return triggered, round(ratio, 4)


def _confidence(triggered: bool, ratio: float, retrieval: float) -> float:
    """置信度 = 检索相关度 × 条件命中比例,由数据算出,不写死。

    触发: 0.5×检索分 + 0.5×条件命中比例
    未触发: 0.3×条件命中比例(部分命中给一点分,全不中给 0)
    """
    retrieval = max(min(retrieval, 1.0), 0.0)
    if triggered:
        return round(0.5 * retrieval + 0.5 * ratio, 4)
    return round(0.3 * ratio, 4)


def hit_context(hit: dict, kb_id=None) -> dict:
    return {
        "content": hit.get("content") or "",
        "rerank_score": hit.get("_score"),
        "kb_id": str(kb_id) if kb_id and not isinstance(kb_id, list) else (hit.get("kb_id") or ""),
        "section": hit.get("title") or hit.get("section") or "",
        "kg_confidence": hit.get("kg_confidence"),
    }


_OP_CN = {"lt": "小于", "lte": "小于等于", "gt": "大于", "gte": "大于等于",
          "eq": "等于", "ne": "不等于", "contains": "包含"}
_FIELD_CN = {"content": "正文", "rerank_score": "检索分数", "kb_id": "知识库",
             "section": "章节", "kg_confidence": "抽取置信度"}


def _render_condition(cond: dict) -> str:
    """把 DSL 条件渲染成前端可直接展示的自然语言。

    优先用 condition.display;没有则从 all/any 子句拼。
    """
    if not isinstance(cond, dict):
        return ""
    if cond.get("display"):
        return str(cond["display"])
    parts = []
    for key, conj in (("all", " 且 "), ("any", " 或 ")):
        clauses = cond.get(key) or []
        rendered = []
        for c in clauses:
            if not isinstance(c, dict):
                continue
            fld = _FIELD_CN.get(str(c.get("field")), str(c.get("field")))
            op = _OP_CN.get(str(c.get("op")), str(c.get("op")))
            val = c.get("value")
            rendered.append(f"{fld} {op} {val}")
        if rendered:
            parts.append(conj.join(rendered))
    return "；".join(parts)


def _render_evidence(triggered: bool, evidence: str | None, name: str) -> str:
    """把证据拼成叙事格式:「观测：… → 条件满足 / → 规则未触发」。"""
    if triggered and evidence:
        return f"观测：{evidence} → 条件满足"
    return f"观测：当前问答上下文未满足「{name}」全部条件 → 规则未触发"


def eval_stored_rules(rules: list[dict], hits: list[dict], kb_scope=None) -> list[dict]:
    """返回每条规则的求值结果(含未触发)。Prompt 只用 triggered 且有 hint 的。

    输出字段对齐前端报告页:
      triggered / confidence / condition_display / evidence_narrative
      action: intent / priority / weight / label / severity / hint
    """
    out: list[dict] = []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rid = str(rule.get("kb_id") or "")
        if kb_scope:
            ids = [str(x) for x in (kb_scope if isinstance(kb_scope, list) else [kb_scope])]
            if rid and rid not in ids:
                continue
        cond = rule.get("condition") or {}
        best_trig = False
        best_ratio = 0.0
        best_retr = 0.0
        evidence = None
        for h in hits:
            ctx = hit_context(h, kb_scope)
            trig, ratio = _match_stats(ctx, cond)
            retr = float(h.get("_score") or 0.0)
            if trig and (not best_trig or ratio > best_ratio or
                         (ratio == best_ratio and retr > best_retr)):
                best_trig, best_ratio, best_retr = True, ratio, retr
                evidence = (h.get("content") or "")[:120]
            elif not best_trig and ratio > best_ratio:
                best_ratio, best_retr = ratio, retr
        action = rule.get("action") or {}
        name = rule.get("name") or rule.get("id") or ""
        conf = _confidence(best_trig, best_ratio, best_retr)
        out.append({
            "id": rule.get("id"),
            "name": name,
            "triggered": best_trig,
            "confidence": conf,
            "confidence_detail": {"retrieval": round(best_retr, 4),
                                  "condition_ratio": best_ratio,
                                  "formula": "0.5×检索+0.5×条件命中" if best_trig else "0.3×条件命中"},
            "condition_display": _render_condition(cond),
            "evidence_narrative": _render_evidence(best_trig, evidence, name),
            "label": action.get("label") or "",
            "severity": action.get("severity") or "info",
            "intent": action.get("intent") or "",
            "priority": action.get("priority") or "",
            "weight": _num(action.get("weight")),
            "hint": (action.get("hint") or "") if best_trig else "",
            "evidence": evidence if best_trig else None,
        })
    return out


def rule_hints(rule_hits: list[dict], limit: int = 5) -> list[str]:
    lines = []
    for r in rule_hits:
        if not r.get("triggered"):
            continue
        hint = r.get("hint") or r.get("label") or r.get("name")
        if not hint:
            continue
        lines.append(f"[{r.get('id')}] {hint}")
        if len(lines) >= limit:
            break
    return lines


def confidence_block(hits: list[dict], rule_hits: list[dict],
                     signal_agree: float | None = None,
                     signal_fields: list | None = None) -> dict:
    scores = [float(h.get("_score") or 0) for h in hits[:5]]
    retrieval = round(sum(scores) / len(scores), 4) if scores else None
    hit_n = sum(1 for r in rule_hits if r.get("triggered"))
    total = len(rule_hits)
    rule = round(hit_n / total, 4) if total else None
    # 基础权重 规则 0.4 / 检索 0.6;有 IQ 对比时加一致度 0.2。缺哪项就去掉后重归一。
    base = {"rule": 0.4, "retrieval": 0.6, "signal_agree": 0.2}
    vals = {"rule": rule, "retrieval": retrieval, "signal_agree": signal_agree}
    present = {k: base[k] for k, v in vals.items() if v is not None}
    wsum = sum(present.values())
    weights = {k: round(present.get(k, 0) / wsum, 4) if wsum else 0 for k in base}
    overall = round(sum(vals[k] * weights[k] for k in present), 4) if wsum else None
    return {
        "rule": {"score": rule, "hit": hit_n, "total": total} if total else None,
        "retrieval": {"score": retrieval, "k": min(5, len(hits))} if retrieval is not None else None,
        "signal_agree": ({"score": signal_agree, "fields": list(signal_fields or [])}
                         if signal_agree is not None else None),
        "overall": overall,
        "weights": weights,
    }
