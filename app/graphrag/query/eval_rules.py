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


def hit_context(hit: dict, kb_id=None) -> dict:
    return {
        "content": hit.get("content") or "",
        "rerank_score": hit.get("_score"),
        "kb_id": str(kb_id) if kb_id and not isinstance(kb_id, list) else (hit.get("kb_id") or ""),
        "section": hit.get("title") or hit.get("section") or "",
        "kg_confidence": hit.get("kg_confidence"),
    }


def eval_stored_rules(rules: list[dict], hits: list[dict], kb_scope=None) -> list[dict]:
    """返回每条规则的求值结果(含未触发)。Prompt 只用 triggered 且有 hint 的。"""
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
        evidence = None
        triggered = False
        for h in hits:
            if _cond(hit_context(h, kb_scope), cond):
                triggered = True
                evidence = (h.get("content") or "")[:80]
                break
        action = rule.get("action") or {}
        out.append({
            "id": rule.get("id"),
            "name": rule.get("name"),
            "triggered": triggered,
            "label": action.get("label") or "",
            "severity": action.get("severity") or "info",
            "hint": (action.get("hint") or "") if triggered else "",
            "evidence": evidence if triggered else None,
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
