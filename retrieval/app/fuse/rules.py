"""预设推理规则引擎（需求(7)「触发预设推理规则」）——确定性求值，先算后写。

纪律：
  - 规则引擎只做【计算与标记】：读 rules 表 → 对每条命中证据独立判条件 → 产出 rule_hits。
  - 绝不改变召回序 / 过滤 / 提权（这些副作用留给未来④路标量约束，规则不碰）。
  - LLM 只拿 rule_hits 当参考事实去"翻译/解释/下结论"，规则命中与否由代码确定，可复现。
  - fail-closed：规则读取失败、条件取不到值、空条件——一律不触发，绝不影响检索主链路。
  - fake 后端无 PG，直接返回空（无规则 = 无规则标记，属稳态而非降级，故不留 degraded）。

条件 DSL 见 docs/api/10-rules.md：condition = {all:[子句], any:[子句]}，子句 = {field, op, value}。
"""
from __future__ import annotations

import logging

from ..config import effective_cfg, get_settings
from ..degrade.orchestrator import DegradedCollector
from .rrf import Hit

log = logging.getLogger("retrieval.fuse.rules")

# 可判定的字段白名单（超出白名单的子句恒不满足，防误触发）
FIELDS = ("kg_confidence", "rerank_score", "rrf_score", "kb_id", "section", "content")
NUM_OPS = {"lt", "lte", "gt", "gte", "eq", "ne"}
STR_OPS = {"eq", "ne", "contains"}


async def load_rules(collector: DegradedCollector) -> list[dict]:
    """读取启用且未删除的规则（real=asyncpg 现查 rules 表；fake=空，fail-closed）。

    规则增删改即时生效：每次请求现查，无缓存。
    读取失败只记 warning、返回空——规则是可选增强层，读不到就当"本查询无规则"，
    绝不拖累主链（故不新增 degraded 码，避免污染检索响应契约的 step 枚举）。
    """
    if effective_cfg().get("_backend") == "fake":
        return []
    try:
        import asyncpg
        conn = await asyncpg.connect(get_settings().pg_dsn)
        try:
            rows = await conn.fetch(
                """
                SELECT id, name, description, condition, action, priority, enabled, kb_id
                FROM rules
                WHERE enabled = true AND deleted_at IS NULL
                ORDER BY priority DESC, id
                """
            )
        finally:
            await conn.close()
        out: list[dict] = []
        for r in rows:
            out.append({
                "id": r["id"],
                "name": r["name"],
                "description": r["description"] or "",
                "condition": r["condition"] if isinstance(r["condition"], dict) else {},
                "action": r["action"] if isinstance(r["action"], dict) else {},
                "priority": int(r["priority"] or 0),
                "kb_id": str(r["kb_id"]) if r["kb_id"] is not None else None,
            })
        return out
    except Exception as e:
        log.warning("rules load failed (视为无规则): %s", e)
        return []


def _resolve(hit: Hit, parents: dict[str, dict], field: str):
    """把命中证据解析成字段值。kg_confidence 来自父块（PG），其余来自 Hit/meta。"""
    p = parents.get(hit.parent_id, {})
    if field == "kg_confidence":
        return p.get("kg_confidence")
    if field == "rerank_score":
        return hit.meta.get("rerank_score")
    if field == "rrf_score":
        return hit.meta.get("rrf_score")
    if field == "kb_id":
        return hit.meta.get("kb_id", "")
    if field == "section":
        return hit.meta.get("section")
    if field == "content":
        return p.get("content") or hit.content
    return None


def _cmp(value, op: str, target) -> bool:
    if value is None:
        return False  # fail-closed：字段取不到值恒不满足
    if op in NUM_OPS:
        try:
            a, b = float(value), float(target)
        except (TypeError, ValueError):
            return False
        return {
            "lt": a < b, "lte": a <= b, "gt": a > b, "gte": a >= b,
            "eq": a == b, "ne": a != b,
        }[op]
    if op in STR_OPS:
        s, t = str(value), str(target)
        return {"eq": s == t, "ne": s != t, "contains": t in s}[op]
    return False


def _clause_matches(clause, hit: Hit, parents: dict[str, dict]) -> bool:
    if not isinstance(clause, dict):
        return False
    field = clause.get("field")
    if field not in FIELDS:
        return False
    return _cmp(_resolve(hit, parents, field), clause.get("op"), clause.get("value"))


def _condition_matches(cond, hit: Hit, parents: dict[str, dict]) -> bool:
    if not isinstance(cond, dict):
        return False
    all_clauses = cond.get("all") or []
    any_clauses = cond.get("any") or []
    if not all_clauses and not any_clauses:
        return False  # fail-closed：空条件不触发（防误标全量）
    if all_clauses and not all(_clause_matches(c, hit, parents) for c in all_clauses):
        return False
    if any_clauses and not any(_clause_matches(c, hit, parents) for c in any_clauses):
        return False
    return True


def evaluate_rules(
    rules: list[dict],
    hits: list[Hit],
    parents: dict[str, dict],
    scope: list[str] | None,
) -> list[dict]:
    """确定性规则求值：返回 rule_hits（命中规则 + 命中证据编号 [n]）。

    返回项：{rule_id, name, label, severity, hint, priority, matched, reason}
      reason 里的 [n] 与 build_prompt 的证据编号（1-based over hits）严格一致。
    """
    if not rules or not hits:
        return []
    scope_set = set(scope or [])
    out: list[dict] = []
    for rule in rules:
        # 作用域：kb_id 为空 = 全局；否则必须命中当前检索 scope
        if rule.get("kb_id") and rule["kb_id"] not in scope_set:
            continue
        matched: list[str] = []
        reason: list[str] = []
        for i, h in enumerate(hits, start=1):
            if _condition_matches(rule["condition"], h, parents):
                matched.append(h.child_id)
                reason.append(f"[{i}]")
        if not matched:
            continue
        action = rule["action"]
        out.append({
            "rule_id": rule["id"],
            "name": rule["name"],
            "label": action.get("label") or rule["name"],
            "severity": action.get("severity", "info"),
            "hint": action.get("hint", ""),
            "priority": rule["priority"],
            "matched": matched,
            "reason": "命中证据 " + "、".join(reason),
        })
    out.sort(key=lambda r: (-r["priority"], r["rule_id"]))
    return out
