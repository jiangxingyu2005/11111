"""预设推理规则引擎（需求(7)「触发预设推理规则」）——确定性求值，先算后写。

纪律：
  - 规则引擎只做【计算与标记】：读 rules 表 → 对每条命中证据独立判条件 → 产出 rule_hits。
  - 绝不改变召回序 / 过滤 / 提权（这些副作用留给未来④路标量约束，规则不碰）。
  - LLM 只拿 rule_hits 当参考事实去"翻译/解释/下结论"，规则命中与否由代码确定，可复现。
  - fail-closed：规则读取失败、条件取不到值、空条件——一律不触发，绝不影响检索主链路。
  - fake 后端无 PG，改读 seed/rules.json（与真库同构）；读不到就当无规则（稳态而非降级，故不留 degraded）。

条件 DSL 见 docs/api/10-rules.md：condition = {all:[子句], any:[子句]}，子句 = {field, op, value}。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import ROOT, effective_cfg, get_settings
from ..degrade.orchestrator import DegradedCollector
from .rrf import Hit

log = logging.getLogger("retrieval.fuse.rules")

# 可判定的字段白名单（超出白名单的子句恒不满足，防误触发）
FIELDS = ("kg_confidence", "rerank_score", "rrf_score", "kb_id", "section", "content")
NUM_OPS = {"lt", "lte", "gt", "gte", "eq", "ne"}
STR_OPS = {"eq", "ne", "contains"}


def _normalize_rule(r: dict) -> dict:
    """把存储层的一条规则规范成引擎消费的 dict（real 与 fake 共用同一形状）。"""
    return {
        "id": r.get("id", ""),
        "name": r.get("name", ""),
        "description": r.get("description") or "",
        "condition": r.get("condition") if isinstance(r.get("condition"), dict) else {},
        "action": r.get("action") if isinstance(r.get("action"), dict) else {},
        "priority": int(r.get("priority") or 0),
        "kb_id": str(r.get("kb_id")) if r.get("kb_id") is not None else None,
    }


def _load_fake_rules() -> list[dict]:
    """fake 后端：从 seed/rules.json 读预设规则（镜像 FakeStore._read），无 PG 也能跑通规则链路。

    rules.json 缺失/损坏 -> 返回空（无规则 = 稳态），与 real 读库失败同一 fail-closed 语义。
    """
    seed_dir = Path(get_settings().retrieval_seed_dir)
    if not seed_dir.is_absolute():
        seed_dir = ROOT / seed_dir
    path = seed_dir / "rules.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("fake rules load failed (视为无规则): %s", e)
        return []
    if not isinstance(data, list):
        return []
    return [_normalize_rule(r) for r in data if isinstance(r, dict)]


async def load_rules(collector: DegradedCollector) -> list[dict]:
    """读取启用且未删除的规则（real=asyncpg 现查 rules 表；fake=读 seed/rules.json，fail-closed）。

    规则增删改即时生效：每次请求现查，无缓存。
    读取失败只记 warning、返回空——规则是可选增强层，读不到就当"本查询无规则"，
    绝不拖累主链（故不新增 degraded 码，避免污染检索响应契约的 step 枚举）。
    """
    if effective_cfg().get("_backend") == "fake":
        return _load_fake_rules()
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
        return [_normalize_rule(dict(r)) for r in rows]
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
    # eq/ne 对 number/string 均适用（见 docs/api/10-rules.md）：
    # 先试数值比较，数值化失败则回退字符串比较——kb_id/section/content 等字符串字段的
    # eq/ne 依赖此回退（否则被 NUM 分支的 float() 误判为恒不满足）。
    if op in {"eq", "ne"}:
        try:
            eq = float(value) == float(target)
        except (TypeError, ValueError):
            eq = str(value) == str(target)
        return eq if op == "eq" else not eq
    if op in NUM_OPS:
        try:
            a, b = float(value), float(target)
        except (TypeError, ValueError):
            return False
        return {
            "lt": a < b, "lte": a <= b, "gt": a > b, "gte": a >= b,
        }[op]
    if op in STR_OPS:
        s, t = str(value), str(target)
        return {"contains": t in s}[op]
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
