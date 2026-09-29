"""规则引擎确定性求值单测（需求(7)/(8)，纯函数，无外部依赖）。

重点回归：_cmp 的 eq/ne 对字符串字段（kb_id/section/content）必须可用——
曾因 NUM 分支先 float() 导致字符串 eq/ne 恒不满足（见 smoke_rules 的 [B] kb_id eq 用例）。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.fuse.rrf import Hit
from app.fuse.rules import evaluate_rules


def _rule(cond: dict, *, rid: str = "R", kb_id: str | None = None, priority: int = 0) -> dict:
    return {"id": rid, "name": rid, "description": "", "condition": cond,
            "action": {"type": "mark", "label": "x", "severity": "info", "hint": ""},
            "priority": priority, "kb_id": kb_id}


def _hit(*, kg: float | None = 0.55, rerank: float = 0.4, rrf: float = 0.03,
         kb_id: str = "kb_public", section: str = "东南海域火控记录",
         content: str = "AN/SPS-49 火控雷达") -> tuple[Hit, dict]:
    h = Hit(child_id="c0", parent_id="p0", content=content)
    h.meta["kb_id"] = kb_id
    h.meta["section"] = section
    h.meta["rerank_score"] = rerank
    h.meta["rrf_score"] = rrf
    parents = {"p0": {"kg_confidence": kg, "content": content}}
    return h, parents


def _fire(cond, h, parents, *, rule_kb_id=None, scope=None) -> bool:
    return bool(evaluate_rules([_rule(cond, kb_id=rule_kb_id)], [h], parents, scope))


def test_string_eq_ne_on_kb_id():
    """字符串字段 eq/ne 可用（回归：曾被 NUM 分支 float() 吞掉恒不满足）。"""
    h, parents = _hit()
    assert _fire({"all": [{"field": "kb_id", "op": "eq", "value": "kb_public"}]}, h, parents)
    assert _fire({"all": [{"field": "kb_id", "op": "ne", "value": "kb_internal"}]}, h, parents)
    assert not _fire({"all": [{"field": "kb_id", "op": "eq", "value": "kb_internal"}]}, h, parents)


def test_string_contains_section_and_content():
    h, parents = _hit()
    assert _fire({"all": [{"field": "section", "op": "contains", "value": "火控"}]}, h, parents)
    assert _fire({"all": [{"field": "content", "op": "contains", "value": "火控雷达"}]}, h, parents)
    assert not _fire({"all": [{"field": "content", "op": "contains", "value": "不存在"}]}, h, parents)


def test_numeric_eq_ne_still_work():
    h, parents = _hit(kg=0.55, rerank=0.4)
    assert _fire({"all": [{"field": "kg_confidence", "op": "eq", "value": 0.55}]}, h, parents)
    assert _fire({"all": [{"field": "rerank_score", "op": "ne", "value": 0.9}]}, h, parents)
    assert _fire({"all": [{"field": "kg_confidence", "op": "lt", "value": 0.7}]}, h, parents)
    assert _fire({"all": [{"field": "rrf_score", "op": "gt", "value": 0}]}, h, parents)


def test_all_any_semantics():
    h, parents = _hit(rerank=0.4)
    # all 满足 + any 无满足 -> 不命中（all 与 any 是 AND 关系）
    assert not _fire({"all": [{"field": "rerank_score", "op": "lt", "value": 0.5}],
                      "any": [{"field": "content", "op": "contains", "value": "不存在"}]}, h, parents)
    # any 之一满足即命中
    assert _fire({"any": [{"field": "content", "op": "contains", "value": "不存在"},
                          {"field": "kg_confidence", "op": "lt", "value": 0.7}]}, h, parents)


def test_fail_closed():
    h, parents = _hit()
    # 空条件不触发
    assert not _fire({"all": [], "any": []}, h, parents)
    # 未知字段（白名单外）不触发
    assert not _fire({"all": [{"field": "not_a_field", "op": "eq", "value": "x"}]}, h, parents)
    # kg_confidence 缺失（None）不触发
    h2, parents2 = _hit(kg=None)
    assert not _fire({"all": [{"field": "kg_confidence", "op": "lt", "value": 0.7}]}, h2, parents2)
    # 空规则集 / 空命中集 -> 空结果
    assert evaluate_rules([], [h], parents, None) == []
    assert evaluate_rules([_rule({"all": [{"field": "kb_id", "op": "eq", "value": "kb_public"}]})],
                          [], {}, None) == []


def test_rule_scope_kb_id():
    h, parents = _hit(kb_id="kb_public")
    rule = {"all": [{"field": "kb_id", "op": "eq", "value": "kb_public"}]}
    # 规则作用域 kb_id=kb_internal 不覆盖命中 kb_public -> 跳过
    assert not _fire(rule, h, parents, rule_kb_id="kb_internal", scope=["kb_public"])
    # 全局规则（kb_id=None）不跳
    assert _fire(rule, h, parents, scope=["kb_public"])
