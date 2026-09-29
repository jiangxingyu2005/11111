"""规则表 + 置信度模块 smoke 验证：fake 后端 + 模拟数据，跑通「规则加载 → 确定性求值 → 全链路触发」。

与 smoke.py 的区别：LLM 生成非本模块关注点，fake 通道下不启 model_stub，
generation 自然降级为 generation_unavailable，但 rules 事件与 done.rule_hits 均在生成前/后
同步产出，可直接断言。四层验证：
  [A] load_rules（fake 读 seed/rules.json）非空 + 形状正确 + priority 降序
  [B] evaluate_rules 确定性：6 个字段（kg_confidence/rerank_score/rrf_score/kb_id/section/content）
      与数值/字符串算子逐一命中；fail-closed（空条件/字段缺失/未知字段/None 值不触发）
  [C] 真实 seed 数据集成：FakeStore 父块 kg_confidence + 子块 content/section/kb_id 喂进引擎，
      验证 kg_confidence 高/低阈值两态、content contains、kb_id eq + section contains
  [D] 全链路 HTTP：POST /api/retrieval，rules 事件 + done.rule_hits 非空，results 带 rrf/rerank 分

用法：python scripts/smoke_rules.py
退出码：0=全绿；非 0=有失败（CI 可直接用）
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_BASE_ENV = {
    "RETRIEVAL_BACKEND": "fake",
    "RETRIEVAL_SEED_DIR": "data/seed",
    "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "retrieval_ro", "POSTGRES_PASSWORD": "dev_only",
    "POSTGRES_DATABASE": "qingxi",
    "MILVUS_HOST": "127.0.0.1", "MILVUS_PORT": "19530",
    "NEO4J_URI": "bolt://127.0.0.1:7687", "NEO4J_USER": "ro", "NEO4J_PASSWORD": "dev_only",
    "GRAPHRAG_EMBEDDING_MODEL": "BAAI/bge-m3",
    "PYTHONIOENCODING": "utf-8",
}
for k, v in _BASE_ENV.items():
    os.environ[k] = v

from app.config import reset_caches  # noqa: E402

reset_caches()
from app.fuse.rrf import Hit  # noqa: E402
from app.fuse.rules import evaluate_rules, load_rules  # noqa: E402
from app.store.fake import get_fake_store  # noqa: E402

SCOPE = ["kb_public", "kb_internal"]
QUERY = "X波段 线性调频 雷达信号，载频 9GHz 左右，PRF 1200Hz"


def _hit(cid: str, pid: str, *, content: str = "", kb_id: str = "", section: str = "",
         rrf: float | None = None, rerank: float | None = None) -> Hit:
    h = Hit(child_id=cid, parent_id=pid, content=content)
    h.meta["kb_id"] = kb_id
    if section:
        h.meta["section"] = section
    if rrf is not None:
        h.meta["rrf_score"] = rrf
    if rerank is not None:
        h.meta["rerank_score"] = rerank
    return h


def _run() -> list[str]:
    fails: list[str] = []

    def chk(cond: bool, msg: str) -> None:
        print(("  [OK] " if cond else "  [FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    # ── [A] 规则加载（fake 读 seed/rules.json）──
    rules = asyncio.run(load_rules(None))
    chk(len(rules) == 4, f"[A] 加载规则条数={len(rules)}（应=4）")
    ids = [r["id"] for r in rules]
    chk("R_001" in ids and "R_004" in ids, f"[A] 含 R_001/R_004（实际={ids}）")
    chk(all(isinstance(r["condition"], dict) and isinstance(r["action"], dict) for r in rules),
        "[A] condition/action 均为 dict")
    prios = [r["priority"] for r in rules]
    chk(prios == sorted(prios, reverse=True), f"[A] 按 priority 降序（{prios}）")

    # ── [B] evaluate_rules 确定性（全字段 + 算子 + fail-closed）──
    parents = {"p0": {"kg_confidence": 0.55, "content": "AN/SPS-49 火控雷达，载频4.65GHz"}}
    h = _hit("c0", "p0", content="火控雷达信号", kb_id="kb_public", section="东南海域火控记录",
             rrf=0.03, rerank=0.4)

    # 六字段逐一命中
    def fire(cond) -> bool:
        return bool(evaluate_rules([{"id": "T", "name": "t", "description": "", "condition": cond,
                                     "action": {"type": "mark", "label": "x", "severity": "info",
                                                "hint": ""}, "priority": 0, "kb_id": None}],
                                   [h], parents, SCOPE))

    chk(fire({"all": [{"field": "kg_confidence", "op": "lt", "value": 0.7}]}),
        "[B] kg_confidence lt 0.7 命中（0.55 < 0.7）")
    chk(fire({"all": [{"field": "rerank_score", "op": "lt", "value": 0.5}]}),
        "[B] rerank_score lt 0.5 命中（0.4 < 0.5）")
    chk(fire({"all": [{"field": "rrf_score", "op": "gt", "value": 0}]}),
        "[B] rrf_score gt 0 命中（0.03 > 0）")
    chk(fire({"all": [{"field": "kb_id", "op": "eq", "value": "kb_public"}]}),
        "[B] kb_id eq 'kb_public' 命中")
    chk(fire({"all": [{"field": "section", "op": "contains", "value": "火控"}]}),
        "[B] section contains '火控' 命中")
    chk(fire({"all": [{"field": "content", "op": "contains", "value": "火控雷达"}]}),
        "[B] content contains '火控雷达' 命中")
    # all/any 组合语义
    chk(not fire({"all": [{"field": "rerank_score", "op": "lt", "value": 0.5}],
                  "any": [{"field": "content", "op": "contains", "value": "不存在"}]}),
        "[B] all 满足 + any 无满足 -> 不命中（AND 语义）")
    # fail-closed：空条件 / 字段缺失 / 未知字段 / None 值
    chk(not fire({"all": [], "any": []}), "[B] 空条件不触发（fail-closed）")
    p_missing = _hit("c1", "p_missing", content="x", kb_id="kb_public", rrf=0.01)
    chk(not evaluate_rules([{"id": "T", "name": "t", "description": "", "condition": {
        "all": [{"field": "kg_confidence", "op": "lt", "value": 0.7}]},
        "action": {"type": "mark", "label": "x", "severity": "info", "hint": ""},
        "priority": 0, "kb_id": None}], [p_missing], {}, SCOPE),
        "[B] kg_confidence 缺失（None）不触发（fail-closed）")
    chk(not fire({"all": [{"field": "not_a_field", "op": "eq", "value": "x"}]}),
        "[B] 未知字段不触发（白名单外恒不满足）")

    # ── [C] 真实 seed 数据集成 ──
    store = get_fake_store()
    low = next((p for p in store.parents.values() if p.get("kg_confidence", 1.0) < 0.7), None)
    hi = next((p for p in store.parents.values() if p.get("kg_confidence", 0.0) >= 0.8), None)
    firec = next((p for p in store.parents.values() if "火控雷达" in p.get("content", "")), None)
    chk(bool(low and hi and firec),
        f"[C] seed 同时含低/高 kg_confidence 与火控雷达父块（low={low['parent_id'] if low else None} "
        f"kg={low['kg_confidence'] if low else None}）")

    def real_hit(p) -> Hit:
        cid = next((c["child_id"] for c in store.chunks.values()
                    if c["parent_id"] == p["parent_id"]), "")
        chunk = store.chunks.get(cid, {})
        hh = _hit(cid, p["parent_id"], content=chunk.get("content", ""),
                  kb_id=chunk.get("kb_id", ""), section=chunk.get("section", ""),
                  rrf=0.02, rerank=0.4)
        return hh

    parents_low = {low["parent_id"]: {"kg_confidence": low["kg_confidence"],
                                      "content": low["content"]}}
    hit_low = real_hit(low)
    rules = asyncio.run(load_rules(None))  # 重载一次（真实 seed 规则）
    rh = evaluate_rules(rules, [hit_low], parents_low, SCOPE)
    chk(any(r["rule_id"] == "R_001" for r in rh),
        f"[C] kg_confidence=0.5 触发 R_001（any: content含火控 OR kg<0.7），命中 {[r['rule_id'] for r in rh]}")

    # 高置信父块：R_001 的 kg<0.7 不满足、且不含火控雷达 -> 不触发（除非 content 命中）
    parents_hi = {hi["parent_id"]: {"kg_confidence": hi["kg_confidence"],
                                    "content": hi["content"]}}
    hit_hi = real_hit(hi)
    rh_hi = evaluate_rules([r for r in rules if r["id"] == "R_001"], [hit_hi], parents_hi, SCOPE)
    hi_has_firec = "火控雷达" in hi["content"]
    expect_trigger = hi_has_firec  # R_001 any 命中 content 才触发
    chk(bool(rh_hi) == expect_trigger,
        f"[C] 高置信父块 R_001 触发={bool(rh_hi)}（含火控={hi_has_firec}，应={expect_trigger}）")

    # ── [D] 全链路 HTTP ──
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    with client.stream("POST", "/api/retrieval", json={"query": QUERY, "scope": SCOPE}) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}"
        events = []
        ev, buf = None, []
        for raw in r.iter_lines():
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw.startswith("event:"):
                ev = raw[6:].strip()
            elif raw.startswith("data:"):
                buf.append(raw[5:].strip())
            elif raw.strip() == "" and ev is not None:
                try:
                    events.append((ev, json.loads("".join(buf))))
                except ValueError:
                    pass
                ev, buf = None, []
    rules_ev = [d for e, d in events if e == "rules"]
    done_ev = [d for e, d in events if e == "done"]
    results_ev = [d for e, d in events if e == "results"]
    chk(bool(rules_ev) and bool(rules_ev[0].get("hits")),
        f"[D] rules 事件非空（{len(rules_ev[0]['hits']) if rules_ev else 0} 条 rule_hits）")
    chk(bool(done_ev) and bool(done_ev[0].get("rule_hits")),
        "[D] done.rule_hits 非空")
    hit_ids = {r["rule_id"] for r in done_ev[0]["rule_hits"]} if done_ev else set()
    chk("R_004" in hit_ids, f"[D] 端到端触发 R_004（rrf>0 恒真），实际 rule_ids={hit_ids}")
    scores = [res.get("score") for d in results_ev for res in d.get("hits", [])]
    chk(all(s and s.get("rrf") is not None and s.get("rerank") is not None for s in scores),
        f"[D] results 每条命中均带 rrf+rerank 置信度（{len(scores)} 条）")

    return fails


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fails = _run()
    print("\n=== 结果 ===")
    if fails:
        print("[FAIL] " + "; ".join(fails))
        return 1
    print("[smoke_rules] 全部通过 [PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
