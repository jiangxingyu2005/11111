"""金标准集构造（EV-0）。

从 data/seed_date/ 装载多域多模态 seed，按 6 类客观 gold 生成评测集，8:2 分层划分。
gold 由客观规则判定，不依赖人工标注：
  A 参数型（标量容差）  B 型号→调制  C 型号→频段
  D 平台型  E 装备档案型（html/docx）  F 媒体视频型（元数据口径）

单一事实源：本文件从 scripts/eval_date.py 原样迁移，eval_date.py 已改为薄壳委托此处，
避免两处逻辑漂移。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

GENERIC_PLATFORMS = {"驱逐舰", "护卫舰", "巡洋舰", "航母", "预警机", "无人机", "轰炸机",
                     "战斗机", "地面站", "机动站", "潜艇", "舰载", "机载", "车载", "陆基"}


def load_seed(seed_dir: Path) -> tuple[list[dict], dict[str, str], list[dict]]:
    """返回 (chunks, entities{entity_id:name}, records)。"""
    def rd(name: str):
        return json.loads((seed_dir / f"{name}.json").read_text(encoding="utf-8"))

    chunks = rd("chunks")
    entities = {e["entity_id"]: e["name"] for e in rd("entities")}
    for c in chunks:
        sp = c.get("sparse")
        if isinstance(sp, dict):
            c["sparse"] = {int(k): float(v) for k, v in sp.items()}
        c["_enames"] = [entities.get(x, "") for x in (c.get("kg_entity_ids") or [])]
    records = rd("records")
    return chunks, entities, records


def build_eval_set(records: list[dict], chunks: list[dict], max_per_type: int) -> list[dict]:
    """6 类客观 gold 评测集，每条 {"q","type","gold":{child_id:grade}}（grade 2=核心 / 1=相关）。"""
    def has(c: dict, name: str) -> bool:
        return bool(name) and name in (c.get("content") or "")

    by_type: dict[str, list[dict]] = {}

    # ── A 参数型：CSV 带完整标量（rf+prf+pw），gold 由标量容差判定 ──
    tol = {"rf": 0.3, "prf_rel": 0.05, "pw_rel": 0.10}
    for r in records:
        rf, prf, pw = r.get("rf_ghz"), r.get("prf_hz"), r.get("pw_us")
        if rf is None or prf is None or pw is None:
            continue
        q = f"载频约{rf}GHz、脉冲重复频率{prf}Hz、脉宽{pw}us 的雷达信号参数"
        gold: dict[str, int] = {}
        for c in chunks:
            sc = c.get("scalar") or {}
            if sc.get("rf_ghz") is None:
                continue
            exact = (sc["rf_ghz"] == rf and sc["prf_hz"] == prf and sc["pw_us"] == pw)
            ok = (abs(sc["rf_ghz"] - rf) <= tol["rf"]
                  and abs(sc["prf_hz"] - prf) <= max(1.0, prf * tol["prf_rel"])
                  and abs(sc["pw_us"] - pw) <= max(0.05, pw * tol["pw_rel"]))
            if exact:
                gold[c["child_id"]] = 2
            elif ok:
                gold[c["child_id"]] = 1
        if gold:
            by_type.setdefault("A参数型", []).append({"q": q, "type": "A参数型", "gold": gold})

    # ── B 型号→调制 ──
    for r in records:
        model, mod = r.get("model", ""), r.get("mod", "")
        if not model or not mod:
            continue
        q = f"{model} 采用的脉内调制方式"
        gold = {}
        for c in chunks:
            hm, hmod = has(c, model), has(c, mod)
            if hm and hmod:
                gold[c["child_id"]] = 2
            elif hm:
                gold[c["child_id"]] = 1
        if gold:
            by_type.setdefault("B型号-调制型", []).append({"q": q, "type": "B型号-调制型", "gold": gold})

    # ── C 型号→频段 ──
    for r in records:
        model, band = r.get("model", ""), r.get("band", "")
        if not model or not band:
            continue
        q = f"{model} 的工作频段"
        gold = {}
        for c in chunks:
            hm, hb = has(c, model), has(c, band)
            if hm and hb:
                gold[c["child_id"]] = 2
            elif hm:
                gold[c["child_id"]] = 1
        if gold:
            by_type.setdefault("C型号-频段型", []).append({"q": q, "type": "C型号-频段型", "gold": gold})

    # ── D 平台型 ──
    seen_plat = set()
    for r in records:
        plat = r.get("platform", "").strip()
        if not plat or plat in seen_plat or plat in GENERIC_PLATFORMS or len(plat) > 30:
            continue
        seen_plat.add(plat)
        q = f"{plat} 平台搭载雷达的类型"
        gold = {}
        for c in chunks:
            if has(c, plat):
                gold[c["child_id"]] = 2
        if gold:
            by_type.setdefault("D平台型", []).append({"q": q, "type": "D平台型", "gold": gold})

    # ── E 装备档案型（装备域：html/docx 装备档案）──
    for r in records:
        if r.get("domain") != "equipment":
            continue
        name = r.get("model", "").strip()
        if not name:
            continue
        q = f"{name} 装备档案"
        gold = {}
        for c in chunks:
            if has(c, name):
                gold[c["child_id"]] = 2
        if gold:
            by_type.setdefault("E装备档案型", []).append({"q": q, "type": "E装备档案型", "gold": gold})

    # ── F 媒体视频型（媒体域：视频↔装备关联，元数据口径）──
    for r in records:
        if r.get("modality") != "video" or not r.get("platform"):
            continue
        name = r["platform"].strip()
        q = f"{name} 的视频"
        gold = {}
        for c in chunks:
            if has(c, name) and "装备视频" in (c.get("content") or ""):
                gold[c["child_id"]] = 2
        if gold:
            by_type.setdefault("F媒体视频型", []).append({"q": q, "type": "F媒体视频型", "gold": gold})

    # 每类去重（同型号在 txt+csv 复现会生成同文本 query，gold 已含全部相关块）
    rng = random.Random(7)
    evalset: list[dict] = []
    for t, items in by_type.items():
        uniq = {}
        for q in items:
            uniq.setdefault(q["q"], q)
        items = list(uniq.values())
        rng.shuffle(items)
        evalset.extend(items[:max_per_type])
    return evalset


def split_82(evalset: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """按类型分层 8:2 划分，验证集每类至少 1 条。"""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in evalset:
        by_type.setdefault(q["type"], []).append(q)
    test, val = [], []
    for t, items in by_type.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.2))
        val.extend(items[:n_val])
        test.extend(items[n_val:])
    rng.shuffle(test)
    rng.shuffle(val)
    return test, val
