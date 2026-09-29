"""抽取评测(金标准 F1):校准实体/关系抽取的 confidence 阈值与字典定义。

用真实标注金标准 GOLDEN 逐条跑 LLMClient.extract_entities/extract_relations,算实体/关系
P/R/F1,并按 confidence 分桶给出阈值建议——用于校准 .env 的 confidence_high/low(默认 0.9/0.6)。

运行(本地 Qwen vLLM 就绪且 .env 设 provider=qwen + base_url 指向内网):
    python -m app.graphrag.eval.extract_eval

「先搭框架」:GOLDEN 默认空占位;真实数据到位后按下方注释逐条填 text + 期望实体/关系即可。
"""

from __future__ import annotations

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ..dict import load_entity_defs, load_relations
from ..llm import LLMClient

# ---- 金标准:真实数据到位后逐条填 ----
# text = 一段文本;entities = 期望实体 [{name, entity_type}];
# relations = 期望关系 [{head, tail, rel_type}](head/tail 用实体名,rel_type 用 en)。
GOLDEN: list[dict] = [
    # {
    #     "text": "……",
    #     "entities": [{"name": "雷达站", "entity_type": "装备"}],
    #     "relations": [{"head": "雷达站", "tail": "…", "rel_type": "BELONGS_TO"}],
    # },
]


def _entity_f1(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    gold_keys = {(g["name"], g["entity_type"]) for g in gold}
    pred_keys = {(p["name"], p["entity_type"]) for p in pred}
    return (len(gold_keys & pred_keys), len(pred_keys - gold_keys), len(gold_keys - pred_keys))


def _relation_f1(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    gold_keys = {(g["head"], g["tail"], g["rel_type"]) for g in gold}
    pred_keys = {(p["head"], p["tail"], p["rel_type"]) for p in pred}
    return (len(gold_keys & pred_keys), len(pred_keys - gold_keys), len(gold_keys - pred_keys))


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


async def main() -> int:
    from ...utils.database import SessionLocal, init_db

    await init_db()
    llm = LLMClient()

    if not GOLDEN:
        print("GOLDEN 为空(占位)。真实数据到位后按文件头注释填 text/entities/relations,"
              "再运行本脚本校准 confidence 阈值。")
        return 0

    async with SessionLocal() as session:
        entity_defs = await load_entity_defs(session)
        relation_defs = (await load_relations(session))["relation_types"]
    entity_types = [d["name"] for d in entity_defs]

    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    recall_by_thresh: dict[float, list[bool]] = {}  # confidence 阈值 -> 每个 gold 实体是否被召回

    for i, case in enumerate(GOLDEN):
        text = case["text"]
        gold_entities = case.get("entities", [])
        gold_relations = case.get("relations", [])

        pred_entities = await asyncio.to_thread(llm.extract_entities, text,
                                                entity_types, entity_defs)
        pred_relations = await asyncio.to_thread(llm.extract_relations, text,
                                                 pred_entities, relation_defs)

        tp, fp, fn = _entity_f1(pred_entities, gold_entities)
        ent_tp += tp
        ent_fp += fp
        ent_fn += fn

        tp, fp, fn = _relation_f1(pred_relations, gold_relations)
        rel_tp += tp
        rel_fp += fp
        rel_fn += fn

        pred_by_key = {(e["name"], e["entity_type"]): e for e in pred_entities}
        gold_keys = {(g["name"], g["entity_type"]) for g in gold_entities}
        for thresh in (0.1, 0.3, 0.5, 0.7, 0.9):
            kept = {k for k, e in pred_by_key.items()
                    if e.get("confidence", 0.0) >= thresh}
            for k in gold_keys:
                recall_by_thresh.setdefault(thresh, []).append(k in kept)

    p, r, f1 = _f1(ent_tp, ent_fp, ent_fn)
    print(f"[实体] TP={ent_tp} FP={ent_fp} FN={ent_fn} → P={p:.3f} R={r:.3f} F1={f1:.3f}")
    p, r, f1 = _f1(rel_tp, rel_fp, rel_fn)
    print(f"[关系] TP={rel_tp} FP={rel_fp} FN={rel_fn} → P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("[confidence 阈值建议] 各阈值下 gold 实体的召回率(召回骤降处即阈值过严):")
    for thresh in sorted(recall_by_thresh):
        hits = sum(recall_by_thresh[thresh])
        total = len(recall_by_thresh[thresh])
        print(f"  conf≥{thresh:.1f}: {hits}/{total} = {hits / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
