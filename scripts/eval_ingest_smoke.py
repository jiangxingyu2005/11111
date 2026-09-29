"""单文档全链路冒烟测试:用评测语料里一篇装备档案走完整入库,确认实体+关系都>0。

用法:
  PYTHONIOENCODING=utf-8 python scripts/eval_ingest_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KB_ID = "28e437c4-ad3e-420b-9f81-35e3ddbe8ee9"  # eval-0929
CORPUS = Path(r"C:/Users/A/Desktop/eval/test_corpus.jsonl")


async def main() -> int:
    # 取一篇装备档案(AN/TPQ-50)
    target = None
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] == "txt/rd_028142.txt":
                target = d
                break
    if not target:
        print("[FAIL] 未在语料中找到 txt/rd_028142.txt")
        return 1

    text = target["text"]
    print(f"[smoke] doc_id={target['doc_id']}  text_len={len(text)}  entities={target['entities']}")

    from app.graphrag.ingest import ingest_document

    doc_uuid = str(uuid4())
    t0 = asyncio.get_event_loop().time()
    summary = await ingest_document(KB_ID, doc_uuid, text, "txt", target["doc_id"])
    dt = asyncio.get_event_loop().time() - t0

    print(f"[smoke] 耗时 {dt:.1f}s  parents={summary['parents']} children={summary['children']} "
          f"entities={summary['entities']} relations={summary['relations']} binds={summary['binds']}")

    if summary["entities"] == 0:
        print("[FAIL] entities=0")
        return 1
    if summary["relations"] == 0:
        print("[FAIL] relations=0")
        return 1
    print(f"[OK] 全链路跑通:实体 {summary['entities']} 个,关系 {summary['relations']} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
