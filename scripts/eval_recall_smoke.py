"""极小冒烟评测:10 查询走 qingxi /graph/query 全链路,算文件级 recall@k / hit@k。

链路(与 POST /graph/query 同一条代码):意图分流 → 实体匹配 → A/B 双路召回 → 重排 →
(关系型)图扩展 → LLM 生成。入库侧走 ingest_document(清洗 md → 切块 → 嵌入 → 实体/关系
抽取 → 消歧 → Neo4j → 回写 Milvus),即「整个程序跑通」。

doc_id 直接复用评测语料 doc_id(如 txt/rd_028142.txt),查询命中 doc_id 即语料 doc_id,
召回映射零成本。

用法:
  PYTHONIOENCODING=utf-8 python scripts/eval_recall_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(r"C:/Users/A/Desktop/eval")
N_QUERIES = 10
N_DISTRACTORS = 15
KB_NAME = "eval-smoke-v2"
TEXT_MODALITIES = {"txt", "csv", "html", "docx", "pdf"}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _get_or_create_kb(name: str) -> str:
    from sqlalchemy import text
    from app.utils.database import SessionLocal
    async with SessionLocal() as db:
        row = (await db.execute(text("SELECT id FROM knowledge_bases WHERE name=:n"),
                                {"n": name})).first()
        if row:
            return str(row[0])
        rid = (await db.execute(text(
            "INSERT INTO knowledge_bases (id, name, created_at, updated_at) "
            "VALUES (gen_random_uuid(), :n, now(), now()) RETURNING id"), {"n": name})).scalar()
        await db.commit()
        return str(rid)


def _pick(qs: list[dict], corpus_by_id: dict, n: int, n_distractors: int):
    """选相关文档数最少的 n 条查询;干扰项取文本模态且不在相关集里的文档。"""
    picked = sorted(qs, key=lambda q: len(q["relevant"]))[:n]
    relevant: set[str] = set()
    for q in picked:
        relevant.update(q["relevant"])
    distractors = [d["doc_id"] for d in corpus_by_id.values()
                   if d["modality"] in TEXT_MODALITIES and d["doc_id"] not in relevant]
    distractors = distractors[:n_distractors]
    return picked, sorted(relevant), distractors


async def main() -> int:
    corpus = _load_jsonl(EVAL_DIR / "val_corpus.jsonl")
    queries = _load_jsonl(EVAL_DIR / "val_queries.jsonl")
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    picked, relevant_docs, distractors = _pick(queries, corpus_by_id, N_QUERIES, N_DISTRACTORS)
    all_docs = relevant_docs + distractors
    kb_id = await _get_or_create_kb(KB_NAME)

    print(f"[eval] kb={kb_id}")
    print(f"[eval] 查询 {len(picked)} 条 | 入库 {len(all_docs)} 篇 "
          f"(相关 {len(relevant_docs)} + 干扰 {len(distractors)})", flush=True)

    from app.graphrag.ingest import ingest_document

    t0 = time.time()
    done = 0
    for i, did in enumerate(all_docs, 1):
        d = corpus_by_id[did]
        text = (d.get("text") or "").strip()
        if not text:
            print(f"[ingest {i}/{len(all_docs)}] 跳过空文本 {did}", flush=True)
            continue
        try:
            s = await ingest_document(kb_id, did, text, d["modality"], did)
            done += 1
            print(f"[ingest {i}/{len(all_docs)}] {did:35s} ent={s['entities']} rel={s['relations']} "
                  f"耗时累计 {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[ingest {i}/{len(all_docs)}] {did:35s} FAIL: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
    print(f"[eval] 入库完成 {done} 篇,总耗时 {time.time()-t0:.0f}s", flush=True)

    # ---- 查询侧(与 /graph/query 同代码路径)----
    from app.graphrag.dict import load_relations
    from app.graphrag.embeddings import EmbeddingClient
    from app.graphrag.llm import LLMClient
    from app.graphrag.milvus_client import AsyncMilvusClient
    from app.graphrag.neo4j_client import Neo4jClient
    from app.graphrag.query.router import search as graph_search
    from app.utils.database import SessionLocal

    embeddings = EmbeddingClient()
    llm = LLMClient()
    milvus = AsyncMilvusClient()
    async with SessionLocal() as session:
        relation_defs = await load_relations(session)

    results: dict[str, dict] = {}
    for qi, q in enumerate(picked, 1):
        qtext = q["query"]
        tq = time.time()
        try:
            async with Neo4jClient() as neo4j:
                res = await graph_search(qtext, milvus, neo4j, embeddings, llm,
                                         relation_defs, kb_id=kb_id)
        except Exception as e:
            print(f"[query {qi}/{len(picked)}] {q['qid']:28s} FAIL: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            results[q["qid"]] = {"query": qtext, "relevant": q["relevant"], "ranked": []}
            continue
        ranked: list[str] = []
        seen: set[str] = set()
        for h in res.get("hits", []):
            did = h.get("doc_id")
            if did and did not in seen:
                seen.add(did)
                ranked.append(did)
        results[q["qid"]] = {"query": qtext, "relevant": q["relevant"], "ranked": ranked,
                             "intent": res.get("intent"),
                             "matched": res.get("matched_entities", [])}
        print(f"[query {qi}/{len(picked)}] {q['qid']:28s} intent={res.get('intent'):10s} "
              f"matched={len(res.get('matched_entities', []))} hits={len(ranked)} "
              f"耗时 {time.time()-tq:.0f}s", flush=True)

    # ---- 算 recall@k / hit@k ----
    ks = [1, 5, 10, 20]
    print("\n===== 逐查询召回 =====", flush=True)
    recall_sums = {k: 0.0 for k in ks}
    hit_sums = {k: 0 for k in ks}
    for qid, r in results.items():
        rel = set(r["relevant"])
        ranked = r["ranked"]
        line = f"{qid:28s} rel={len(rel):2d}"
        for k in ks:
            found = len(rel & set(ranked[:k]))
            recall = found / len(rel) if rel else 0.0
            recall_sums[k] += recall
            hit_sums[k] += 1 if found else 0
            line += f"  R@{k}={recall:.2f}"
        print(line, flush=True)

    n = len(results)
    print("\n===== 宏平均 (10 查询) =====", flush=True)
    for k in ks:
        print(f"  Recall@{k}: {recall_sums[k]/n:.4f}    Hit@{k}: {hit_sums[k]}/{n} = {hit_sums[k]/n:.4f}",
              flush=True)

    # 存结果
    out = EVAL_DIR / "results_qingxi_smoke.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for qid, r in results.items():
            f.write(json.dumps({"qid": qid, "doc_ids": r["ranked"]}, ensure_ascii=False) + "\n")
    print(f"\n[OK] 结果已写 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
