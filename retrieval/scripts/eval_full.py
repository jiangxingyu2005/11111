# -*- coding: utf-8 -*-
"""评测集走【当前程序完整召回链路】—— 用 retrieval 的真实模块，非复刻。

链路（与 app/api/retrieval.py 一致，只去掉对召回率无关的 LLM 生成 / L2 解析）：
  QRY 查询 -> ① vector 路(Milvus dense+sparse hybrid) + ③ graph 路(实体链接 KNN + Neo4j 图扩展
             + BINDS 回捞) 并行 -> 外层 RRF(k=60) -> rerank(bge-reranker-v2-m3) -> 输出 parent_id(=文件)

数据：把评测集 test_corpus.jsonl 的 411 个有文本文件 + 197 个实体，灌进当前程序正在用的
  Milvus(retrieval_chunks / retrieval_entities) + Neo4j(Entity/Chunk/BINDS)，scope=kb_eval 隔离。

用法:
    python scripts/eval_full.py --seed          # 建库（灌评测集数据）
    python scripts/eval_full.py --run           # 走完整链路检索 + 输出结果
    python scripts/eval_full.py                 # seed + run 一起
产出(写到 C:/Users/A/Desktop/eval/)：
    results_full_recall.jsonl   RRF 后 top-50（召回阶段，用于 recall@1/5/10/20）
    results_full_rerank.jsonl   rerank 后 top-5（当前程序最终进上下文的候选，只能到 k=5）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(r"C:\Users\A\Desktop\eval")
CORPUS = EVAL_DIR / "test_corpus.jsonl"
QUERIES = EVAL_DIR / "test_queries.jsonl"
EMBED_URL = "http://127.0.0.1:8010/embed"
SCOPE = ["kb_eval"]
KB = "kb_eval"
CACHE_DOC = EVAL_DIR / "_emb_cache_doc.json"
CACHE_ENT = EVAL_DIR / "_emb_cache_entity.json"


# ────────────────────────── 嵌入（复用 model_stub，缓存落盘） ──────────────────────────
def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _embed_batch(texts: list[str], typ: str) -> list[dict]:
    out = []
    for i in range(0, len(texts), 16):
        body = _post(EMBED_URL, {"texts": texts[i:i + 16], "type": typ})
        for e in body["embeddings"]:
            out.append({"dense": [float(x) for x in e["dense"]],
                        "sparse": {int(k): float(v) for k, v in (e.get("sparse") or {}).items()}})
    return out


def embed_cached(texts: list[str], typ: str, cache_path: Path) -> list[dict]:
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    miss = [i for i, t in enumerate(texts) if f"{typ}::{t}" not in cache]
    if miss:
        print(f"[embed] {typ} 缺 {len(miss)} 条，调 model_stub ...", file=sys.stderr)
        t0 = time.time()
        embs = _embed_batch([texts[i] for i in miss], typ)
        for i, e in zip(miss, embs):
            cache[f"{typ}::{texts[i]}"] = e
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"[embed] {typ} 完成 {len(miss)} 条 {time.time()-t0:.1f}s", file=sys.stderr)
    return [cache[f"{typ}::{t}"] for t in texts]


# ────────────────────────── 语料加载 ──────────────────────────
def load_corpus() -> tuple[list[str], list[str], list[list[str]]]:
    doc_ids, doc_texts, doc_entities = [], [], []
    for line in open(CORPUS, encoding="utf-8"):
        o = json.loads(line)
        t = (o.get("text") or "").strip()
        if not t:
            continue  # 图片/音频/视频无文本，且永不出现在 relevant 里
        doc_ids.append(o["doc_id"])
        doc_texts.append(t)
        doc_entities.append(o.get("entities") or [])
    return doc_ids, doc_texts, doc_entities


def load_entity_names() -> dict[str, str]:
    """canon -> 原始名（取最长的，与 build_queries 的 disp 一致）。"""
    ent: dict[str, str] = {}
    for line in open(CORPUS, encoding="utf-8"):
        o = json.loads(line)
        for c, n in zip(o.get("entities") or [], o.get("entity_names") or []):
            if c not in ent or len(n) > len(ent[c]):
                ent[c] = n
    return ent


# ────────────────────────── 建库 ──────────────────────────
def seed() -> None:
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient

    from app.config import get_settings
    s = get_settings()

    doc_ids, doc_texts, doc_entities = load_corpus()
    entities = load_entity_names()
    canon_list = list(entities.keys())
    name_list = [entities[c] for c in canon_list]

    print(f"[seed] 文件={len(doc_ids)} 实体={len(canon_list)}", file=sys.stderr)
    doc_emb = embed_cached(doc_texts, "doc", CACHE_DOC)
    ent_emb = embed_cached(name_list, "query", CACHE_ENT)

    client = MilvusClient(uri=s.milvus_uri)
    for col in (s.milvus_chunk_collection, s.milvus_entity_collection):
        try:
            client.delete(collection_name=col, filter=f'kb_id == "{KB}"')
        except Exception as e:
            print(f"[seed] delete {col} kb_eval failed: {e}", file=sys.stderr)

    # 灌 chunk（一个文件一个 chunk，parent_id = 文件路径，直接对齐评测集 doc_id）
    rows = []
    for did, text, ents, emb in zip(doc_ids, doc_texts, doc_entities, doc_emb):
        rows.append({"child_id": "eval_" + did, "parent_id": did, "content": text[:4000],
                     "kb_id": KB, "section": "", "span_start": 0, "span_end": len(text),
                     "content_version": 1, "kg_entity_ids": ents,
                     "embedding": emb["dense"],
                     "text_sparse": emb["sparse"] or {}})
    client.insert(collection_name=s.milvus_chunk_collection, data=rows)

    erows = [{"entity_id": "ent_" + c, "name": n, "type": "Equipment", "kb_id": KB,
              "embedding": e["dense"]}
             for c, n, e in zip(canon_list, name_list, ent_emb)]
    client.insert(collection_name=s.milvus_entity_collection, data=erows)
    client.flush(s.milvus_chunk_collection)
    client.flush(s.milvus_entity_collection)
    print(f"[seed] milvus ok: chunks={len(rows)} entities={len(erows)}", file=sys.stderr)

    # Neo4j：Entity + Chunk + BINDS（文件-实体绑定；无实体间关系，符合评测集语义）
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        with driver.session() as sess:
            sess.run("MATCH (n {kb_id: $kb}) DETACH DELETE n", kb=KB)
            sess.run("UNWIND $rows AS r MERGE (e:Entity {entity_id: r.entity_id}) "
                     "SET e.name=r.name, e.type=r.type, e.kb_id=r.kb_id",
                     rows=[{"entity_id": "ent_" + c, "name": n, "type": "Equipment",
                            "kb_id": KB} for c, n in zip(canon_list, name_list)])
            sess.run("UNWIND $rows AS r MERGE (c:Chunk {chunk_id: r.chunk_id}) "
                     "SET c.parent_id=r.parent_id, c.content=r.content, c.kb_id=r.kb_id, "
                     "    c.span_start=r.span_start, c.span_end=r.span_end",
                     rows=[{"chunk_id": "eval_" + did, "parent_id": did,
                            "content": text[:4000], "kb_id": KB,
                            "span_start": 0, "span_end": len(text)}
                           for did, text in zip(doc_ids, doc_texts)])
            binds = [{"chunk_id": "eval_" + did, "entity_id": "ent_" + c}
                     for did, ents in zip(doc_ids, doc_entities) for c in ents]
            sess.run("UNWIND $rows AS r MATCH (c:Chunk {chunk_id: r.chunk_id}) "
                     "MATCH (e:Entity {entity_id: r.entity_id}) MERGE (c)-[:BINDS]->(e)",
                     rows=binds)
            print(f"[seed] neo4j ok: entities={len(canon_list)} chunks={len(doc_ids)} "
                  f"binds={len(binds)}", file=sys.stderr)
    finally:
        driver.close()


# ────────────────────────── 完整链路检索 ──────────────────────────
def _write_results(path: Path, results: dict[str, list[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_ids in results.items():
            f.write(json.dumps({"qid": qid, "doc_ids": doc_ids}, ensure_ascii=False) + "\n")
    print(f"[out] {path} ({len(results)} queries)", file=sys.stderr)


async def run() -> None:
    from app.degrade.orchestrator import DegradedCollector
    from app.fuse.rerank import rerank
    from app.fuse.rrf import rrf_fuse
    from app.recall import graph_route, vector_route

    queries = []
    for line in open(QUERIES, encoding="utf-8"):
        o = json.loads(line)
        queries.append((o["qid"], o["query"], o["relevant"]))

    recall_res: dict[str, list[str]] = {}
    rerank_res: dict[str, list[str]] = {}
    vec_n, graph_n = [], []

    t0 = time.time()
    for qi, (qid, qtext, _qrel) in enumerate(queries):
        collector = DegradedCollector()
        vec_hits, graph_hits = await asyncio.gather(
            vector_route.recall(qtext, scope=SCOPE, collector=collector, conditions=None),
            graph_route.recall(qtext, scope=SCOPE, collector=collector),
        )
        fused = rrf_fuse({"vector_hybrid": vec_hits, "graph_local": graph_hits},
                         k=60, top_k=50)
        top = await rerank(qtext, fused, collector)
        recall_res[qid] = [h.parent_id for h in fused]
        rerank_res[qid] = [h.parent_id for h in top]
        vec_n.append(len(vec_hits))
        graph_n.append(len(graph_hits))
        if (qi + 1) % 10 == 0:
            print(f"[run] {qi+1}/{len(queries)}  elapsed={time.time()-t0:.0f}s",
                  file=sys.stderr)

    _write_results(EVAL_DIR / "results_full_recall.jsonl", recall_res)
    _write_results(EVAL_DIR / "results_full_rerank.jsonl", rerank_res)
    print(f"[run] done {len(queries)} queries in {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"[run] vec hits avg={np.mean(vec_n):.1f}  graph hits avg={np.mean(graph_n):.1f} "
          f"(graph=0 说明评测集无实体间关系，图扩展无贡献，预期内)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.seed and not args.run:
        args.seed = args.run = True
    if args.seed:
        seed()
    if args.run:
        asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
