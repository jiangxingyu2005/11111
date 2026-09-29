"""把 data/seed/*.json 灌进真三库（PG / Milvus / Neo4j）。

前置：
  docker compose up -d
  python scripts/init_milvus.py [--with-sparse]
  docker compose exec neo4j cypher-shell -u neo4j -p dev_only -f /import/001_constraints.cypher

用法：
  python scripts/seed_real.py                  # 全灌
  python scripts/seed_real.py --only pg,milvus # 只灌指定库
  python scripts/seed_real.py --wipe           # 先清 Neo4j（PG/Milvus 用 upsert 覆盖）

注意：
  - 本脚本是【写侧】，与在线 retrieval 的只读账号无关；离线入库（qingxi）才是生产写路径
  - seed 的向量来自 gen_seed.py 指定的通道；换通道必须重跑 gen_seed.py，否则串空间
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = ROOT / "data" / "seed"


def load(name: str) -> list[dict]:
    p = SEED / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"seed 缺失 {p} —— 先跑 `python scripts/gen_seed.py`")
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────── PG ───────────────────────────
async def seed_pg(dsn: str) -> int:
    import asyncpg

    parents, chunks = load("parents"), load("chunks")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            """INSERT INTO parent_document
               (parent_id, doc_id, title, content, summary, keywords, content_version, kb_id, source_type, toa_text)
               VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10)
               ON CONFLICT (parent_id) DO UPDATE SET content=EXCLUDED.content,
                   summary=EXCLUDED.summary, content_version=EXCLUDED.content_version,
                   updated_at=now()""",
            [(p["parent_id"], p["doc_id"], p.get("title"), p["content"], p.get("summary"),
              json.dumps(p.get("keywords", []), ensure_ascii=False),
              p.get("content_version", 1), p["kb_id"], p.get("source_type"), p.get("toa_text"))
             for p in parents])
        await conn.executemany(
            """INSERT INTO child_chunk
               (child_id, parent_id, kb_id, content, section, span_start, span_end,
                content_version, kg_entity_ids, scalar)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)
               ON CONFLICT (child_id) DO UPDATE SET content=EXCLUDED.content,
                   span_start=EXCLUDED.span_start, span_end=EXCLUDED.span_end""",
            [(c["child_id"], c["parent_id"], c["kb_id"], c["content"], c.get("section"),
              (c.get("span") or [None, None])[0], (c.get("span") or [None, None])[1],
              c.get("content_version", 1), list(c.get("kg_entity_ids", [])),
              json.dumps(c.get("scalar", {}), ensure_ascii=False))
             for c in chunks])
        # 标量抽取表（④ 路预留，当前不参与在线过滤）
        rows = [(c["child_id"], c["parent_id"], (c.get("scalar") or {}).get("rf_ghz"),
                 (c.get("scalar") or {}).get("prf_hz"), (c.get("scalar") or {}).get("pw_us"),
                 (c.get("scalar") or {}).get("doa_deg"), (c.get("scalar") or {}).get("snr_db"))
                for c in chunks]
        await conn.executemany(
            """INSERT INTO scalar_extract (child_id, parent_id, rf_ghz, prf_hz, pw_us, doa_deg, snr_db)
               VALUES ($1,$2,$3,$4,$5,$6,$7)
               ON CONFLICT (child_id) DO UPDATE SET rf_ghz=EXCLUDED.rf_ghz""", rows)
        await conn.executemany(
            """INSERT INTO entity_registry (entity_id, name, type, aliases, kb_id)
               VALUES ($1,$2,$3,$4,$5)
               ON CONFLICT (entity_id) DO UPDATE SET name=EXCLUDED.name""",
            [(e["entity_id"], e["name"], e["type"], list(e.get("aliases", [])), e.get("kb_id"))
             for e in load("entities")])
        n = await conn.fetchval("SELECT count(*) FROM child_chunk")
        print(f"[seed_real] pg ok: parents={len(parents)} chunks={len(chunks)} (child_chunk rows={n})")
        return len(chunks)
    finally:
        await conn.close()


# ─────────────────────────── Milvus ───────────────────────────
def seed_milvus(with_sparse: bool, with_secondary: bool) -> int:
    from pymilvus import MilvusClient

    from app.config import get_settings
    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)
    chunks = load("chunks")
    rows = []
    for c in chunks:
        sp = c.get("span") or [0, 0]
        r = {"child_id": c["child_id"], "parent_id": c["parent_id"],
             "content": c["content"][:4000], "kb_id": c["kb_id"],
             "section": (c.get("section") or "")[:500],
             "span_start": int(sp[0]), "span_end": int(sp[1]),
             "content_version": int(c.get("content_version", 1)),
             "kg_entity_ids": list(c.get("kg_entity_ids", []))[:32],
             "embedding": c["embedding"]}
        if with_secondary and c.get("embedding_v2"):
            r["embedding_v2"] = c["embedding_v2"]
        if with_sparse and c.get("sparse"):
            r["text_sparse"] = {int(k): float(v) for k, v in c["sparse"].items()}
        rows.append(r)
    for i in range(0, len(rows), 200):
        client.insert(collection_name=s.milvus_chunk_collection, data=rows[i:i + 200])

    ents = load("entities")
    erows = [{"entity_id": e["entity_id"], "name": e["name"], "type": e["type"],
              "kb_id": e.get("kb_id", ""), "embedding": e["embedding"]} for e in ents]
    for i in range(0, len(erows), 200):
        client.insert(collection_name=s.milvus_entity_collection, data=erows[i:i + 200])

    client.flush(s.milvus_chunk_collection)
    client.flush(s.milvus_entity_collection)
    print(f"[seed_real] milvus ok: chunks={len(rows)} entities={len(erows)} "
          f"sparse={with_sparse} secondary={with_secondary}")
    return len(rows)


# ─────────────────────────── Neo4j ───────────────────────────
def seed_neo4j(wipe: bool) -> int:
    from neo4j import GraphDatabase

    from app.config import get_settings
    s = get_settings()
    ents, edges, chunks = load("entities"), load("edges"), load("chunks")
    drv = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        with drv.session() as sess:
            if wipe:
                sess.run("MATCH (n) DETACH DELETE n")
            sess.run(
                "UNWIND $rows AS r MERGE (e:Entity {entity_id: r.entity_id}) "
                "SET e.name = r.name, e.type = r.type, e.kb_id = r.kb_id",
                rows=[{"entity_id": e["entity_id"], "name": e["name"],
                       "type": e["type"], "kb_id": e.get("kb_id", "")} for e in ents])
            sess.run(
                "UNWIND $rows AS r MERGE (s:Chunk {chunk_id: r.child_id}) "
                "SET s.parent_id = r.parent_id, s.content = r.content, s.kb_id = r.kb_id, "
                "    s.span_start = r.span_start, s.span_end = r.span_end",
                rows=[{"child_id": c["child_id"], "parent_id": c["parent_id"],
                       "content": c["content"][:4000], "kb_id": c["kb_id"],
                       "span_start": int((c.get("span") or [0, 0])[0]),
                       "span_end": int((c.get("span") or [0, 0])[1])} for c in chunks])
            binds = [e for e in edges if e.get("rel") == "BINDS"]
            sess.run(
                "UNWIND $rows AS r MATCH (e:Entity {entity_id: r.src}) "
                "MATCH (s:Chunk {chunk_id: r.dst}) MERGE (s)-[:BINDS]->(e)",
                rows=[{"src": e["src"], "dst": e["dst"]} for e in binds])
            typed = {}
            for e in edges:
                if e.get("rel") != "BINDS":
                    typed.setdefault(e["rel"], []).append(e)
            for rel, items in typed.items():
                sess.run(
                    f"UNWIND $rows AS r MATCH (a:Entity {{entity_id: r.src}}) "
                    f"MATCH (b:Entity {{entity_id: r.dst}}) "
                    f"MERGE (a)-[x:{rel}]->(b) "
                    f"SET x.confidence = r.confidence, x.evidence_child_id = r.evidence_child_id, "
                    f"    x.span_start = r.span_start, x.span_end = r.span_end, x.parent_id = r.parent_id",
                    rows=[{"src": e["src"], "dst": e["dst"],
                           "confidence": float(e.get("weight", 1.0)),
                           "evidence_child_id": (e.get("evidence") or {}).get("child_id", ""),
                           "span_start": int(((e.get("evidence") or {}).get("span") or [0, 0])[0]),
                           "span_end": int(((e.get("evidence") or {}).get("span") or [0, 0])[1]),
                           "parent_id": (e.get("evidence") or {}).get("parent_id")}
                          for e in items])
        print(f"[seed_real] neo4j ok: entities={len(ents)} segments={len(chunks)} edges={len(edges)}")
        return len(edges)
    finally:
        drv.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="pg,milvus,neo4j")
    ap.add_argument("--with-sparse", action="store_true")
    ap.add_argument("--with-secondary", action="store_true")
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()
    targets = {t.strip() for t in args.only.split(",") if t.strip()}

    from app.config import get_settings
    s = get_settings()
    if "pg" in targets:
        asyncio.run(seed_pg(s.pg_dsn))
    if "milvus" in targets:
        seed_milvus(args.with_sparse, args.with_secondary)
    if "neo4j" in targets:
        seed_neo4j(args.wipe)
    print("[seed_real] all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
