"""生成种子数据集（fake 后端 / 真库共用同一份）。

产出 data/seed/：
  parents.json   父块全文（PG parent_document 同源）
  chunks.json    子块 + 向量 + 稀疏 + span（Milvus ems_chunks 同源）
  entities.json  实体（Milvus ems_entities 同源）
  edges.json     实体边 + BINDS 绑定边（Neo4j 同源）
  rules.json     预设推理规则种子（fake 后端规则引擎数据源，与真库 rules 表同构）
  meta.json      生成指纹（dim / embed 通道 / 条数）——用于校验"seed 与在线同空间"

用法：
  python scripts/gen_seed.py                       # 本地 hash 嵌入（默认，无需模型）
  python scripts/gen_seed.py --embed service       # 调 RETRIEVAL_EMBED_URL 真模型建库（BGE-M3）
  python scripts/gen_seed.py --docs 40 --seed 7    # 调规模 / 随机种子

纪律：
  - 合成数据只验证【逻辑与链路】不验证【召回率】（V3.0 §12）：语义能力有限，禁止当评测结论
  - 换嵌入通道必须重跑本脚本，否则 query 与 doc 不在同一空间，检索结果无意义
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.store.hash_embed import hash_embedding, hash_numeric_embedding, hash_sparse  # noqa: E402

OUT = ROOT / "data" / "seed"

# ─────────────────────────────────────────────────────────────
# 领域语料（电磁信号侦察）模板
# ─────────────────────────────────────────────────────────────
PLATFORMS = ["驱逐舰", "护卫舰", "预警机", "车载机动站", "地面固定站", "无人机平台"]
FUNCTIONS = ["搜索雷达", "跟踪雷达", "火控雷达", "导航雷达", "测高雷达"]
BANDS = {"S波段": (2.0, 4.0), "C波段": (4.0, 8.0), "X波段": (8.0, 12.0), "Ku波段": (12.0, 18.0)}
MODS = ["线性调频LFM", "非线性调频NLFM", "相位编码PSK", "脉冲多普勒PD", "频率捷变", "重频抖动"]
AREAS = ["东南海域A", "东南海域B", "西部高原C", "北部边境D", "近岸训练区E"]
COUNTRIES = ["A国", "B国", "C国"]
RADAR_SERIES = ["AN/SPY-1D", "AN/SPS-49", "AN/APG-68", "某型三坐标雷达", "某型相控阵雷达",
                "某型舰载预警雷达", "某型车载火控雷达", "某型地面警戒雷达"]

ENTITY_REGISTRY: list[tuple[str, str, list[str]]] = []
for _r in RADAR_SERIES:
    ENTITY_REGISTRY.append((_r, "Radar", []))
for _p in PLATFORMS:
    ENTITY_REGISTRY.append((_p, "Platform", []))
for _f in FUNCTIONS:
    ENTITY_REGISTRY.append((_f, "Function", []))
for _b in BANDS:
    ENTITY_REGISTRY.append((_b, "Band", []))
for _m in MODS:
    ENTITY_REGISTRY.append((_m, "Modulation", []))
for _a in AREAS:
    ENTITY_REGISTRY.append((_a, "Area", []))
for _c in COUNTRIES:
    ENTITY_REGISTRY.append((_c, "Country", []))

# ─────────────────────────────────────────────────────────────
# 预设推理规则种子（fake 后端 rules.json；条件 DSL 见 docs/api/10-rules.md）
# ─────────────────────────────────────────────────────────────
SEED_RULES = [
    {
        "id": "R_001",
        "name": "火控雷达疑似优先",
        "description": "正文命中火控雷达或 KG 置信度不足时标记疑似优先",
        "condition": {
            "all": [],
            "any": [
                {"field": "content", "op": "contains", "value": "火控雷达"},
                {"field": "kg_confidence", "op": "lt", "value": 0.7},
            ],
        },
        "action": {"type": "mark", "label": "疑似优先", "severity": "high",
                   "hint": "建议优先核查该辐射源"},
        "priority": 10,
        "kb_id": None,
    },
    {
        "id": "R_002",
        "name": "低重排分复核",
        "description": "重排分低于 0.5 的证据提示复核（验证 rerank_score 数值算子）",
        "condition": {"all": [{"field": "rerank_score", "op": "lt", "value": 0.5}]},
        "action": {"type": "mark", "label": "低分复核", "severity": "warning",
                   "hint": "该证据与问题的匹配度偏低"},
        "priority": 5,
        "kb_id": None,
    },
    {
        "id": "R_003",
        "name": "域内火控信号",
        "description": "kb_public 且章节含火控时标记（验证 kb_id + section 子句）",
        "condition": {
            "all": [
                {"field": "kb_id", "op": "eq", "value": "kb_public"},
                {"field": "section", "op": "contains", "value": "火控"},
            ],
        },
        "action": {"type": "mark", "label": "域内火控", "severity": "info", "hint": ""},
        "priority": 1,
        "kb_id": None,
    },
    {
        "id": "R_004",
        "name": "融合票分存在",
        "description": "RRF 融合票分 > 0（任何融合命中都满足，验证 rrf_score 数值算子与端到端触发）",
        "condition": {"all": [{"field": "rrf_score", "op": "gt", "value": 0}]},
        "action": {"type": "mark", "label": "已融合", "severity": "info", "hint": ""},
        "priority": 0,
        "kb_id": None,
    },
]


def _ent_id(name: str) -> str:
    return "ent_" + str(abs(hash(name)) % (10 ** 8)).zfill(8)


def build_docs(n_docs: int, rng: random.Random) -> list[dict]:
    docs = []
    for i in range(n_docs):
        radar = rng.choice(RADAR_SERIES)
        plat = rng.choice(PLATFORMS)
        func = rng.choice(FUNCTIONS)
        band = rng.choice(list(BANDS))
        mod = rng.choice(MODS)
        area = rng.choice(AREAS)
        country = rng.choice(COUNTRIES)
        lo, hi = BANDS[band]
        rf = round(rng.uniform(lo + 0.3, hi - 0.3), 2)
        prf = rng.choice([500, 800, 1000, 1200, 1500, 2000, 3000, 5000])
        pw = round(rng.uniform(0.3, 3.5), 2)
        doa = rng.randint(0, 359)
        snr = rng.randint(6, 32)
        ymd = f"2025年{rng.randint(1,12)}月{rng.randint(1,28)}日"
        kb = "kb_public" if i % 2 == 0 else "kb_internal"

        title = f"{area}{plat}载{radar}{func}信号侦察记录（{ymd}）"
        paras = [
            f"{ymd}，侦察站于{area}截获一批{country}辐射源信号，载台判定为{plat}，"
            f"辐射源型号初判为{radar}，用途归类为{func}。该批次信号持续约{rng.randint(5,90)}分钟，"
            f"期间共截获{rng.randint(20,300)}个脉冲，信号稳定无明显漂移。",

            f"技术参数测量：载频{rf}GHz，工作于{band}（{lo}~{hi}GHz）；"
            f"脉冲重复频率PRF {prf}Hz，重频类型{rng.choice(['固定','参差','抖动'])}；"
            f"脉宽PW {pw}us；脉内调制方式为{mod}，"
            f"调制带宽{rng.randint(1,20)}MHz，调制周期{rng.randint(5,60)}us。",

            f"测向与信号质量：来波方位DOA {doa}度，仰角{rng.randint(1,45)}度，"
            f"信噪比SNR {snr}dB，信号强度{rng.randint(-90,-40)}dBm。"
            f"天线扫描周期{rng.randint(2,12)}秒，扫描方式{rng.choice(['圆周扫描','扇扫','边扫描边跟踪'])}，"
            f"主瓣宽度约{rng.uniform(1.0,4.0):.1f}度。",

            f"特征比对：将该信号参数与{radar}条目比对，载频、脉宽、调制三项吻合度"
            f"{rng.randint(75,98)}%，方位与{area}历史活动规律一致。综合判定为{country}{plat}搭载的"
            f"{func}主瓣照射，威胁等级{rng.choice(['低','中','中高','高'])}。",

            f"处置建议：① 将本次{rf}GHz、PRF {prf}Hz、{mod}特征入库，更新{radar}参数包；"
            f"② 对{area}方向保持持续监视，重点关注{func}开机规律；"
            f"③ 该{plat}平台通常与{mod}体制配合出现，建议联动核查同批次其他辐射源。",
        ]
        if rng.random() < 0.5:
            paras.append(
                f"补充说明：本次截获期间同时发现{mod}与{rng.choice(MODS)}交替出现的现象，"
                f"疑似{radar}具备波形捷变能力，需在后续{band}监测中重点甄别。"
            )

        docs.append({
            "doc_id": f"doc_{i:04d}",
            "kb_id": kb,
            "title": title,
            "paras": paras,
            "radar": radar, "platform": plat, "function": func,
            "band": band, "mod": mod, "area": area, "country": country,
            "rf_ghz": rf, "prf_hz": prf, "pw_us": pw, "doa_deg": doa, "snr_db": snr,
            "toa": ymd,
        })
    return docs


def embed_fn_factory(mode: str, url: str, dim: int):
    if mode == "service":
        import httpx
        def _fn(texts: list[str]) -> tuple[list[list[float]], list[dict]]:
            r = httpx.post(url, json={"texts": texts, "type": "doc"}, timeout=1200.0)
            r.raise_for_status()
            embs = r.json()["embeddings"]
            dense = [e["dense"] if isinstance(e, dict) else e for e in embs]
            sparse = [(e.get("sparse") or {}) if isinstance(e, dict) else {} for e in embs]
            return dense, sparse
        return _fn

    def _fn(texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        return [hash_embedding(t, dim) for t in texts], [hash_sparse(t) for t in texts]
    return _fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=24, help="父文档数（每个 4~6 子块）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed", choices=["hash", "service"], default="hash")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8001/embed")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = build_docs(args.docs, rng)
    embed = embed_fn_factory(args.embed, args.embed_url, args.dim)

    parents, chunks, binds = [], [], []
    doc_chunks_all: dict[str, list] = {}   # parent_id -> [(child_id, span_s, span_e, text)]
    child_i = 0
    cursor = 0
    texts: list[str] = []

    for di, d in enumerate(docs):
        pid = f"p_{d['doc_id']}"
        body_parts, spans = [], []
        for para in d["paras"]:
            start, end = cursor, cursor + len(para)
            spans.append((start, end))
            body_parts.append(para)
            cursor = end + 1
        content = "\n".join(body_parts)
        parents.append({
            "parent_id": pid, "doc_id": d["doc_id"], "title": d["title"],
            "content": content,
            "summary": f"{d['area']}{d['platform']}载{d['radar']}，载频{d['rf_ghz']}GHz（{d['band']}），"
                       f"PRF {d['prf_hz']}Hz，脉宽{d['pw_us']}us，{d['mod']}，DOA {d['doa_deg']}度。",
            "keywords": [d["radar"], d["platform"], d["band"], d["mod"], d["function"]],
            "content_version": 1, "kb_id": d["kb_id"], "source_type": "recon_record",
            "toa_text": d["toa"],
            # KG 实体抽取置信度（入库时自设置的置信度表，存 PG parent_document.kg_confidence）。
            # fake seed 用确定性渐变值 0.5~0.96，供规则引擎的 kg_confidence 子句验证触发/不触发两态。
            "kg_confidence": round(0.5 + 0.02 * di, 2),
        })
        # 子块（段落）与父块内的字符区间 span（G-API-02：每条边必须有 evidence，span 是硬要求）
        doc_chunks: list[tuple[str, int, int, str]] = []
        for (para, (s, e)) in zip(d["paras"], spans):
            cid = f"c_{child_i:05d}"
            ents = [n for n, _t, _a in ENTITY_REGISTRY if n and n in para]
            chunks.append({
                "child_id": cid, "parent_id": pid, "content": para,
                "kb_id": d["kb_id"], "content_version": 1,
                "kg_entity_ids": [_ent_id(x) for x in ents],
                "section": d["title"], "span": [s, e],
                "scalar": {"rf_ghz": d["rf_ghz"], "prf_hz": d["prf_hz"],
                           "pw_us": d["pw_us"], "doa_deg": d["doa_deg"], "snr_db": d["snr_db"]},
                "embedding_v2": hash_numeric_embedding({
                    "rf": d["rf_ghz"], "pw": d["pw_us"], "prf": d["prf_hz"],
                    "doa": d["doa_deg"], "snr": d["snr_db"]}),
            })
            for x in ents:
                binds.append({"src": _ent_id(x), "dst": cid, "rel": "BINDS",
                              "weight": 1.0, "kb_id": d["kb_id"], "source_parent_id": pid,
                              "evidence": {"child_id": cid, "span": [s, e], "parent_id": pid}})
            doc_chunks.append((cid, s, e, para))
            texts.append(para)
            child_i += 1
        doc_chunks_all[pid] = doc_chunks

    # 实体向量（name + type + 共现描述）
    ent_texts = [f"{n} {t}" for n, t, _a in ENTITY_REGISTRY]
    all_texts = texts + ent_texts
    dense_all, sparse_all = embed(all_texts)
    dense_chunks, dense_ents = dense_all[:len(texts)], dense_all[len(texts):]
    sparse_chunks = sparse_all[:len(texts)]

    for c, dv, sv in zip(chunks, dense_chunks, sparse_chunks):
        c["embedding"] = dv
        c["sparse"] = sv

    entities = []
    for (name, typ, aliases), dv in zip(ENTITY_REGISTRY, dense_ents):
        entities.append({"entity_id": _ent_id(name), "name": name, "type": typ,
                         "aliases": aliases, "kb_id": "kb_public",
                         "embedding": dv,
                         "desc": f"{name}（{typ}）"})

    # 实体-实体边：结构化关系 + 同段共现
    edges = list(binds)
    eid = {n: _ent_id(n) for n, _t, _a in ENTITY_REGISTRY}
    for d in docs:
        for src, dst, rel in (
            (d["radar"], d["platform"], "MOUNTED_ON"),
            (d["radar"], d["mod"], "USES_MODULATION"),
            (d["radar"], d["band"], "OPERATES_IN"),
            (d["radar"], d["function"], "HAS_FUNCTION"),
            (d["radar"], d["country"], "BELONGS_TO"),
            (d["platform"], d["area"], "ACTIVE_IN"),
        ):
            # 证据锚点：优先取同时提到两端实体的子块，否则退到该文档首个子块
            ev = None
            for cid, s, e, para in doc_chunks_all[f"p_{d['doc_id']}"]:
                if src in para and dst in para:
                    ev = {"child_id": cid, "span": [s, e], "parent_id": f"p_{d['doc_id']}"}
                    break
            if ev is None:
                cs = doc_chunks_all[f"p_{d['doc_id']}"][0]
                ev = {"child_id": cs[0], "span": [cs[1], cs[2]], "parent_id": f"p_{d['doc_id']}"}
            edges.append({"src": eid[src], "dst": eid[dst], "rel": rel,
                          "weight": 1.0, "kb_id": d["kb_id"],
                          "source_parent_id": f"p_{d['doc_id']}", "evidence": ev})

    # 社区摘要（③-B Scene KNN 的 scene_vector 来源）：按波段分组，成员=波段实体 + 代表雷达实体
    communities = []
    for i, band in enumerate(BANDS):
        lo, hi = BANDS[band]
        members = [_ent_id(band)] + [_ent_id(r) for r in RADAR_SERIES[:3]]
        summary = (f"{band}（{lo}~{hi}GHz）雷达信号群：涵盖{RADAR_SERIES[0]}、{RADAR_SERIES[1]}、"
                   f"{RADAR_SERIES[2]}等辐射源，主要活动于{AREAS[0]}、{AREAS[1]}方向，"
                   f"载频、脉宽、调制参数分布见各信号侦察记录。")
        communities.append({
            "community_id": f"comm_{i:04d}", "summary": summary,
            "member_entity_ids": members, "kb_id": "kb_public", "stale": False,
        })
    comm_dense, _ = embed([c["summary"] for c in communities])
    for c, dv in zip(communities, comm_dense):
        c["scene_vector"] = dv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("parents", parents), ("chunks", chunks),
                      ("entities", entities), ("edges", edges),
                      ("communities", communities)):
        (out / f"{name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "rules.json").write_text(
        json.dumps(SEED_RULES, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embed_channel": args.embed,
        "dim": args.dim,
        "docs": len(parents), "chunks": len(chunks),
        "entities": len(entities), "edges": len(edges),
        "communities": len(communities),
        "seed": args.seed,
        "note": "换嵌入通道必须重跑本脚本；合成数据只验逻辑不验召回率",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gen_seed] out={out}")
    print(f"[gen_seed] parents={len(parents)} chunks={len(chunks)} "
          f"entities={len(entities)} edges={len(edges)} communities={len(communities)} "
          f"embed={args.embed} dim={args.dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
