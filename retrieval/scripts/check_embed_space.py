"""向量空间哨兵检查：证明"当前用的是真模型还是哈希占位"。

做什么：
  用 seed 自带的弱监督对（父块 summary → 该父块的子块）算 Recall@5 / MRR@10，
  并做一组"字面不重叠但语义相关"的硬案例。
  真 BGE-M3 应该轻松过线；哈希嵌入只能靠字面重叠，会明显低一截。

为什么需要它：
  "以为换了 BGE-M3 其实还在跑 hash"是本项目最危险的静默漂移——
  链路全绿、召回全是噪声，且没有任何报错。必须有可执行的是否断言。

用法：
  python scripts/check_embed_space.py              # dev 通道：只报告，不判失败
  python scripts/check_embed_space.py --expect real # 真模型：不过线退出码非 0

纪律：
  这是【弱监督哨兵】，不是评测结论。summary 与子块天然共享词，
  哈希嵌入也能拿到一些分——所以 dev 模式判 FAIL 没有意义，只做基线记录。
  真正的召回率结论只能来自人工标注的 golden 集（E5）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = ROOT / "data" / "seed"

# 真模型通道的验收线（E1 完成后必须达到）
REAL_GATES = {"recall@5": 0.80, "mrr@10": 0.60, "top1_cosine": 0.70}

# 字面不重叠的硬案例：query 里的词尽量不在正例文本中出现
HARD_CASES = [
    {"id": "band-alias",
     "query": "工作在公分波段的火控雷达",
     "pos_kw": ["GHz", "波段"],
     "neg_kw": ["信噪比", "仰角"]},
    {"id": "waveform",
     "query": "脉内调制为相位编码的辐射源",
     "pos_kw": ["PSK", "相位编码"],
     "neg_kw": ["驱逐舰", "护卫舰"]},
]


def load(name: str) -> list[dict]:
    p = SEED / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"seed 缺失：{p} —— 先跑 python scripts/gen_seed.py")
    return json.loads(p.read_text(encoding="utf-8"))


def embed(texts: list[str], mode: str, url: str) -> list[list[float]]:
    if mode == "service":
        import httpx
        r = httpx.post(url, json={"texts": texts, "type": "doc"}, timeout=300.0)
        r.raise_for_status()
        out = r.json()["embeddings"]
        return [e["dense"] if isinstance(e, dict) else e for e in out]
    from app.store.hash_embed import hash_embedding
    return [hash_embedding(t) for t in texts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", choices=["hash", "service"], default="hash",
                    help="hash=本地哈希占位；service=调 RETRIEVAL_EMBED_URL 真模型")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8001/embed")
    ap.add_argument("--expect", choices=["dev", "real"], default="dev")
    ap.add_argument("--n", type=int, default=40, help="抽样父块数")
    args = ap.parse_args()

    parents, chunks = load("parents"), load("chunks")
    meta = json.loads((SEED / "meta.json").read_text(encoding="utf-8"))
    by_parent: dict[str, list[dict]] = {}
    for c in chunks:
        by_parent.setdefault(c["parent_id"], []).append(c)

    rng = random.Random(7)
    sample = rng.sample([p for p in parents if by_parent.get(p["parent_id"])],
                        min(args.n, len(parents)))

    queries = [p.get("summary") or p["title"] for p in sample]
    gold = [p["parent_id"] for p in sample]
    pool = list(chunks)

    print(f"[check] embed={args.embed}  seed_dim={meta.get('dim')}  "
          f"seed_channel={meta.get('embed_channel')}  n={len(sample)}")

    # 先把全部候选编码一次（130 条级别，真模型 CPU 也就一两分钟）
    corpus_vec = embed([c["content"] for c in pool], args.embed, args.embed_url)
    dim = len(corpus_vec[0])

    hits5 = hits = 0
    rr_sum = top1_cos = 0.0
    for q, gid in zip(queries, gold):
        qv = embed([q], args.embed, args.embed_url)[0]
        scored = sorted(
            ((sum(a * b for a, b in zip(qv, cv)), c) for cv, c in zip(corpus_vec, pool)),
            key=lambda x: x[0], reverse=True)
        top1_cos += scored[0][0]
        ranks = [i for i, (_s, c) in enumerate(scored[:10], 1) if c["parent_id"] == gid]
        if ranks:
            hits += 1
            rr_sum += 1.0 / ranks[0]
            if ranks[0] <= 5:
                hits5 += 1

    r5 = hits5 / len(sample)
    mrr = rr_sum / len(sample)
    t1 = top1_cos / len(sample)
    print(f"[check] dim={dim}")
    print(f"[check] Recall@5 = {r5:.3f}   MRR@10 = {mrr:.3f}   Top1余弦 = {t1:.3f}")

    # 硬案例：语义相关但字面不重叠
    print("[check] 硬案例（字面不重叠，考语义能力）：")
    for case in HARD_CASES:
        qv = embed([case["query"]], args.embed, args.embed_url)[0]
        sims = [(sum(a * b for a, b in zip(qv, cv)), c) for cv, c in zip(corpus_vec, pool)]
        sims.sort(key=lambda x: x[0], reverse=True)
        pos = [s for s, c in sims if any(k in c["content"] for k in case["pos_kw"])]
        top_pos = pos[0] if pos else 0.0
        hit_neg = any(k in sims[0][1]["content"] for k in case["neg_kw"])
        flag = "正例" if any(k in sims[0][1]["content"] for k in case["pos_kw"]) \
            else ("负例" if hit_neg else "无关")
        print(f"    {case['id']:<12} 命中正例最高余弦 = {top_pos:.3f}  "
              f"Top1={flag:<3}  文本: {sims[0][1]['content'][:34]}…")

    if args.expect == "real":
        fails = []
        if r5 < REAL_GATES["recall@5"]:
            fails.append(f"Recall@5={r5:.3f} < {REAL_GATES['recall@5']}")
        if mrr < REAL_GATES["mrr@10"]:
            fails.append(f"MRR@10={mrr:.3f} < {REAL_GATES['mrr@10']}")
        if t1 < REAL_GATES["top1_cosine"]:
            fails.append(f"Top1余弦={t1:.3f} < {REAL_GATES['top1_cosine']}（哈希量纲≈0.43，换模型必升）")
        if dim != 1024:
            fails.append(f"dim={dim} != 1024（BGE-M3 dense 必须 1024）")
        if meta.get("embed_channel") != "service":
            fails.append(f"seed 仍是 {meta.get('embed_channel')} 通道建的库——换模型必须重跑 gen_seed.py")
        if fails:
            print("[FAIL] " + "；".join(fails))
            return 1
        print("[OK] 真模型通道验收通过")
    else:
        print("[dev] 仅记录基线：哈希/占位通道无语义能力，禁止据此评估召回率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
