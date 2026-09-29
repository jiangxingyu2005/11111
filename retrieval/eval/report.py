"""汇总 + 报告渲染：report.json / report.md / report.html（自包含，可双击打开）。"""
from __future__ import annotations

import html as _html
import json
import time
from pathlib import Path

from .metrics import BUDGET_S, agg, latency_stats, metrics

# 展示用指标（按偏好序，过滤出 summary 里实际存在的）
_PREF = ["crecall@5", "crecall@10", "crecall@20", "recall@5", "recall@10", "recall@20",
         "cprec@5", "cprec@10", "cprec@20", "mrr", "ndcg@10"]


def _final_ids(r: dict) -> list[str]:
    return [h["child_id"] for h in r["final"]]


def summarize(results: list[dict], ks: list[int], budget_s: float = BUDGET_S) -> dict:
    """test/val 或分类型的一组结果 → {final, fused, latency, n_q}。"""
    if not results:
        return {"final": {}, "fused": None, "latency": latency_stats([], budget_s), "n_q": 0}
    final_rows = [metrics(_final_ids(r), r["gold"], ks) for r in results]
    has_fused = any(r["fused"] for r in results)
    fused_rows = [metrics(r["fused"], r["gold"], ks) for r in results if r["fused"]] if has_fused else []
    return {
        "final": agg(final_rows),
        "fused": agg(fused_rows) if fused_rows else None,
        "latency": latency_stats([r["lat_ms"] for r in results], budget_s),
        "n_q": len(results),
    }


def _hit_rate(results: list[dict], k: int = 5) -> float:
    """命中率：top-k 命中 ≥1 条核心 gold 的 query 占比。"""
    if not results:
        return 0.0
    n = 0
    for r in results:
        core = {c for c, g in r["gold"].items() if g >= 2}
        top = [h["child_id"] for h in r["final"][:k]]
        if core and set(top) & core:
            n += 1
    return n / len(results)


def _by_type(results: list[dict], ks: list[int], budget_s: float = BUDGET_S) -> dict:
    d: dict[str, list[dict]] = {}
    for r in results:
        d.setdefault(r["type"], []).append(r)
    out = {}
    for t, rows in sorted(d.items()):
        s = summarize(rows, ks, budget_s)
        s["hit_rate@5"] = _hit_rate(rows, 5)
        out[t] = s
    return out


def _show_keys(summ: dict) -> list[str]:
    avail = set(summ.get("final", {}).keys())
    return [k for k in _PREF if k in avail]


def _fmt(m: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)


def _fmt_lat(lat: dict) -> str:
    return (f"mean={lat['mean_ms']:.1f}ms p50={lat['p50_ms']:.1f}ms p90={lat['p90_ms']:.1f}ms "
            f"p95={lat['p95_ms']:.1f}ms p99={lat['p99_ms']:.1f}ms max={lat['max_ms']:.1f}ms "
            f"超预算={lat['over_budget']}/{lat['n']}")


def write_report(out_dir: Path, test_results: list[dict], val_results: list[dict],
                 ks: list[int], backend: str, seed_dir, n_chunks: int,
                 max_per_type: int, mode: str, budget_s: float = BUDGET_S) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_test = summarize(test_results, ks, budget_s)
    res_val = summarize(val_results, ks, budget_s)
    by_type = _by_type(test_results + val_results, ks, budget_s)
    hit = {"test": _hit_rate(test_results), "val": _hit_rate(val_results)}
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode, "backend": backend, "seed_dir": str(seed_dir),
        "n_chunks": n_chunks, "max_per_type": max_per_type,
        "ks": ks, "budget_s": budget_s,
        "split": {"test": len(test_results), "val": len(val_results),
                  "ratio": "8:2", "stratified_by_type": True},
    }

    # ── JSON ──
    payload = dict(meta)
    payload["test"] = res_test
    payload["val"] = res_val
    payload["by_type"] = by_type
    payload["hit_rate@5"] = hit
    payload["per_query"] = _lean(test_results + val_results)
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Markdown ──
    (out_dir / "report.md").write_text(_render_md(meta, res_test, res_val, by_type, hit, ks), encoding="utf-8")

    # ── HTML ──
    (out_dir / "report.html").write_text(
        _render_html(meta, res_test, res_val, by_type, hit, ks, test_results), encoding="utf-8")

    return out_dir


def _lean(results: list[dict]) -> list[dict]:
    """per-query 明细（final 每条补 grade）。"""
    out = []
    for r in results:
        gold = r["gold"]
        final = [{"child_id": h["child_id"], "parent_id": h.get("parent_id"),
                  "rerank_score": h.get("rerank_score"), "grade": gold.get(h["child_id"], 0),
                  "snippet": h.get("snippet", "")} for h in r["final"]]
        out.append({"q": r["q"], "type": r["type"], "lat_ms": round(r["lat_ms"], 2),
                    "first_token_ms": r.get("first_token_ms"), "degraded": r.get("degraded", []),
                    "gold_n": len(gold), "core_n": sum(1 for g in gold.values() if g >= 2),
                    "fused_n": len(r["fused"]), "final": final})
    return out


# ────────────────────────── Markdown ──────────────────────────
def _render_md(meta, res_test, res_val, by_type, hit, ks) -> str:
    keys = _show_keys(res_test)
    L = []
    L.append("# retrieval 测评报告（召回率 + 响应时间）\n")
    L.append(f"- 模式：{meta['mode']}｜后端：{meta['backend']}｜seed：{meta['seed_dir']}")
    L.append(f"- 生成时间：{meta['generated_at']}｜chunks={meta['n_chunks']}｜预算={meta['budget_s']}s")
    L.append(f"- 划分：test={meta['split']['test']} / val={meta['split']['val']}（8:2，按类型分层）\n")

    for name, r in (("测试集 test", res_test), ("验证集 val", res_val)):
        L.append(f"## {name}（n={r['n_q']}）")
        L.append(f"- 召回率·精排后：{_fmt(r['final'], keys)}")
        if r.get("fused"):
            L.append(f"- 召回率·融合后(RRF)：{_fmt(r['fused'], keys)}")
        L.append(f"- 响应时间·检索侧：{_fmt_lat(r['latency'])}")
        L.append(f"- 命中率@5：{hit['test' if name.startswith('测试') else 'val']:.1%}")
        L.append("")

    L.append("## 分类型明细（test+val，精排后）\n")
    L.append("| 类型 | n | crecall@5 | recall@5 | cprec@5 | mrr | 命中率@5 | mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t, s in by_type.items():
        f = s["final"]
        L.append(f"| {t} | {s['n_q']} | {f.get('crecall@5', 0):.3f} | {f.get('recall@5', 0):.3f} | "
                 f"{f.get('cprec@5', 0):.3f} | {f.get('mrr', 0):.3f} | {s['hit_rate@5']:.1%} | "
                 f"{s['latency']['mean_ms']:.1f}ms |")

    L.append("\n> 口径：响应时间为检索侧（parse→召回→融合→精排），不含 LLM 生成；"
             "11.5s 首 token 验收需接生成后测。crecall=核心召回（gold grade≥2）。")
    return "\n".join(L)


# ────────────────────────── HTML ──────────────────────────
_CSS = """
:root{--ok:#1b8a3a;--part:#c9820a;--miss:#8a8f98;--bg:#f6f7f9;--card:#fff;--line:#e5e7eb}
*{box-sizing:border-box}body{margin:0;font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
background:var(--bg);color:#1f2328}a{color:inherit}
.wrap{max-width:1180px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.meta{color:#666;font-size:13px;margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h3{margin:0 0 8px;font-size:15px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.kpi{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kpi .v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:#666}
.lat{color:#555;font-size:13px;margin-top:8px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{background:#f0f2f5;font-weight:600;font-size:13px}
tr:last-child td{border-bottom:none}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:12px 14px}
.q .t{font-weight:600}
.q .qt{color:#444;margin:2px 0 6px;font-size:13px}
.hit{display:flex;align-items:center;gap:10px;padding:6px 8px;border-radius:7px;margin:4px 0;font-size:13px}
.hit .dot{width:9px;height:9px;border-radius:50%;flex:none}
.hit .sc{font-weight:700;min-width:58px;text-align:right;font-variant-numeric:tabular-nums}
.hit .id{color:#888;font-size:11px;font-family:ui-monospace,monospace}
.hit .sn{color:#555;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.g2{background:#e8f6ec}.g2 .dot{background:var(--ok)}
.g1{background:#fdf3e0}.g1 .dot{background:var(--part)}
.g0{background:#f2f3f5}.g0 .dot{background:var(--miss)}
.badge{font-size:11px;border-radius:4px;padding:1px 6px;font-weight:600}
.b2{background:#d3eed9;color:var(--ok)}.b1{background:#f7e3b8;color:var(--part)}.b0{background:#e2e4e8;color:var(--miss)}
.note{color:#666;font-size:12px;margin-top:20px;border-top:1px solid var(--line);padding-top:10px}
"""


def _esc(s) -> str:
    return _html.escape(str(s if s is not None else ""))


def _card(name: str, r: dict, keys: list[str], hit_rate: float) -> str:
    f = r["final"]
    lat = r["latency"]

    def kpi(label, val):
        return f'<div class="kpi"><div class="v">{val}</div><div class="l">{label}</div></div>'

    kpis = "".join([
        kpi("核心召回 crecall@5", f"{f.get('crecall@5', 0):.3f}"),
        kpi("召回 recall@5", f"{f.get('recall@5', 0):.3f}"),
        kpi("命中率@5", f"{hit_rate:.1%}"),
        kpi("MRR", f"{f.get('mrr', 0):.3f}"),
        kpi("nDCG@10", f"{f.get('ndcg@10', 0):.3f}"),
        kpi("P95 延迟", f"{lat['p95_ms']:.0f}ms"),
    ])
    lat_line = (f"<div class='lat'>响应时间·检索侧：mean={lat['mean_ms']:.1f}ms "
                f"p50={lat['p50_ms']:.1f}ms p90={lat['p90_ms']:.1f}ms p99={lat['p99_ms']:.1f}ms "
                f"max={lat['max_ms']:.1f}ms｜超预算({'>5.2s'})={lat['over_budget']}/{lat['n']}</div>")
    return f'<div class="card"><h3>{_esc(name)}（n={r["n_q"]}）</h3><div class="kpis">{kpis}</div>{lat_line}</div>'


def _type_table(by_type: dict) -> str:
    rows = []
    for t, s in by_type.items():
        f = s["final"]
        rows.append(f"<tr><td>{_esc(t)}</td><td>{s['n_q']}</td>"
                    f"<td>{f.get('crecall@5', 0):.3f}</td><td>{f.get('recall@5', 0):.3f}</td>"
                    f"<td>{f.get('cprec@5', 0):.3f}</td><td>{f.get('mrr', 0):.3f}</td>"
                    f"<td>{s['hit_rate@5']:.1%}</td><td>{s['latency']['mean_ms']:.1f}ms</td></tr>")
    head = ("<tr><th>类型</th><th>n</th><th>crecall@5</th><th>recall@5</th>"
            "<th>cprec@5</th><th>MRR</th><th>命中率@5</th><th>mean</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def _hit_row(h: dict, gold: dict) -> str:
    g = gold.get(h["child_id"], 0)
    cls = "g2" if g == 2 else ("g1" if g == 1 else "g0")
    bcls = "b2" if g == 2 else ("b1" if g == 1 else "b0")
    blabel = "核心命中" if g == 2 else ("相关" if g == 1 else "未命中")
    sc = f"{h['rerank_score']*100:.1f}%" if isinstance(h.get("rerank_score"), (int, float)) else "—"
    return (f'<div class="hit {cls}"><span class="dot"></span>'
            f'<span class="badge {bcls}">{blabel}</span>'
            f'<span class="sc">{_esc(sc)}</span>'
            f'<span class="id">{_esc(h.get("child_id", ""))}</span>'
            f'<span class="sn">{_esc(h.get("snippet", ""))}</span></div>')


def _query_blocks(results: list[dict]) -> str:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    parts = []
    for t in sorted(by_type):
        parts.append(f"<h2>{_esc(t)}（{len(by_type[t])} 条）</h2>")
        for r in by_type[t]:
            rows = "".join(_hit_row(h, r["gold"]) for h in r["final"])
            rows = rows or '<div class="hit g0"><span class="dot"></span><span class="sn">（无命中）</span></div>'
            parts.append(f'<div class="q"><div class="t">{_esc(t)}</div>'
                         f'<div class="qt">{_esc(r["q"])}</div>{rows}</div>')
    return "".join(parts)


def _render_html(meta, res_test, res_val, by_type, hit, ks, test_results) -> str:
    keys = _show_keys(res_test)
    cards = (_card("测试集 test", res_test, keys, hit["test"])
             + _card("验证集 val", res_val, keys, hit["val"]))
    meta_line = (f"<div class='meta'>模式：{_esc(meta['mode'])}｜后端：{_esc(meta['backend'])}｜"
                 f"seed：{_esc(meta['seed_dir'])}｜chunks={meta['n_chunks']}｜"
                 f"生成：{_esc(meta['generated_at'])}｜预算 {meta['budget_s']}s</div>")
    note = ("<div class='note'>口径：响应时间为检索侧（parse→召回→融合→精排），不含 LLM 生成；"
            "11.5s 首 token 验收需接生成后测。crecall=核心召回（gold grade≥2）；"
            "Top5 分数为重排置信度（rerank score），绿=核心命中、黄=相关、灰=未命中。</div>")
    return ("<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>retrieval 测评报告</title><style>{_CSS}</style></head><body><div class='wrap'>"
            f"<h1>retrieval 测评报告（召回率 + 响应时间）</h1>{meta_line}"
            f"<div class='cards'>{cards}</div>"
            f"<h2>分类型明细（test+val）</h2>{_type_table(by_type)}"
            f"<h2>逐 query Top5 命中（测试集）</h2>{_query_blocks(test_results)}"
            f"{note}</div></body></html>")
