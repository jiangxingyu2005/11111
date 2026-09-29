"""CLI 入口：python -m eval ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.golden import build_eval_set, load_seed, split_82  # noqa: E402
from eval.report import write_report  # noqa: E402
from eval.runner import run_http, run_in_process  # noqa: E402

DEFAULT_SEED = ROOT / "data" / "seed_date"
DEFAULT_OUT = ROOT / "data" / "eval"


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m eval",
                                 description="retrieval 测评模块：召回率 + 响应时间")
    ap.add_argument("--mode", choices=["in_process", "http"], default="in_process",
                    help="in_process=进程内直连（默认）；http=端到端回放（需先起 retrieval 服务）")
    ap.add_argument("--target-url", default="http://127.0.0.1:8100", help="http 模式被测服务地址")
    ap.add_argument("--seed-dir", default=str(DEFAULT_SEED), help="seed 目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="报告输出目录")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--max-per-type", type=int, default=200)
    ap.add_argument("--scope", nargs="+", default=["kb_public"])
    ap.add_argument("--no-warm", action="store_true", help="跳过预热（仅 in_process 生效）")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir)
    chunks, entities, records = load_seed(seed_dir)
    evalset = build_eval_set(records, chunks, args.max_per_type)
    test, val = split_82(evalset)
    combined = test + val

    print(f"[eval] chunks={len(chunks)} entities={len(entities)} records={len(records)}")
    print(f"[eval] 评测集 queries={len(evalset)} -> test={len(test)} / val={len(val)}（8:2，按类型分层）")
    for t in sorted({q["type"] for q in evalset}):
        st = sum(1 for q in test if q["type"] == t)
        sv = sum(1 for q in val if q["type"] == t)
        print(f"       {t}: 共 {st + sv}（test {st} / val {sv}）")

    if args.mode == "in_process":
        results = run_in_process(combined, args.scope, seed_dir, warm=not args.no_warm)
        backend = "fake(hash embed)"
    else:
        print(f"[eval] http 回放 -> {args.target_url}（需服务已启动）", flush=True)
        results = run_http(combined, args.scope, args.target_url)
        backend = f"http {args.target_url}"

    # results 与 combined 顺序一一对应（test 在前、val 在后）
    test_results = results[:len(test)]
    val_results = results[len(test):]

    out = write_report(Path(args.out_dir), test_results, val_results, args.ks, backend,
                       seed_dir, len(chunks), args.max_per_type, args.mode)
    print(f"\n[eval] 完成：{len(results)} 条 query，报告 -> {out}（report.json / report.md / report.html）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
