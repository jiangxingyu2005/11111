"""打分模块的独立验证入口。

用法:
    python scripts/score_demo.py output/<doc_id>/clean.md
"""

import sys
from pathlib import Path

# Windows 控制台默认 GBK,强制 UTF-8 避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.utils.scoring import score_markdown  # noqa: E402


def main(path: str) -> None:
    md_path = Path(path)
    if not md_path.exists():
        print(f"文件不存在: {md_path}")
        sys.exit(1)

    results = score_markdown(
        md_path.read_text(encoding="utf-8"),
        min_chunk_len=settings.score_min_chunk_len,
        max_chunk_len=settings.score_max_chunk_len,
        w_len=settings.score_w_len,
        w_info=settings.score_w_info,
        w_content=settings.score_w_content,
        w_hits=settings.score_w_hits,
    )

    q_keep = settings.score_q_keep
    q_quarantine = settings.score_q_quarantine

    print(f"文件: {md_path}")
    print(f"总块数: {len(results)}")
    print("=" * 72)

    for i, r in enumerate(results, 1):
        if r.get("skipped"):
            print(f"[{i:>3}] SKIP(图片/base64, {r.get('length', 0)} 字符)")
            continue
        hits = ",".join(r.get("hits_list", [])[:5])
        approx = " (content近似)" if r.get("content_approx") else ""
        print(
            f"[{i:>3}] q={r['q']:.3f}  "
            f"len={r['f_len']:.2f} info={r['info']:.2f} "
            f"content={r['content']:.2f}{approx} hits={r['hits']}  | {hits}"
        )

    print("=" * 72)
    keep = sum(1 for r in results if not r.get("skipped") and r["q"] >= q_keep)
    mid = sum(1 for r in results if not r.get("skipped") and q_quarantine <= r["q"] < q_keep)
    low = sum(1 for r in results if not r.get("skipped") and r["q"] < q_quarantine)
    skipped = sum(1 for r in results if r.get("skipped"))
    print(
        f"汇总: 高分(>={q_keep}) {keep} | 中等({q_quarantine}~{q_keep}) {mid} "
        f"| 低分(<{q_quarantine}) {low} | 跳过 {skipped}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/score_demo.py <markdown文件路径>")
        sys.exit(1)
    main(sys.argv[1])
