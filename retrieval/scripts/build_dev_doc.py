"""从 retrieval 仓库真实源码生成《在线检索框架开发文档_V3.1》的代码基线附录。

用法：python scripts/build_dev_doc.py <源文档.md> <输出文档.md>
约定：源码即文档——附录中的代码全部从仓库文件实时读取，杜绝手抄漂移。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 纳入文档的文件（相对仓库根），顺序即章节顺序
CODE_FILES = [
    ("入口与配置", [
        "app/main.py",
        "app/config.py",
        ".env.example",
    ]),
    ("dev 通道（哈希嵌入 + 内存三库 + 故障注入）", [
        "app/store/hash_embed.py",
        "app/store/fake.py",
    ]),
    ("API 层（SSE 主链 + 独立图谱接口）", [
        "app/api/retrieval.py",
        "app/api/graph.py",
    ]),
    ("查询理解（QRY-02）", [
        "app/understand/query_parser.py",
    ]),
    ("三路召回", [
        "app/recall/vector_route.py",
        "app/recall/bm25_route.py",
        "app/recall/graph_route.py",
    ]),
    ("融合 / 重排 / 父块回捞", [
        "app/fuse/rrf.py",
        "app/fuse/rerank.py",
        "app/fuse/parent_backfill.py",
    ]),
    ("生成 / 降级 / 观测", [
        "app/generate/report.py",
        "app/degrade/codes.py",
        "app/degrade/orchestrator.py",
        "app/trace/spans.py",
        "app/graph/expand.py",
    ]),
    ("可运行件套（种子数据 / 模型 stub / 真环境基础设施）", [
        "scripts/gen_seed.py",
        "services/model_stub.py",
        "docker-compose.yml",
        "sql/001_init.sql",
        "scripts/init_milvus.py",
        "scripts/seed_real.py",
    ]),
    ("一键冒烟与测试（已通过的断言）", [
        "scripts/smoke.py",
        "tests/contract/test_pipeline_fake.py",
        "tests/degrade/test_degrade_matrix.py",
        "tests/test_config.py",
        "tests/test_rrf.py",
    ]),
    ("qingxi 离线侧配套（BGE-M3 双通道接口）", [
        "../qingxi/app/graphrag/embeddings.py",
    ]),
]

LANG = {".py": "python", ".yaml": "yaml", ".json": "json", ".md": "markdown"}


def fence(path: Path) -> str:
    ext = path.suffix if path.suffix in LANG else ".py"
    body = path.read_text(encoding="utf-8").rstrip()
    return f"```{LANG[ext]}\n# {path.as_posix()}\n{body}\n```"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_dev_doc.py <src.md> <out.md>")
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    doc = src.read_text(encoding="utf-8")

    doc = doc.replace("在线检索框架开发文档 V3.0（实施版）",
                      "在线检索框架开发文档 V3.1（实施版 · 含已验证代码基线）")
    doc = doc.replace("> **生成日期**：2026-09-28",
                      "> **生成日期**：2026-09-28 ｜ **V3.1 变更**：新增「附录 C · 代码基线」（今日交付并验证通过的全部源码，由 `scripts/build_dev_doc.py` 从仓库实时生成，杜绝手抄漂移）。本文为**唯一主施工文档**，V3.0 内容已并入。")

    parts = [
        "\n---\n\n## 附录 C · 代码基线（今日交付并通过验证）\n",
        "> 说明：以下源码全部来自 `retrieval/` 仓库真实文件，由 `scripts/build_dev_doc.py` 实时读取生成。\n"
        "> 代码状态分三类：**【已实现并验证】**（fake 通道端到端跑通 + 断言覆盖）、"
        "**【骨架就位】**（接口签名+降级路径已通，待真实服务联调填肉）、**【预留接口】**（二期启用只改配置）。\n"
        "> 复跑验证：`python scripts/smoke.py`（端到端 + 5 个降级注入场景）与 `python -m pytest -q`。\n",
    ]
    for idx, (title, files) in enumerate(CODE_FILES, start=1):
        parts.append(f"\n### C.{idx} {title}\n")
        for rel in files:
            p = ROOT / rel
            if not p.exists():
                raise SystemExit(f"missing file: {rel}")
            parts.append(f"\n{fence(p)}\n")

    parts.append(
        f"\n### C.{len(CODE_FILES) + 1} 验证记录与复跑 SOP\n\n"
        "```bash\n"
        "cd retrieval\n"
        "python scripts/gen_seed.py     # 生成 data/seed（换嵌入通道必须重跑）\n"
        "python scripts/smoke.py        # 端到端 + 5 个降级注入，退出码 0 = 全绿\n"
        "python -m pytest -q            # 24 条契约 / 降级 / 配置测试\n"
        "```\n\n"
        "已通过（fake 通道，conda rag 环境，真实 HTTP 层）：\n\n"
        "| 项 | 结果 |\n"
        "|---|---|\n"
        "| SSE 事件序列 | `meta → progress×2 → results → assets → token×N → done`，无 error |\n"
        "| 聚合响应契约 | 逐字段通过 `api/retrieval_response.v1.schema.json` |\n"
        "| 证据锚点 | 每条命中 `evidence{child_id, span}` 均为真值（非 null / 非 [0,0]） |\n"
        "| 父块回捞 | 5 条命中全部带父块正文（small-to-big 生效） |\n"
        "| 图谱二次取数 | `nodes=12 edges=28`，通过 `subgraph_view.v1.schema.json`，每条边带 evidence |\n"
        "| 权限红线 | 缺 scope → 422；只给 kb_public 时命中集合 ⊆ {kb_public} |\n"
        "| 降级矩阵 | vector/graph/pg/llm/rerank 五路注入：行为生效 + degraded 留痕 + 主链存活 |\n"
        "| 无静默降级 | 主链路场景 `degraded=[]` |\n\n"
        "> 口径提醒：FastAPI 0.139 起 `include_router` 为懒挂载（`_IncludedRouter`），\n"
        "> `app.routes` 不再扁平展开——路由验证必须走 TestClient/真实请求，不能数 `app.routes`。\n"
        "> 另一个已踩过的坑：`.env` 里写成 `KEY=#注释`（`#` 前无空格）会被 dotenv 当值读入，\n"
        "> 曾导致 `RETRIEVAL_FAULT_INJECT` 静默带上全部故障项。注释必须单独一行。\n"
    )
    out.write_text(doc + "".join(parts), encoding="utf-8")
    print(f"generated: {out}  ({len((doc + ''.join(parts)).splitlines())} lines)")


if __name__ == "__main__":
    main()
