"""Nacos 配置中心播种脚本：把根 .env / retrieval/.env / config/*.yaml 发布进 Nacos。

用法（开发机，先起 Nacos）：
    docker compose -f retrieval/docker-compose.yml up -d nacos
    python scripts/nacos_seed.py [--server http://127.0.0.1:8848]

行为：
  - 读根 .env        -> dataId=qingxi.env（剥 NACOS_* bootstrap 键 + 行内注释）
  - 读 retrieval/.env -> dataId=retrieval.env（同上）
  - 读 retrieval/config/{retrieval,query_parser,relations}.yaml -> 同名 dataId（type=yaml）
  - 幂等：重复运行覆盖同 dataId。

注意：5 个 spec-only yaml（structured_filter/scalar_extract/scalar_adapter/
adapter_patch_generated/community_summary）代码不读，不迁、留在本地。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.nacos_client import NacosConfigClient  # noqa: E402


def _strip_inline_comment(v: str) -> str:
    """只剥「空白后跟 #」的行内注释；不剥值中无前置空格的 #（URL fragment/口令）。"""
    for i, ch in enumerate(v):
        if ch == "#" and i > 0 and v[i - 1].isspace():
            return v[:i].rstrip()
    return v.rstrip()


def _clean_env(path: Path, drop_prefix: str = "NACOS_") -> str:
    """.env -> 干净 properties：剥 drop_prefix 键 + 行内注释；保留整行注释/空行结构。"""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("#"):
            lines.append(line)
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith(drop_prefix):
            continue
        lines.append(f"{k}={_strip_inline_comment(v)}")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _read_nacos_bootstrap(env_path: Path) -> dict[str, str]:
    """从 env 文件读 NACOS_* bootstrap 键（决定连哪个 Nacos）。"""
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("NACOS_"):
            out[k] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="", help="Nacos 地址，默认读根 .env 的 NACOS_SERVER_ADDR")
    ap.add_argument("--group", default="", help="group，默认读 NACOS_GROUP 或 DEFAULT_GROUP")
    args = ap.parse_args()

    b = _read_nacos_bootstrap(ROOT / ".env")
    server = args.server or b.get("NACOS_SERVER_ADDR", "http://127.0.0.1:8848")
    group = args.group or b.get("NACOS_GROUP", "DEFAULT_GROUP")
    namespace = b.get("NACOS_NAMESPACE", "")

    client = NacosConfigClient(server, namespace=namespace, group=group,
                               username=b.get("NACOS_USERNAME", ""),
                               password=b.get("NACOS_PASSWORD", ""))

    jobs: list[tuple[str, str, str]] = []  # (dataId, content, type)
    jobs.append(("qingxi.env", _clean_env(ROOT / ".env"), "properties"))
    jobs.append(("retrieval.env", _clean_env(ROOT / "retrieval" / ".env"), "properties"))
    for name in ("retrieval", "query_parser", "relations"):
        p = ROOT / "retrieval" / "config" / f"{name}.yaml"
        if p.exists():
            jobs.append((f"{name}.yaml", p.read_text(encoding="utf-8"), "yaml"))

    print(f"-> Nacos: {server}  group={group}  namespace={namespace or '(public)'}")
    ok = 0
    for data_id, content, type_ in jobs:
        try:
            client.publish(data_id, content, type_=type_)
            ok += 1
            print(f"   [OK] {data_id}  ({len(content)} bytes, type={type_})")
        except Exception as e:
            print(f"   [FAIL] {data_id}: {e}")
    print(f"done: {ok}/{len(jobs)} published")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
