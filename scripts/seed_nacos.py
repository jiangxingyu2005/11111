"""Nacos 播种脚本（离线包）：把扁平目录里的配置发布进 Nacos。

与 nacos_seed.py 的区别：后者读开发机 monorepo 布局（根 .env + retrieval/.env +
retrieval/config/*.yaml）；本脚本读一个扁平目录，按文件名作为 dataId 逐文件发布，
供离线包在容器内播种用（离线机未必有 Python，复用 qingxi-app 镜像里的 Python）。

用法（容器内）：
    python scripts/seed_nacos.py --server http://nacos:8848 --seed-dir /nacos-seed

行为：
  - <seed-dir>/<name>.env  -> dataId=<name>.env   type=properties
  - <seed-dir>/<name>.yaml -> dataId=<name>.yaml  type=yaml
  - 幂等：重复运行覆盖同 dataId。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.nacos_client import NacosConfigClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8848", help="Nacos 地址")
    ap.add_argument("--group", default="DEFAULT_GROUP", help="group")
    ap.add_argument("--namespace", default="", help="namespace（public 留空）")
    ap.add_argument("--seed-dir", default="/nacos-seed", help="配置文件扁平目录")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir)
    if not seed_dir.is_dir():
        print(f"错误：--seed-dir 不存在：{seed_dir}")
        return 1

    client = NacosConfigClient(args.server, namespace=args.namespace, group=args.group)

    files = sorted(p for p in seed_dir.iterdir()
                   if p.suffix in (".env", ".yaml", ".yml"))
    if not files:
        print(f"警告：{seed_dir} 下没有 .env/.yaml 文件")
        return 1

    print(f"-> Nacos: {args.server}  group={args.group}  seed-dir={seed_dir}")
    ok = 0
    for p in files:
        data_id = p.name
        type_ = "yaml" if p.suffix in (".yaml", ".yml") else "properties"
        content = p.read_text(encoding="utf-8")
        try:
            client.publish(data_id, content, type_=type_)
            ok += 1
            print(f"   [OK] {data_id}  ({len(content)} bytes, type={type_})")
        except Exception as e:  # noqa: BLE001 —— 逐条上报，不因一条失败中断
            print(f"   [FAIL] {data_id}: {e}")
    print(f"done: {ok}/{len(files)} published")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
