"""pytest 根配置：统一注入仓库根与 services 到 sys.path。

为什么需要：tests/ 下有子目录（tests/generate/），各文件自己算 ROOT 极易差一层，
之前就踩过 `parent.parent` vs `parent.parent.parent` 的坑。统一在这里注入一次。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "services")):
    if p not in sys.path:
        sys.path.insert(0, p)
