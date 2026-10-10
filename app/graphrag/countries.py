"""国别白名单:消歧否决、提及补抽、prompt 共用。"""
from __future__ import annotations

import re

from .ids import normalize_name

# 别名 -> 规范中文名
CANONICAL = {
    "俄罗斯": "俄罗斯", "russia": "俄罗斯", "苏联": "俄罗斯", "俄联邦": "俄罗斯",
    "美国": "美国", "usa": "美国", "us": "美国", "unitedstates": "美国", "united states": "美国",
    "中国": "中国", "china": "中国", "prc": "中国",
    "英国": "英国", "uk": "英国", "unitedkingdom": "英国", "united kingdom": "英国", "britain": "英国",
    "法国": "法国", "france": "法国",
    "德国": "德国", "germany": "德国",
    "以色列": "以色列", "israel": "以色列",
    "日本": "日本", "japan": "日本",
    "印度": "印度", "india": "印度",
    "伊朗": "伊朗", "iran": "伊朗",
    "挪威": "挪威", "norway": "挪威",
    "瑞典": "瑞典", "sweden": "瑞典",
    "意大利": "意大利", "italy": "意大利",
    "澳大利亚": "澳大利亚", "australia": "澳大利亚",
    "韩国": "韩国", "southkorea": "韩国",
    "乌克兰": "乌克兰", "ukraine": "乌克兰",
    "欧洲": "欧洲", "esa": "欧洲", "eu": "欧洲",
}

# 后面紧跟这些则视为更长组织名的一部分,不当作独立国别
_ORG_TAIL = ("皇家", "海军", "空军", "陆军", "军队", "国防部", "航天")

_DISPLAY = {
    "俄罗斯": "俄罗斯", "美国": "美国", "中国": "中国", "英国": "英国", "法国": "法国",
    "德国": "德国", "以色列": "以色列", "日本": "日本", "印度": "印度", "伊朗": "伊朗",
    "挪威": "挪威", "瑞典": "瑞典", "意大利": "意大利", "澳大利亚": "澳大利亚",
    "韩国": "韩国", "乌克兰": "乌克兰", "欧洲": "欧洲",
}


def canon(value: str) -> str | None:
    if not value:
        return None
    return CANONICAL.get(normalize_name(value)) or CANONICAL.get(str(value).strip())


def scan_countries(text: str) -> list[str]:
    """独立国别提及(规范中文名)。不拆「英国皇家海军」里的英国。"""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    # 长别名优先
    aliases = sorted(CANONICAL.keys(), key=len, reverse=True)
    for alias in aliases:
        if not alias or alias.isascii() and len(alias) <= 2:
            continue
        disp = CANONICAL[alias]
        if disp in seen:
            continue
        pat = re.compile(
            r"(?<![A-Za-z0-9一-鿿])" + re.escape(alias) + r"(?![A-Za-z0-9一-鿿]|"
            + "|".join(_ORG_TAIL) + r")",
            re.IGNORECASE,
        )
        if pat.search(text):
            seen.add(disp)
            found.append(_DISPLAY.get(disp, disp))
    return found
