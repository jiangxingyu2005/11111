"""MinerU 输出清洗管线。

对 markdown 文本按行清洗:图片引用 → 页码 → 页眉/页脚 → 空短段 → 去重。
返回 (clean_md, stats)。规则阈值可调。
"""

import re
from collections import Counter
from hashlib import md5

# 页码形态
_PAGE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),               # 纯数字
    re.compile(r"^\s*第\s*\d+\s*页\s*$"),          # 第 N 页
    re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),            # - N -
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),          # 3 / 12
]

# 结构行开头标记,不当作碎片短段删除
_MARKDOWN_MARKERS = ("#", "```", "|", "-", ">", "*", "+", "[!")

# markdown 图片引用:![alt](url),含 base64 内嵌 / 本地 / 远程路径
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def strip_image_refs(text: str) -> str:
    """去掉 markdown 图片引用 ![alt](url)(base64 内嵌 / 本地 / 远程),返回纯文本。"""
    return _IMAGE_REF.sub("", text)


def _is_page_number(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # 剥掉 markdown 标题前缀(如 "## 42" -> "42"),兼容 MinerU 把页码误判为标题的情况
    t = re.sub(r"^#{1,6}\s*", "", t)
    return bool(t) and any(p.match(t) for p in _PAGE_PATTERNS)


def _hash(text: str) -> str:
    return md5(text.strip().encode("utf-8")).hexdigest()


def _is_structural(text: str) -> bool:
    return text.strip().startswith(_MARKDOWN_MARKERS)


def clean(md_text: str, header_threshold: int = 3, header_max_len: int = 60) -> tuple[str, dict]:
    stats = {
        "blocks_before": 0,
        "blocks_after": 0,
        "removed_images": 0,
        "removed_page_numbers": 0,
        "removed_headers": 0,
        "removed_short_blocks": 0,
        "removed_duplicates": 0,
    }

    lines = md_text.splitlines()
    stats["blocks_before"] = len(lines)

    # 0) 图片引用:去掉 markdown 图片(base64 内嵌 / 本地 / 远程),对文本检索无意义且会膨胀正文
    kept = []
    for line in lines:
        line, n = _IMAGE_REF.subn("", line)
        stats["removed_images"] += n
        if line.strip():
            kept.append(line)
    lines = kept

    # 1) 页码
    kept = []
    for line in lines:
        if _is_page_number(line):
            stats["removed_page_numbers"] += 1
        else:
            kept.append(line)
    lines = kept

    # 2) 页眉/页脚:短文本(长度 < header_max_len)且全局重复 >= header_threshold 次。
    #    长正文即使重复也只交给去重规则处理(保留首次),避免误删。
    counts = Counter(_hash(line) for line in lines if line.strip())
    kept = []
    for line in lines:
        t = line.strip()
        if t and len(t) < header_max_len and counts[_hash(line)] >= header_threshold:
            stats["removed_headers"] += 1
        else:
            kept.append(line)
    lines = kept

    # 3) 空段/短段
    kept = []
    for line in lines:
        t = line.strip()
        if not t:
            continue
        if len(t) < 2 and not _is_structural(t):
            stats["removed_short_blocks"] += 1
            continue
        kept.append(line)
    lines = kept

    # 4) 去重(保留首次出现)
    seen = set()
    kept = []
    for line in lines:
        h = _hash(line)
        if h in seen:
            stats["removed_duplicates"] += 1
            continue
        seen.add(h)
        kept.append(line)
    lines = kept

    clean_md = "\n".join(lines).rstrip() + "\n"
    stats["blocks_after"] = len(lines)
    return clean_md, stats
