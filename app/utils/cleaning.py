"""MinerU 输出清洗管线。

对 markdown 文本按行清洗:图片引用 → 页码 → 页眉/页脚 → 空短段 → 去重。
返回 (clean_md, stats)。规则阈值可调。
"""

import re
from collections import Counter
from hashlib import md5

# 页码形态。纯数字只认 1–3 位:4 位多为年份(2024),极少真页码 ≥1000。
_PAGE_PATTERNS = [
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^\s*第\s*\d+\s*页\s*$"),
    re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
]

# 结构行开头标记,不当作碎片短段删除
_MARKDOWN_MARKERS = ("#", "```", "|", "-", ">", "*", "+", "[!")
# 页眉步只豁免表头/标题,不用 '-' —— 否则重复 bullet 也会跳过页眉删除
_HEADER_KEEP_PREFIX = ("#", "|")

_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SYMBOL_KEEP = frozenset("≥≤±∑~→%<>")
_DETAIL_CAP = 200
CLEAN_RULESET_VERSION = "v1"


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


def _is_table_or_heading(text: str) -> bool:
    return text.strip().startswith(_HEADER_KEEP_PREFIX)


def _note(stats: dict, rule: str, line_no: int, text: str) -> None:
    detail = stats["removed_detail"]
    if len(detail) >= _DETAIL_CAP:
        return
    detail.append({"rule": rule, "line_no": line_no, "text": (text or "")[:80]})


def clean(md_text: str, header_threshold: int = 3, header_max_len: int = 60) -> tuple[str, dict]:
    stats = {
        "blocks_before": 0,
        "blocks_after": 0,
        "removed_images": 0,
        "removed_page_numbers": 0,
        "removed_headers": 0,
        "removed_short_blocks": 0,
        "removed_duplicates": 0,
        "removed_detail": [],
    }

    lines = md_text.splitlines()
    stats["blocks_before"] = len(lines)

    # 0) 图片引用
    kept = []
    for i, line in enumerate(lines):
        new_line, n = _IMAGE_REF.subn("", line)
        stats["removed_images"] += n
        if n and not new_line.strip():
            _note(stats, "image", i + 1, line)
        if new_line.strip():
            kept.append(new_line)
    lines = kept

    # 1) 页码
    kept = []
    for i, line in enumerate(lines):
        if _is_page_number(line):
            stats["removed_page_numbers"] += 1
            _note(stats, "page_number", i + 1, line)
        else:
            kept.append(line)
    lines = kept

    # 2) 页眉/页脚:短文本且全局重复 ≥ 阈值则整段删(页眉就该全删)。
    #    长正文重复交给步骤 4 保留首次。表头/标题(|、#)即使重复也不当页眉。
    counts = Counter(_hash(line) for line in lines if line.strip())
    kept = []
    for i, line in enumerate(lines):
        t = line.strip()
        if (t and len(t) < header_max_len and counts[_hash(line)] >= header_threshold
                and not _is_table_or_heading(t)):
            stats["removed_headers"] += 1
            _note(stats, "header", i + 1, line)
        else:
            kept.append(line)
    lines = kept

    # 3) 空段/短段。单字符计量/关系符号保留,避免 ≥ ± 被当碎片删掉导致语义反转。
    kept = []
    for i, line in enumerate(lines):
        t = line.strip()
        if not t:
            continue
        if len(t) < 2 and not _is_structural(t) and t not in _SYMBOL_KEEP:
            stats["removed_short_blocks"] += 1
            _note(stats, "short_block", i + 1, line)
            continue
        kept.append(line)
    lines = kept

    # 4) 去重(保留首次出现)
    seen = set()
    kept = []
    for i, line in enumerate(lines):
        h = _hash(line)
        if h in seen:
            stats["removed_duplicates"] += 1
            _note(stats, "duplicate", i + 1, line)
            continue
        seen.add(h)
        kept.append(line)
    lines = kept

    clean_md = "\n".join(lines).rstrip() + "\n"
    stats["blocks_after"] = len(lines)
    return clean_md, stats
