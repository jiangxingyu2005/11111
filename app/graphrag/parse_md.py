"""md 工具:front-matter 剥离与内容指纹。

融合后 qingxi 的 clean.md 无 front-matter(三元元数据置空),故不解析 yaml、
不校验必填字段;仅保留剥离与哈希工具给 chunk 阶段用。
"""

from __future__ import annotations

import hashlib
import re

_FM_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict, str]:
    """剥离 front-matter,返回 ({}, body)。融合后不解析元数据字段。"""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    return {}, text[m.end():]


def content_hash(content: str) -> str:
    """内容指纹(full sha256),独立落库做变更检测,不拼入 ID。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
