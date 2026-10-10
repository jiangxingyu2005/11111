"""实体名形态闸门:像句子/字段残片的名字不进图。"""
from __future__ import annotations

import re

_NAME_PRED = re.compile(
    r"(定位为|遂行|负责|包括|属于|的是|该型|作为|用于|承担|成为|通常|公开估计)"
)
_PATHY = re.compile(r"^[一-鿿]{2,8}[-－][一-鿿]{2,8}$")
_FIELD_LABELS = {
    "首飞", "服役", "标准", "少量列装", "公开估计", "来源", "类别", "型号",
    "国别", "制造商", "特色", "用途", "平台", "制导",
}
_MAX_ZH = 24


def looks_like_entity_name(name: str) -> bool:
    """False = 不像实体名(句子碎片/表头/大类-细类)。含拉丁字母的不限长。"""
    n = str(name or "").strip()
    if not n or n in _FIELD_LABELS:
        return False
    if _NAME_PRED.search(n) or _PATHY.match(n):
        return False
    if not re.search(r"[A-Za-z]", n) and len(n) > _MAX_ZH:
        return False
    return True
