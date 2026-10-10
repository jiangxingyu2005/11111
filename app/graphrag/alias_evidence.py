"""消歧第 ① 层:确定性证据。从原文里找「同一实体的另一个名字」,写成实体别名。

两类证据:
  - 文中明写同指:「伊斯坎德尔-M（9K720 Iskander-M）」「A 又称 B」「A，即 B」「代号 B」;
  - 型号编码:9K720、MGM-140、AN/TPQ-50 这类编码本身就是同一型号的标识。
别名进入注册表后,跨文档的确定性匹配(名称/别名)就能直接合并,全库通用。
只在同类型间成立:括号里如果是另一个已抽出的、不同类型的实体(如「雷达站（含对空搜索雷达）」),不算同指。
"""

from __future__ import annotations

import re

from .ids import normalize_name

_PAREN_RE = re.compile(r"[（(]([^（）()]{2,40})[）)]")
_ALSO_RE = re.compile(r"(?:又称|亦称|简称|俗称|别称|代号为?|，即|,即)\s*[「“\"]?([^，,。；;：:、\s「」“”\"（）()]{2,30})")
_LEAD_STRIP = ("又称", "亦称", "简称", "俗称", "别称", "代号为", "代号", "即")
_NOT_ALIAS_LEAD = ("含", "包括", "包含", "如", "例如", "下辖", "隶属", "配属", "见", "注")
_PARAM_RE = re.compile(r"^[\d.\s~～\-至到]*\d[\d.\s~～\-至到]*(ghz|mhz|khz|hz|μs|us|ms|ns|db|dbm|w|kw|km|m|kg|t|%)?$",
                       re.IGNORECASE)
# 型号编码:至少 2 个字母 + 横线/斜线 + 数字(MGM-140、AN/TPQ-50、Su-35),或 数字-字母-数字(9K720)
_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,4}(?:/[A-Za-z]{2,4})?-\d{1,4}[A-Za-z]?|\d{1,2}[A-Za-z]\d{2,4})(?![A-Za-z0-9])")
_CODE_TYPES = {"辐射源", "平台", "武器"}


def iter_model_codes(text: str) -> list[str]:
    """原文里出现的型号编码(保持写法,不去重归一)。"""
    return _CODE_RE.findall(text or "")


def model_codes(texts: list) -> set[str]:
    """从名称/型号/别名里取型号编码(归一化),如 {"mgm-140"}、{"9k720"}、{"an/tpq-50"}。"""
    out: set[str] = set()
    for t in texts:
        for m in _CODE_RE.findall(str(t or "")):
            out.add(normalize_name(m))
    return out


def entity_codes(name: str, entity_type: str, properties: dict, aliases: list) -> set[str]:
    if entity_type not in _CODE_TYPES:
        return set()
    return model_codes([name, (properties or {}).get("型号", "")] + list(aliases or []))


def _clean_alias(s: str) -> str | None:
    s = s.strip().strip("「」“”\"'")
    for lead in _LEAD_STRIP:
        if s.startswith(lead):
            s = s[len(lead):].strip()
            break
    if not (2 <= len(s) <= 30) or s.startswith(_NOT_ALIAS_LEAD) or s.endswith("等"):
        return None
    if any(c in s for c in "，,；;：:、"):
        return None
    if _PARAM_RE.match(s.replace(" ", "")):
        return None
    return s


def add_alias_evidence(entities: list, parents: list[dict]) -> int:
    """原地给实体补别名(同指写法 + 型号编码),返回新增别名条数。"""
    by_key: dict[str, object] = {}
    for e in entities:
        for n in [e.name] + list(e.aliases or []):
            by_key.setdefault(normalize_name(n), e)
    keys = sorted(by_key, key=len, reverse=True)
    added = 0

    def _add(e, alias: str) -> None:
        nonlocal added
        if normalize_name(alias) not in {normalize_name(a) for a in [e.name] + list(e.aliases or [])}:
            e.aliases.append(alias)
            added += 1

    for p in parents:
        text = f"{p.get('title') or ''}\n{p.get('content') or ''}"
        for rx in (_PAREN_RE, _ALSO_RE):
            for m in rx.finditer(text):
                alias = _clean_alias(m.group(1))
                if not alias:
                    continue
                before = normalize_name(text[max(0, m.start() - 40):m.start()]).rstrip("，,")
                owner = next((by_key[k] for k in keys if before.endswith(k)), None)
                if owner is None:
                    continue
                other = by_key.get(normalize_name(alias))
                if other is not None and other is not owner:
                    if other.entity_type != owner.entity_type:
                        continue   # 括号里是另一个不同类型的实体,不是同指
                    _add(other, owner.name)   # 两个都被抽成了实体:互为别名,图增强时按别名合并
                _add(owner, alias)

    for e in entities:   # 型号编码作为别名:跨文档同编码即同型号
        for code in entity_codes(e.name, e.entity_type, e.properties, e.aliases):
            _add(e, code)
    return added
