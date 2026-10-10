"""原文提及扫描 + 覆盖补全。

生产默认 bind_only:标题/后缀/别名只挂到已有实体,只有型号编码(或标题+型号成对)可建节点。
国别独立提及建组织节点(origin=country),已有更长组织名则不拆。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import settings
from ..utils.scoring import is_garbled_name
from .alias_evidence import _ALSO_RE, _PAREN_RE, _clean_alias, iter_model_codes
from .countries import scan_countries
from .ids import canonical_entity_id, normalize_name
from .models import Entity
from .name_gate import looks_like_entity_name

_GENERIC = {
    "雷达", "导弹", "干扰机", "平台", "武器", "火炮", "飞机", "舰", "艇",
    "装备", "系统", "信号", "无标题", "首飞", "服役", "标准", "少量列装",
}
_CREATE_SOURCES = {"code", "model_label"}
_DEMONSTRATIVE = ("上述", "前述", "所述", "该", "此", "这", "本", "那")

# 长后缀优先,避免「反舰导弹」被切成「导弹」
_SUFFIX_TYPE: list[tuple[str, str]] = [
    ("反炮兵雷达", "辐射源"), ("相控阵雷达", "辐射源"), ("搜索雷达", "辐射源"),
    ("火控雷达", "辐射源"), ("预警雷达", "辐射源"), ("干扰机", "辐射源"), ("雷达", "辐射源"),
    ("弹道导弹", "武器"), ("巡航导弹", "武器"), ("空空导弹", "武器"), ("反舰导弹", "武器"),
    ("反坦克导弹", "武器"), ("自行榴弹炮", "武器"), ("榴弹炮", "武器"), ("火箭炮", "武器"),
    ("导弹", "武器"), ("鱼雷", "武器"),
    ("轰炸机", "平台"), ("战斗机", "平台"), ("运输机", "平台"), ("驱逐舰", "平台"),
    ("护卫舰", "平台"), ("航空母舰", "平台"), ("潜艇", "平台"),
]
_BAD_LEAD = "用于承担成为属于主要包括在从对为的了和与及并则即"
_SUFFIX_RE = re.compile(
    r"(?<![A-Za-z0-9一-鿿" + _BAD_LEAD + r"])"
    r"([A-Za-z0-9/\-·．.]{1,12}|[一-鿿]{2,10})"
    r"(?:" + "|".join(re.escape(s) for s, _ in _SUFFIX_TYPE) + r")"
)
_MODEL_LABEL_RE = re.compile(
    r"型号\s*[:：为是]?\s*([A-Za-z0-9][A-Za-z0-9/.\-]{1,32})"
)
_BOOK_RE = re.compile(r"《([^》]{2,40})》")


@dataclass
class Mention:
    text: str
    parent_id: str
    source: str
    hint: str | None


def guess_code_type(code: str) -> str:
    u = (code or "").upper().replace(" ", "")
    if u.startswith("AN/") or any(k in u for k in ("APG", "SPY", "TPQ", "SLQ", "SPS", "APY", "TPS")):
        return "辐射源"
    if re.match(r"^\d+[A-Z]", u) or any(u.startswith(k) for k in ("MGM", "AGM", "AIM", "PGM", "BGM", "R-")):
        return "武器"
    if re.match(r"^(F|SU|J|B|A|C|E|P|KC)-?\d", u):
        return "平台"
    return "辐射源"


def guess_span_type(span: str) -> str | None:
    s = (span or "").strip()
    for suf, t in _SUFFIX_TYPE:
        if s.endswith(suf) and s != suf:
            return t
    codes = iter_model_codes(s)
    return guess_code_type(codes[0]) if codes else None


def _ok_span(span: str) -> bool:
    s = (span or "").strip()
    if not s or s in _GENERIC or len(s) < 2:
        return False
    if any(s.startswith(d) for d in _DEMONSTRATIVE):
        return False
    if is_garbled_name(s) or not looks_like_entity_name(s):
        return False
    return True


def scan_parent(parent: dict) -> list[Mention]:
    """从一节标题+正文扫出必须覆盖的提及。"""
    pid = parent.get("parent_id") or ""
    title = (parent.get("title") or "").strip()
    content = parent.get("content") or ""
    text = f"{title}\n{content}" if title else content
    out: list[Mention] = []
    seen: set[str] = set()

    def add(span: str, source: str, hint: str | None = None) -> None:
        span = (span or "").strip().strip("「」“”\"'")
        if not _ok_span(span):
            return
        key = normalize_name(span)
        if not key or key in seen:
            return
        seen.add(key)
        out.append(Mention(span, pid, source, hint or guess_span_type(span)))

    if title and title != "无标题":
        add(title, "title")
    for code in iter_model_codes(text):
        add(code, "code", guess_code_type(code))
    for m in _MODEL_LABEL_RE.finditer(text):
        add(m.group(1), "model_label", guess_code_type(m.group(1)))
    for rx in (_PAREN_RE, _ALSO_RE):
        for m in rx.finditer(text):
            alias = _clean_alias(m.group(1))
            if alias:
                add(alias, "alias")
    for m in _BOOK_RE.finditer(text):
        add(m.group(1), "title")
    for m in _SUFFIX_RE.finditer(text):
        add(m.group(0), "suffix")  # 整段含后缀,不是只有前缀
    for c in scan_countries(text):
        add(c, "country", "组织")
    return out


def _index(entities: list[Entity]) -> dict[str, Entity]:
    by: dict[str, Entity] = {}
    for e in entities:
        for n in [e.name] + list(e.aliases or []) + [str((e.properties or {}).get("型号") or "")]:
            k = normalize_name(n)
            if k:
                by.setdefault(k, e)
    return by


def _find(by: dict[str, Entity], key: str) -> Entity | None:
    if not key:
        return None
    if key in by:
        return by[key]
    # 编码/专名互相包含(NH90 vs NH-90 已在 normalize 后仍可能差横杠,调用方会再 fold)
    fold = key.replace("-", "").replace("_", "")
    for k, e in by.items():
        kf = k.replace("-", "").replace("_", "")
        if fold == kf:
            return e
        if len(fold) >= 4 and (fold in kf or kf in fold):
            return e
    return None


def cover_mentions(entities: list[Entity], parents: list[dict],
                   allowed_types: list[str]) -> tuple[list[Entity], list[dict]]:
    """每个提及必须落在某实体 name/alias 上;否则按 hint 补节点。"""
    notes: list[dict] = []
    by = _index(entities)
    allowed = set(allowed_types) if allowed_types else set()

    # 同节标题与「型号 XXX」绑到同一节点
    title_by_parent = {}
    for p in parents:
        t = (p.get("title") or "").strip()
        if t and t != "无标题" and _ok_span(t):
            title_by_parent[p.get("parent_id")] = t

    leftover: list[Mention] = []
    for p in parents:
        mentions = scan_parent(p)
        codes_here = [m for m in mentions if m.source in ("code", "model_label")]
        title = title_by_parent.get(p.get("parent_id"))
        for m in mentions:
            e = _find(by, normalize_name(m.text))
            if e is None and m.source in ("title", "suffix") and codes_here:
                e = _find(by, normalize_name(codes_here[0].text))
            if e is None and m.source in ("code", "model_label") and title:
                e = _find(by, normalize_name(title))
            if e is not None:
                if (looks_like_entity_name(m.text)
                        and normalize_name(m.text) != normalize_name(e.name)
                        and m.text not in (e.aliases or [])):
                    e.aliases.append(m.text)
                    by[normalize_name(m.text)] = e
                if m.source in ("code", "model_label") and not (e.properties or {}).get("型号"):
                    e.properties["型号"] = m.text
                continue
            leftover.append(m)

    # LLM 全漏时:同节标题和「型号 XXX」建成一个节点,互为别名
    still: list[Mention] = []
    by_pid: dict[str, list[Mention]] = {}
    for m in leftover:
        by_pid.setdefault(m.parent_id, []).append(m)
    paired: set[int] = set()
    for pid, group in by_pid.items():
        title_m = next((x for x in group if x.source == "title"), None)
        code_m = next((x for x in group if x.source in ("code", "model_label")), None)
        if title_m and code_m:
            hint = code_m.hint if (not allowed or code_m.hint in allowed) else title_m.hint
            if allowed and hint not in allowed:
                hint = allowed_types[0]
            eid = canonical_entity_id(normalize_name(code_m.text), hint)
            ent = Entity(
                entity_id=eid, name=normalize_name(code_m.text), entity_type=hint,
                aliases=[title_m.text, code_m.text],
                properties={"型号": code_m.text, "origin": "code"},
                confidence=0.75, source_parent_id=pid,
            )
            entities.append(ent)
            by[normalize_name(code_m.text)] = ent
            by[normalize_name(title_m.text)] = ent
            notes.append({
                "parent_id": pid, "stage": "entity",
                "payload": {"name": code_m.text, "entity_type": hint, "aliases": [title_m.text]},
                "reason": "规则补抽:标题与型号编码同指,LLM 未覆盖",
                "attempts": 1,
            })
            paired.add(id(title_m))
            paired.add(id(code_m))
        for m in group:
            if id(m) not in paired:
                still.append(m)
    leftover = still
    mode = (getattr(settings, "graphrag_mentions_cover_mode", None) or "bind_only").lower()
    allow_create = mode == "create"

    for m in leftover:
        if m.source == "country":
            continue
        can_new = allow_create or m.source in _CREATE_SOURCES
        if not can_new or not m.hint:
            notes.append({
                "parent_id": m.parent_id, "stage": "entity",
                "payload": {"name": m.text, "source": m.source},
                "reason": f"提及未覆盖但来源{m.source}无建节点权限:仅留痕",
                "attempts": 1,
            })
            continue
        hint = m.hint if (not allowed or m.hint in allowed) else (
            next((t for t in ("辐射源", "武器", "平台") if t in allowed),
                 allowed_types[0] if allowed_types else "辐射源")
        )
        eid = canonical_entity_id(normalize_name(m.text), hint)
        existing = next((x for x in entities if x.entity_id == eid), None)
        if existing is not None:
            if m.text not in existing.aliases and normalize_name(m.text) != normalize_name(existing.name):
                if looks_like_entity_name(m.text):
                    existing.aliases.append(m.text)
            continue
        ent = Entity(
            entity_id=eid, name=normalize_name(m.text), entity_type=hint,
            aliases=[m.text] if normalize_name(m.text) != m.text else [],
            properties={"型号": m.text, "origin": "code"} if m.source in _CREATE_SOURCES
            else {"origin": "cover"},
            confidence=0.75, source_parent_id=m.parent_id,
        )
        entities.append(ent)
        by[normalize_name(m.text)] = ent
        notes.append({
            "parent_id": m.parent_id, "stage": "entity",
            "payload": {"name": m.text, "entity_type": hint, "source": m.source},
            "reason": f"规则补抽:原文{m.source}提及未被 LLM 覆盖",
            "attempts": 1,
        })
    entities, cnotes = _cover_countries(entities, parents, allowed_types)
    notes.extend(cnotes)
    return entities, notes


def _longer_org(by: dict, country_norm: str) -> Entity | None:
    for k, e in by.items():
        if e.entity_type == "组织" and country_norm in k and k != country_norm:
            return e
    return None


def _cover_countries(entities: list[Entity], parents: list[dict],
                     allowed_types: list[str]) -> tuple[list[Entity], list[dict]]:
    """独立国别提及 → 组织节点(origin=country);已有更长组织名则只写属性不拆。"""
    if allowed_types and "组织" not in allowed_types:
        return entities, []
    notes: list[dict] = []
    by = _index(entities)
    for p in parents:
        text = f"{p.get('title') or ''}\n{p.get('content') or ''}"
        pid = p.get("parent_id") or ""
        for c in scan_countries(text):
            kn = normalize_name(c)
            longer = _longer_org(by, kn)
            if longer is not None:
                longer.properties.setdefault("国别", c)
                continue
            e = _find(by, kn)
            if e is not None:
                e.properties.setdefault("国别", c)
                e.properties.setdefault("origin", e.properties.get("origin") or "llm")
                if e.entity_type == "组织":
                    e.properties["origin"] = e.properties.get("origin") or "country"
                    e.properties["role"] = "country"
                continue
            eid = canonical_entity_id(kn, "组织")
            ent = Entity(
                entity_id=eid, name=kn, entity_type="组织",
                aliases=[c] if c != kn else [],
                properties={"origin": "country", "role": "country", "国别": c},
                confidence=0.75, source_parent_id=pid,
            )
            entities.append(ent)
            by[kn] = ent
            notes.append({
                "parent_id": pid, "stage": "entity",
                "payload": {"name": c, "entity_type": "组织", "source": "country"},
                "reason": "规则补抽:独立国别提及",
                "attempts": 1,
            })
    return entities, notes
