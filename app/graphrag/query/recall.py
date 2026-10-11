"""GP-08 QRY-01:A/B 双路召回(Plan §3.9,async 移植)。

A 路:绑定子块,ARRAY_CONTAINS(kg_entity_ids, entity_id) 过滤命中;
B 路:未绑定子块,ARRAY_LENGTH(kg_entity_ids) == 0。
两路均带 enabled == true 过滤(禁用联动),并加 kb_id 过滤(知识库隔离)。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from ..config import get
from ..embeddings import EmbeddingClient
from ..ids import normalize_name
from ..mesh.derived import _freq_interval
from ..milvus_client import F_ENABLED, F_KB_ID, F_KG_ENTITY_IDS, AsyncMilvusClient
from .nl2query import nl2query, parse_quantities

# 型号常写成「编号 + 名称」(MGM-140 ATACMS / 9K720 Iskander-M),用户只说其中一段
_SPLIT_RE = re.compile(r"[\s/、,，;；()（）]+")
_MIN_KEY_LEN = 3   # 太短的片段(如「M」)不单独当匹配键,避免误命中


def _fold(s: str) -> str:
    """匹配用:去掉连字符/下划线,让 NH-90 能对上 NH90。不改入库 ID。"""
    return s.replace("-", "").replace("_", "")


def _match_keys(names: list[str]) -> set[str]:
    """实体名/别名 -> 匹配键:全名 + 按空格/斜杠等切出的片段(归一化后)。"""
    keys: set[str] = set()
    for n in names:
        if not n:
            continue
        keys.add(normalize_name(n))
        for part in _SPLIT_RE.split(str(n)):
            p = normalize_name(part)
            if len(p) >= _MIN_KEY_LEN:
                keys.add(p)
    folded = {_fold(k) for k in keys}
    return {k for k in (keys | folded) if k}


# 属性名里带这些字、值又没写单位时,按量纲族补全(射程 500 → 500km)
_KEY_FAM = (
    (("频率", "频段", "重频", "带宽"), "hz"),
    (("射程", "探测", "航程", "半径", "距离"), "km"),
    (("脉宽",), "s"),
)


def _qty_from_prop(key: str, val) -> list[tuple[float, float, str]]:
    text = str(val or "")
    found = parse_quantities(text)
    if found:
        return found
    if not re.search(r"\d", text):
        return []
    fam = next((f for keys, f in _KEY_FAM if any(k in key for k in keys)), None)
    if not fam:
        return []
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        lo, hi = float(nums[0]), float(nums[1])
        return [(min(lo, hi), max(lo, hi), fam)]
    if len(nums) == 1:
        x = float(nums[0])
        return [(x, x, fam)]
    return []


def _qty_overlap(query_qtys, prop_qtys) -> bool:
    for a in query_qtys:
        for b in prop_qtys:
            if a[2] == b[2] and a[0] <= b[1] and b[0] <= a[1]:
                return True
    return False


def _qty_key_ok(query: str, key: str) -> bool:
    """问句点了字段名则只比对应属性,避免 500km 打到探测距离。"""
    if "射程" in query:
        return "射程" in key
    if "探测" in query or "作用距离" in query:
        return "探测" in key or "作用距离" in key
    if "航程" in query:
        return "航程" in key
    if "作战半径" in query or "半径" in query:
        return "半径" in key
    if any(x in query.lower() for x in ("ghz", "mhz", "khz", "频率")):
        return any(k in key for k in ("频率", "频段", "重频", "带宽"))
    return True



def match_entities(query: str, entities: list[dict]) -> list[str]:
    """名称/别名/型号子串命中,或问句中的数量/频段/调制与实体属性重合。

    「2.4GHz 有哪些装备」「射程 500km 是哪型导弹」走属性区间比对(同量纲才比),
    后续新数字字段只要写进 properties 且带单位(或属性名能推断单位)即可回推,不必再改匹配。
    """
    q = _fold(normalize_name(query))
    cond = nl2query(query)
    q_freq = _freq_interval(cond.get("freq_band") or query)
    q_mod = _fold(normalize_name(cond.get("modulation_type") or ""))
    q_qtys = list(cond.get("quantities") or [])
    if q_freq:
        q_qtys.append((q_freq[0], q_freq[1], "hz"))
    hits = []
    for e in entities:
        props = _props(e.get("properties"))
        names = [e.get("name", "")] + list(e.get("aliases") or [])
        model = props.get("型号")
        if model:
            names.append(str(model))
        if any(_fold(k) in q for k in _match_keys(names)):
            hits.append(e["entity_id"])
            continue
        if q_qtys:
            matched_qty = False
            for k, v in props.items():
                if k in ("型号", "origin", "role", "粒度", "国别"):
                    continue
                if not _qty_key_ok(query, str(k)):
                    continue
                if _qty_overlap(q_qtys, _qty_from_prop(str(k), v)):
                    hits.append(e["entity_id"])
                    matched_qty = True
                    break
            if matched_qty:
                continue
        band = _fold(normalize_name(str(props.get("频段") or "")))
        if band and len(band) >= 2 and band in q:
            hits.append(e["entity_id"])
            continue
        mod = _fold(normalize_name(str(props.get("调制方式") or "")))
        if mod and len(mod) >= 2 and (mod in q or (q_mod and q_mod in mod)):
            hits.append(e["entity_id"])
    return hits


def _props(v) -> dict:
    """Neo4j 里 properties 存成 JSON 字符串,这里解析回 dict。"""
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v) if v else {}
    except (TypeError, ValueError):
        return {}


def and_expr(*parts: str) -> str:
    """拼接 Milvus 过滤表达式(自动跳过空串)。"""
    return " && ".join(p for p in parts if p)


def kb_expr(kb_id) -> str:
    """kb_id 或 id 列表 → Milvus 过滤;空 = 不按库滤。"""
    if kb_id is None or kb_id == "" or kb_id == []:
        return ""
    if isinstance(kb_id, (list, tuple, set)):
        ids = [str(x) for x in kb_id if x not in (None, "")]
        if not ids:
            return ""
        if len(ids) == 1:
            return f"{F_KB_ID} == '{ids[0]}'"
        inner = ", ".join(f"'{x}'" for x in ids)
        return f"{F_KB_ID} in [{inner}]"
    return f"{F_KB_ID} == '{kb_id}'"



def _enabled_expr() -> str:
    return f"{F_ENABLED} == true"


async def recall_a(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                   entity_ids: list, kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """A 路:绑定子块,按实体 ARRAY_CONTAINS 过滤。"""
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    hits: list[dict] = []
    seen: set = set()
    for eid in entity_ids:
        expr = and_expr(_enabled_expr(), kb_expr(kb_id),
                        f"ARRAY_CONTAINS({F_KG_ENTITY_IDS}, '{eid}')")
        for h in await milvus.search(vec, topk, expr=expr):
            if h.get("child_id") not in seen:
                seen.add(h["child_id"])
                hits.append(h)
    return hits


async def recall_b(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                   kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """B 路:纯向量兜底,覆盖全部启用子块。

    不再限定「未绑定子块」:入库抽取后大部分子块都绑定了实体,若 B 路只搜未绑定的,
    实体名没匹配上时这些子块就彻底召回不到。两路重叠由 rerank 按 child_id 去重。
    """
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    expr = and_expr(_enabled_expr(), kb_expr(kb_id))
    return await milvus.search(vec, topk, expr=expr)


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")   # 拼进 Milvus 表达式的实体 ID 只允许安全字符


async def recall_expanded(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                          entity_ids: list, kb_id=None, topk: Optional[int] = None) -> list[dict]:
    """图扩展补召回:绑定到扩展实体的子块,一次检索(ARRAY_CONTAINS_ANY)。

    原先每个扩展实体各搜一次,扩展出 N 个实体就是 N 次检索;现在固定 1 次、最多 topk 条,
    结果交给 rerank 与 A/B 两路统一排序截断。
    """
    ids = [e for e in entity_ids if e and _SAFE_ID_RE.match(str(e))]
    if not ids:
        return []
    topk = topk or int(get("query.topk", 10))
    vec = await asyncio.to_thread(embeddings.embed_one, query)
    id_list = ", ".join(f"'{e}'" for e in ids)
    expr = and_expr(_enabled_expr(), kb_expr(kb_id), f"ARRAY_CONTAINS_ANY({F_KG_ENTITY_IDS}, [{id_list}])")
    return await milvus.search(vec, topk, expr=expr)


async def dual_recall(query: str, milvus: AsyncMilvusClient, embeddings: EmbeddingClient,
                      entity_ids: list, kb_id=None,
                      topk: Optional[int] = None) -> dict:
    """A/B 双路召回,返回 {a, b}。"""
    return {
        "a": await recall_a(query, milvus, embeddings, entity_ids, kb_id, topk),
        "b": await recall_b(query, milvus, embeddings, kb_id, topk),
    }
