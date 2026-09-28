"""确定性 ID / 归一化助手(实体/关系 ID 从规范化内容哈希,同父块重跑不翻倍)。"""

from __future__ import annotations

import hashlib
import re

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)


def normalize_name(name: str) -> str:
    """名称归一化:去首尾空白、全角转半角、折叠内部空白、ASCII 小写。"""
    s = str(name).translate(_FULLWIDTH).strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def canonical_entity_id(name: str, entity_type: str) -> str:
    """确定性实体 ID:hash(归一化名称 | 类型) 前 16 位。"""
    key = f"{normalize_name(name)}|{entity_type}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def canonical_relation_id(head_id: str, tail_id: str, rel_type: str, source_parent_id: str) -> str:
    """确定性关系 ID:hash(head|rel_type|tail|source_parent_id) 前 16 位。"""
    key = f"{head_id}|{rel_type}|{tail_id}|{source_parent_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
