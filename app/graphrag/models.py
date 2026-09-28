"""GraphRAG 内存数据模型(抽取/绑定/导入环节使用;PG 表走 app/models/graphrag.py)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Entity:
    """实体(Neo4j :Entity)。"""

    entity_id: str
    name: str
    entity_type: str
    aliases: list = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    source_parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bind:
    """实体↔父块绑定(Neo4j :BINDS)。"""

    entity_id: str
    parent_id: str
    kg_role: str                         # primary | supplemental
    kg_confidence: float
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relation:
    """关系(Neo4j 关系边)。"""

    relation_id: str
    head_entity_id: str
    tail_entity_id: str
    rel_type: str                        # BELONGS_TO | SUPPORTS | CO_LINK | INVOLVES | SAME_BAND | CO_TIME | CO_LOCATE
    confidence: float
    evidence: str = ""
    source_parent_id: Optional[str] = None
    derived: bool = False                # True = 元数据派生边(可整体重算)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Support:
    """关系证据句(用于引用标注)。"""

    relation_id: str
    entity_id: str
    position: str                        # head | tail
    evidence_sentence: str
    parent_id: str

    def to_dict(self) -> dict:
        return asdict(self)
