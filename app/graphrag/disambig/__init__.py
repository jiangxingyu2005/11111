"""消歧(实体归一对齐)独立模块 —— 先独立验证、暂不接入入库链路。

三阶段匹配管道(确定性规则 → 向量相似度 → 大模型灰区)+ 持久化注册表语义
+ 全图批量对账(DSU + 四层回填)。本模块只依赖内存替身(stores.py),纯同步、
无 I/O、不 import PG/Neo4j/Milvus,可离线独立验证。

集成时:stores 换成 PG entity_registry / Milvus ems_entities / Neo4j / PG+Milvus
回写;engine 的匹配/归并逻辑不变。
"""
from .engine import (
    NeverJudge, ResolverConfig, align, global_resolve, persist, resolve_one,
)
from .stores import (
    DocIndexStore, EntityRecord, EntityRegistryStore, EntityVectorIndex, GraphStore,
)

__all__ = [
    "NeverJudge", "ResolverConfig", "align", "global_resolve", "persist", "resolve_one",
    "DocIndexStore", "EntityRecord", "EntityRegistryStore", "EntityVectorIndex", "GraphStore",
]
