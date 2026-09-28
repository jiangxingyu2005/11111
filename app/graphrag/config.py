"""GraphRAG 配置 shim。

取代 ems 的 yaml 配置加载:基础设施/阈值直接读 qingxi `.env`(app.config.settings),
实体/关系字典改由 extraction_dict 表读取(app/graphrag/dict.py,异步,每次现查)。
本模块只提供静态的嵌套配置树(供 llm/embeddings/neo4j/milvus 客户端构造)与 few-shot 读取。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import settings

_FEW_SHOT_DIR = Path(__file__).resolve().parent / "few_shot" / "relations"


def build_settings() -> dict:
    """把 qingxi `.env` 拼成 ems 客户端期望的嵌套 dict 结构(与 settings.yaml 对应)。"""
    return {
        "llm": {
            "provider": settings.graphrag_llm_provider,
            "base_url": settings.graphrag_llm_base_url,
            "api_key": settings.graphrag_llm_api_key,
            "model": settings.graphrag_llm_model,
            "temperature": settings.graphrag_llm_temperature,
            "max_tokens": settings.graphrag_llm_max_tokens,
            "max_retries": settings.graphrag_llm_max_retries,
        },
        "embedding": {
            "provider": settings.graphrag_embedding_provider,
            "model": settings.graphrag_embedding_model,
            "dim": settings.graphrag_embedding_dim,
            "batch_size": settings.graphrag_embedding_batch_size,
            "device": settings.graphrag_embedding_device,
            "use_fp16": settings.graphrag_embedding_use_fp16,
        },
        "neo4j": {
            "uri": settings.neo4j_uri,
            "user": settings.neo4j_user,
            "password": settings.neo4j_password,
        },
        "milvus": {
            "host": settings.milvus_host,
            "port": settings.milvus_port,
            "collection": settings.milvus_collection,
            "entity_collection": settings.graphrag_entity_collection,
            "dim": settings.milvus_dim,
            "iq_collection": settings.iq_collection,
            "iq_dim": settings.iq_dim,
        },
        "chunk": {
            "chunk_size": settings.graphrag_chunk_size,
            "overlap": settings.graphrag_chunk_overlap,
        },
        "graph": {
            "confidence_gate": {
                "high": settings.graphrag_confidence_high,
                "low": settings.graphrag_confidence_low,
            },
            "tau_a": settings.graphrag_tau_a,
            "tau_b": settings.graphrag_tau_b,
        },
        "query": {
            "topk": settings.graphrag_query_topk,
            "rerank_topk": settings.graphrag_query_rerank_topk,
            "expand_depth": settings.graphrag_query_expand_depth,
            "entity_boost_weight": settings.graphrag_query_entity_boost_weight,
        },
        "kafka": {
            "bootstrap_servers": settings.kafka_bootstrap_servers,
            "topic": settings.kafka_topic,
            "group_id": settings.kafka_group_id,
        },
        "route": {
            "file_size_mb": settings.graphrag_route_file_size_mb,
            "batch_count": settings.graphrag_route_batch_count,
        },
    }


def get(key: str, default: Any = None) -> Any:
    """按点号路径取值(如 get('chunk.chunk_size'))。仅静态配置;实体/关系走 dict.py。"""
    cur: Any = build_settings()
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_few_shot(rel_type: str) -> str:
    """读取 per-type few-shot(few_shot/relations/{REL_TYPE}.md);缺失返回空串。"""
    p = _FEW_SHOT_DIR / f"{rel_type}.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""
