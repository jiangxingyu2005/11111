"""建 Milvus collection（ems_chunks / ems_entities / community_scene）。

用法：
  python scripts/init_milvus.py                 # 建库（dense-only，与当前 IF-1 前状态一致）
  python scripts/init_milvus.py --with-sparse   # 一并建 text_sparse 列（IF-1 完成后用）
  python scripts/init_milvus.py --with-secondary# 一并建 embedding_v2 列（dense B 路）
  python scripts/init_milvus.py --with-community# 一并建 community_scene 集合（IF-4 后）
  python scripts/init_milvus.py --drop          # 先删再建（重建 collection 会清空数据，慎用于生产）

字段口径必须与 config/retrieval.yaml 一致：
  - 向量字段 embedding（dense 1024，COSINE），embedding_v2（B 路 256），sparse 字段 text_sparse（BM25）
  - scope 过滤字段 kb_id（R4：过滤表达式必带 scope）
  - 证据锚点 span_start / span_end（IF-2：没有它就没有"回原文高亮"）
  - content_version：父块版本漂移检测
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.store.hash_embed import NUM_DIM  # noqa: E402

DIM = 1024   # BGE-M3 dense 维度


def build_chunk_schema(client, with_sparse: bool, with_secondary: bool):
    from pymilvus import DataType, MilvusClient  # noqa: F401
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("child_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
    schema.add_field("section", DataType.VARCHAR, max_length=512)
    schema.add_field("span_start", DataType.INT64)
    schema.add_field("span_end", DataType.INT64)
    schema.add_field("content_version", DataType.INT64)
    schema.add_field("kg_entity_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=32, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    if with_secondary:
        schema.add_field("embedding_v2", DataType.FLOAT_VECTOR, dim=NUM_DIM)
    if with_sparse:
        schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
    return schema


def build_entity_schema():
    from pymilvus import DataType, MilvusClient
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("entity_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("type", DataType.VARCHAR, max_length=64)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    return schema


def build_community_schema():
    from pymilvus import DataType, MilvusClient
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("community_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("summary", DataType.VARCHAR, max_length=4096)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
    schema.add_field("stale", DataType.BOOL)
    schema.add_field("member_entity_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=64, max_length=64)
    schema.add_field("scene_vector", DataType.FLOAT_VECTOR, dim=DIM)
    return schema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-sparse", action="store_true", help="一并建 text_sparse 列（IF-1 后）")
    ap.add_argument("--with-secondary", action="store_true", help="一并建 embedding_v2 列（B 路）")
    ap.add_argument("--with-community", action="store_true", help="一并建 community_scene 集合（IF-4 后）")
    ap.add_argument("--drop", action="store_true", help="先删后建（会清空数据）")
    args = ap.parse_args()

    from pymilvus import MilvusClient

    s = get_settings()
    client = MilvusClient(uri=s.milvus_uri)

    for name, schema in ((s.milvus_chunk_collection,
                          build_chunk_schema(client, args.with_sparse, args.with_secondary)),
                         (s.milvus_entity_collection, build_entity_schema())):
        if args.drop and client.has_collection(name):
            client.drop_collection(name)
            print(f"[init_milvus] dropped {name}")
        if client.has_collection(name):
            print(f"[init_milvus] exists, skip: {name}")
            continue
        idx = client.prepare_index_params()
        idx.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        if args.with_secondary and name == s.milvus_chunk_collection:
            idx.add_index(field_name="embedding_v2", index_type="AUTOINDEX", metric_type="COSINE")
        if args.with_sparse and name == s.milvus_chunk_collection:
            idx.add_index(field_name="text_sparse", index_type="SPARSE_INVERTED_INDEX",
                          metric_type="IP")
        client.create_collection(collection_name=name, schema=schema, index_params=idx)
        print(f"[init_milvus] created {name}")

    if args.with_community:
        name = s.milvus_community_collection
        if args.drop and client.has_collection(name):
            client.drop_collection(name)
            print(f"[init_milvus] dropped {name}")
        if not client.has_collection(name):
            idx = client.prepare_index_params()
            idx.add_index(field_name="scene_vector", index_type="AUTOINDEX", metric_type="COSINE")
            client.create_collection(collection_name=name, schema=build_community_schema(),
                                     index_params=idx)
            print(f"[init_milvus] created {name}")

    client.load_collection(s.milvus_chunk_collection)
    client.load_collection(s.milvus_entity_collection)
    if args.with_community:
        client.load_collection(s.milvus_community_collection)
    print(f"[init_milvus] done. sparse={'on' if args.with_sparse else 'off'} "
          f"secondary={'on' if args.with_secondary else 'off'} "
          f"community={'on' if args.with_community else 'off'} dim={DIM}")
    if args.with_sparse:
        print("[init_milvus] 记得把 config/retrieval.yaml 的 milvus.sparse.enabled 改为 true")
    if args.with_secondary:
        print("[init_milvus] 记得把 config/retrieval.yaml 的 milvus.dense.secondary.enabled 改为 true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
