"""Milvus 客户端(ems common/milvus_client.py 移植,新增 kb_id 字段 + async 门面)。

Collection = ems_chunks,含 enabled / parent_enabled / kg_entity_ids,融合新增 kb_id(按库过滤)。
pymilvus 无官方 async 客户端:同步实现保留原样,AsyncMilvusClient 经 asyncio.to_thread
委托到线程池(唯一同步孤岛,见 docs)。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .config import build_settings

# 字段名常量(逐字段,含 enabled BOOL 与 kb_id)
F_CHILD_ID = "child_id"
F_CONTENT = "content"
F_PARENT_ID = "parent_id"
F_DOC_ID = "doc_id"
F_KB_ID = "kb_id"                 # 融合新增:知识库 id 过滤
F_ENABLED = "enabled"
F_PARENT_ENABLED = "parent_enabled"
F_KG_ENTITY_IDS = "kg_entity_ids"
F_VECTOR = "content_vector"

FIELD_NAMES = [
    F_CHILD_ID, F_CONTENT, F_PARENT_ID, F_DOC_ID, F_KB_ID,
    F_ENABLED, F_PARENT_ENABLED, F_KG_ENTITY_IDS, F_VECTOR,
]

# 实体向量库 ems_entities 字段(消歧语义匹配层 ②)
F_ENTITY_ID = "entity_id"
F_ENTITY_TYPE = "entity_type"
F_ENTITY_NAME = "name"
F_NAME_VECTOR = "name_vector"

ENTITY_FIELD_NAMES = [F_ENTITY_ID, F_ENTITY_TYPE, F_ENTITY_NAME, F_NAME_VECTOR]


class MilvusClient:
    """pymilvus 包装(同步)。写入全部 upsert,重跑幂等。"""

    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or build_settings()

    def connect(self) -> None:
        from pymilvus import connections  # 延迟导入
        s = self.settings["milvus"]
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=s["host"], port=s["port"])

    @property
    def collection_name(self) -> str:
        return self.settings["milvus"]["collection"]

    @property
    def entity_collection_name(self) -> str:
        return self.settings["milvus"]["entity_collection"]

    def create_collection(self) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
        if utility.has_collection(self.collection_name, using="default"):
            return
        dim = int(self.settings["milvus"]["dim"])
        fields = [
            FieldSchema(name=F_CHILD_ID, dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name=F_CONTENT, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=F_PARENT_ID, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=F_DOC_ID, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=F_KB_ID, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=F_ENABLED, dtype=DataType.BOOL),
            FieldSchema(name=F_PARENT_ENABLED, dtype=DataType.BOOL),
            FieldSchema(name=F_KG_ENTITY_IDS, dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR, max_capacity=64, max_length=256),
            FieldSchema(name=F_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="电磁信号子块向量库(kb_id 隔离)")
        coll = Collection(name=self.collection_name, schema=schema, using="default")
        coll.create_index(F_VECTOR, {"metric_type": "COSINE", "index_type": "IVF_FLAT",
                                     "params": {"nlist": 128}})

    def upsert(self, rows: list[dict]) -> None:
        from pymilvus import Collection
        coll = Collection(name=self.collection_name, using="default")
        data = [[r.get(f) for r in rows] for f in FIELD_NAMES]
        coll.upsert(data)

    def query(self, expr: str, output_fields: Optional[list] = None, limit: int = 1000) -> list[dict]:
        from pymilvus import Collection
        coll = Collection(name=self.collection_name, using="default")
        coll.load()
        return coll.query(expr=expr, output_fields=output_fields or FIELD_NAMES, limit=limit)

    def count(self) -> int:
        from pymilvus import Collection
        coll = Collection(name=self.collection_name, using="default")
        return coll.num_entities

    def search(self, vector: list, topk: int, expr: Optional[str] = None) -> list[dict]:
        from pymilvus import Collection
        coll = Collection(name=self.collection_name, using="default")
        coll.load()
        params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        res = coll.search([vector], F_VECTOR, params, limit=topk, expr=expr,
                          output_fields=[f for f in FIELD_NAMES if f != F_VECTOR])
        out = []
        for hit in res[0]:
            # pymilvus 3.x:Hit.fields 才是 output_fields 的 dict;hit.entity 是 Hit 自身
            row = dict(hit.fields)
            row["_score"] = float(hit.distance)   # COSINE 相似度,供 query 重排使用
            out.append(row)
        return out

    def update_kg_entity_ids(self, child_id: str, ids: list) -> None:
        """部分更新:回写 kg_entity_ids。"""
        self.update_fields(child_id, **{F_KG_ENTITY_IDS: list(ids)})

    def update_fields(self, child_id: str, **fields) -> None:
        """通用部分更新(SEG 开关 / META 回写共用)。"""
        rows = self.query(f"{F_CHILD_ID} == '{child_id}'", output_fields=FIELD_NAMES)
        for row in rows:
            row.update(fields)
            self.upsert([row])

    def delete(self, ids: list[str]) -> None:
        """按主键 child_id 批量删除(SEG 编辑增删 / CDC 子块删除)。"""
        if not ids:
            return
        from pymilvus import Collection
        coll = Collection(name=self.collection_name, using="default")
        coll.load()
        quoted = ", ".join(f"'{c}'" for c in ids)
        coll.delete(expr=f"{F_CHILD_ID} in [{quoted}]")

    def flush(self, name: Optional[str] = None) -> None:
        """强制落盘(子块 upsert 不自动 flush,不 flush 则 search 看不到新向量)。"""
        from pymilvus import Collection
        Collection(name=name or self.collection_name, using="default").flush()

    def drop_collection(self, name: str) -> None:
        """删除 collection(不存在则跳过,幂等);供重建向量等维护场景用。"""
        from pymilvus import utility
        if utility.has_collection(name, using="default"):
            utility.drop_collection(name, using="default")

    # ---- 实体向量库 ems_entities(消歧语义匹配层 ②)----
    def create_entity_collection(self) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
        name = self.entity_collection_name
        if utility.has_collection(name, using="default"):
            return
        dim = int(self.settings["milvus"]["dim"])
        fields = [
            FieldSchema(name=F_ENTITY_ID, dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name=F_ENTITY_TYPE, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=F_ENTITY_NAME, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=F_NAME_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="实体名向量库(消歧语义匹配)")
        coll = Collection(name=name, schema=schema, using="default")
        coll.create_index(F_NAME_VECTOR, {"metric_type": "COSINE", "index_type": "IVF_FLAT",
                                          "params": {"nlist": 128}})

    def upsert_entity_vector(self, entity_id: str, name: str, entity_type: str,
                             vector: list) -> None:
        from pymilvus import Collection
        coll = Collection(name=self.entity_collection_name, using="default")
        data = [[entity_id], [entity_type], [name], [vector]]
        coll.upsert(data)
        coll.flush()  # pymilvus 3.x upsert 不自动 flush,不 flush 则 search 看不到新向量

    def search_entities(self, vector: list, entity_type: str, topk: int) -> list[tuple]:
        """实体名向量检索(类型作用域),返 [(entity_id, name, score)] 按相似度降序。"""
        from pymilvus import Collection
        coll = Collection(name=self.entity_collection_name, using="default")
        coll.load()
        params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        expr = f'{F_ENTITY_TYPE} == "{entity_type}"'
        res = coll.search([vector], F_NAME_VECTOR, params, limit=topk, expr=expr,
                          output_fields=[F_ENTITY_ID, F_ENTITY_NAME, F_ENTITY_TYPE])
        out: list[tuple] = []
        for hit in res[0]:
            fields = dict(hit.fields)
            out.append((fields[F_ENTITY_ID], fields[F_ENTITY_NAME], float(hit.distance)))
        return out

    def delete_entity_vector(self, entity_id: str) -> None:
        from pymilvus import Collection
        coll = Collection(name=self.entity_collection_name, using="default")
        coll.load()
        coll.delete(expr=f'{F_ENTITY_ID} == "{entity_id}"')
        coll.flush()  # 删除不 flush 则 search 仍命中已删向量,破坏对账幂等

    def remap_kg_entity_ids(self, old: str, new: str) -> int:
        """全量对账:ems_chunks.kg_entity_ids 数组内 old -> new 重映射,返更新条数。"""
        rows = self.query(f'array_contains({F_KG_ENTITY_IDS}, "{old}")',
                          output_fields=FIELD_NAMES)
        count = 0
        for row in rows:
            ids = list(row.get(F_KG_ENTITY_IDS) or [])
            new_ids: list[str] = []
            for x in ids:
                x2 = new if x == old else x
                if x2 not in new_ids:
                    new_ids.append(x2)
            self.update_kg_entity_ids(row[F_CHILD_ID], new_ids)
            count += 1
        return count


class AsyncMilvusClient:
    """async 门面:pymilvus 同步调用经 asyncio.to_thread 委托(唯一同步孤岛)。"""

    def __init__(self, settings: Optional[dict] = None):
        self._sync = MilvusClient(settings)
        self._connected = False

    async def _ensure(self) -> None:
        if not self._connected:
            await asyncio.to_thread(self._sync.connect)
            self._connected = True

    @property
    def collection_name(self) -> str:
        return self._sync.collection_name

    @property
    def entity_collection_name(self) -> str:
        return self._sync.entity_collection_name

    async def create_collection(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.create_collection)

    async def upsert(self, rows: list[dict]) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.upsert, rows)

    async def query(self, expr: str, output_fields: Optional[list] = None, limit: int = 1000) -> list[dict]:
        await self._ensure()
        return await asyncio.to_thread(self._sync.query, expr, output_fields, limit)

    async def search(self, vector: list, topk: int, expr: Optional[str] = None) -> list[dict]:
        await self._ensure()
        return await asyncio.to_thread(self._sync.search, vector, topk, expr)

    async def count(self) -> int:
        await self._ensure()
        return await asyncio.to_thread(self._sync.count)

    async def update_kg_entity_ids(self, child_id: str, ids: list) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.update_kg_entity_ids, child_id, ids)

    async def update_fields(self, child_id: str, **fields) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.update_fields, child_id, **fields)

    async def delete(self, ids: list[str]) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.delete, ids)

    # ---- 实体向量库门面 ----
    async def create_entity_collection(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.create_entity_collection)

    async def upsert_entity_vector(self, entity_id: str, name: str, entity_type: str,
                                   vector: list) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.upsert_entity_vector, entity_id, name, entity_type, vector)

    async def search_entities(self, vector: list, entity_type: str, topk: int) -> list[tuple]:
        await self._ensure()
        return await asyncio.to_thread(self._sync.search_entities, vector, entity_type, topk)

    async def delete_entity_vector(self, entity_id: str) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.delete_entity_vector, entity_id)

    async def remap_kg_entity_ids(self, old: str, new: str) -> int:
        await self._ensure()
        return await asyncio.to_thread(self._sync.remap_kg_entity_ids, old, new)

    async def flush(self, name: Optional[str] = None) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.flush, name)

    async def drop_collection(self, name: str) -> None:
        await self._ensure()
        await asyncio.to_thread(self._sync.drop_collection, name)


def get_milvus() -> MilvusClient:
    return MilvusClient()


def get_async_milvus() -> AsyncMilvusClient:
    return AsyncMilvusClient()
