"""GraphRAG(ems_graphrag 融合)子包。

把 ems_graphrag 的「清洗 md → 父/子块切分 → Milvus 向量 → Neo4j 图谱」落地进 qingxi。
算法/纯逻辑尽量原样移植;I/O 层改为异步(PG=SQLAlchemy async,Neo4j=neo4j.asyncio,
Milvus=pymilvus 经 asyncio.to_thread 委托——无官方 async 客户端)。
配置不走 yaml:基础设施/阈值读 qingxi `.env`,实体/关系字典读 extraction_dict 表。
"""
