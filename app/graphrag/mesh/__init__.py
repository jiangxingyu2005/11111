"""GP-07/GP-12 MESH:归一化与连通性保障(ems mesh/ 的忠实移植,纯逻辑无 I/O)。

- registry.py  实体注册表(别名 + 属性指纹双路归一)
- fallback.py  孤儿父块兜底绑定
- derived.py   元数据派生边(SAME_BAND/CO_TIME/CO_LOCATE)
- fusion.py    retired 引用计数 + 四指标门禁
- apply.py     入库「对齐」入口(归一消歧 + 派生边 + 保真绑定)
"""
