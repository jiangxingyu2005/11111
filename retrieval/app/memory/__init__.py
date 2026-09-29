"""记忆管理子包：多人隔离 + 会话多轮 + 长期事实（mem0）。

三层：
  - window（最近 N 轮原文）     —— 会话级，PG messages / 内存 dict
  - summary（递归压缩叙事）    —— 会话级，PG conversations.summary / 内存 dict
  - facts（mem0 长期事实）     —— 用户级，mem0+Milvus / 内存 dict

纪律（与全项目一致）：
  - fail-closed：记忆读写/压缩失败只记 warning，绝不影响检索主链、不留 degraded
  - fake/real 分发：fake=内存实现，real=PG/mem0，route 层只认接口不认实现
"""
