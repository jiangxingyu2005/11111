"""retrieval 测评模块（EVAL 子系统，EV-1 回放 + 进程内直连两种模式）。

职责：
  从 date/ 数据（经 ingest_date.py 产出的 data/seed_date/）构造客观金标准集，
  跑在线检索链路，产出需求文档两条验收硬指标——召回率 + 响应时间，
  并渲染 JSON / Markdown / HTML 三份报告（HTML 含逐 query Top5 命中直观展示）。

用法：
  cd retrieval
  python -m eval --mode in_process                     # 进程内直连（默认，当下主用）
  python -m eval --mode http --target-url http://127.0.0.1:8100   # 端到端回放（EV-1）

说明：
  - 当前未接 LLM 生成，响应时间口径为检索侧（对比预算 5.2s）；11.5s 首 token 验收需接生成后测。
  - RAGAS 判分、EV-2 三档被引口径、EV-3 中间结果消融为后续项，本模块暂不实现。
"""
__version__ = "1.0.0"
