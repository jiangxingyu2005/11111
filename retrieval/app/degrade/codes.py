"""降级码字典（QRY-07 / M-D1 的唯一事实源）。

对应 V2.0.1 §6 降级矩阵 18 条 + 附录 A 预留 1 条。
每条注入测试必须同时验证：行为 + degraded[] 留痕 + 前端角标（G-QC-03）。
"""
from __future__ import annotations

from enum import Enum


class DegradedCode(str, Enum):
    # 前置层
    FAQ_UNAVAILABLE = "faq_unavailable"                    # 1  Redis 不可用，跳过 FAQ 短路
    NL2QUERY_LLM_TIMEOUT = "nl2query_llm_timeout"          # 2  L2 超时，只用 L1
    CONDITION_INVALID = "condition_invalid"                # 3  L3 校验失败，条件降 soft/丢弃

    # 召回层
    VECTOR_UNAVAILABLE = "vector_unavailable"              # 4  Milvus 熔断，剥离①路
    SPARSE_UNAVAILABLE = "sparse_unavailable"              # 5  sparse 索引缺失（阶段二前常态）
    BM25_UNAVAILABLE = "bm25_unavailable"                  # 18 ② 路熔断（启用后）
    GRAPH_UNAVAILABLE = "graph_unavailable"                # 7  Neo4j 熔断，剥离③路

    # 社区摘要路（IF-4 deferred，预留码位）
    COMMUNITY_UNAVAILABLE = "community_unavailable"        # 8
    COMMUNITY_NO_START = "community_no_start"              # 9
    COMMUNITY_STALE = "community_stale"                    # 10
    COMMUNITY_SUMMARY_PARTIAL = "community_summary_partial"  # 11
    COMMUNITY_NO_EVIDENCE = "community_no_evidence"        # 12

    # ④ 标量约束路（附录 A 预留，当前版本不生效）
    SCALAR_FILTER_FALLBACK = "scalar_filter_fallback"      # 13（预留）

    # 融合与重排
    RERANK_TIMEOUT = "rerank_timeout"                      # 6  RRF 序直出
    RERANK_SKIPPED = "rerank_skipped"                      # 6b 预算 L3：主动跳过精排

    # 预算看门狗
    BUDGET_CUT_COMMUNITY = "budget_cut_community"          # 14 >4.5s 砍社区保遍历
    BUDGET_CUT_ROUTE3 = "budget_cut_route3"                # 15 >3.5s 且①已出，砍③路

    # 回捞与生成
    PARENT_FETCH_UNAVAILABLE = "parent_fetch_unavailable"  # 17 PG 熔断，行内 content 兜底
    GENERATION_UNAVAILABLE = "generation_unavailable"      # 16 LLM 熔断，命中列表+证据直出

    # 证据质量标记（非整路剥离，hit 级标记）
    PARENT_FETCH_FAILED = "parent_fetch_failed"            # 单条父块回捞失败/版本漂移
    UNIT_INFERRED = "unit_inferred"                        # L1/L2 裸数字按基准单位解释
    EVIDENCE_SPAN_MISSING = "evidence_span_missing"        # IF-2 未就位：证据无字符区间，前端不渲染高亮跳转
    ARTIFACT_FAILED = "artifact_failed"                    # M-R7 中间产物生成失败（禁占位图）


# ────────────────────────────────────────────────────────────
# 降级码 -> 契约 step（api/retrieval_response.v1.schema.json 的枚举）
# 契约 step 是必填枚举，此处是唯一映射源，禁止在各模块里各写一套
# ────────────────────────────────────────────────────────────
STEP_MAP: dict[str, str] = {
    "faq_unavailable": "faq",
    "nl2query_llm_timeout": "query_parse",
    "condition_invalid": "query_parse",
    "unit_inferred": "query_parse",
    "vector_unavailable": "vector",
    "sparse_unavailable": "sparse",
    "bm25_unavailable": "vector",
    "graph_unavailable": "graph",
    "community_unavailable": "graph",
    "community_no_start": "graph",
    "community_stale": "graph",
    "community_summary_partial": "graph",
    "community_no_evidence": "graph",
    "scalar_filter_fallback": "scalar_filter",
    "rerank_timeout": "rerank",
    "rerank_skipped": "rerank",
    "budget_cut_community": "graph",
    "budget_cut_route3": "graph",
    "parent_fetch_unavailable": "pg_fetch",
    "parent_fetch_failed": "pg_fetch",
    "generation_unavailable": "generate",
    "evidence_span_missing": "vector",
    "artifact_failed": "asset",
}


def step_of(code: str) -> str:
    return STEP_MAP.get(code, "vector")


# 预算等级（config/retrieval.yaml: degradation.cutoffs 为准）
class BudgetLevel(str, Enum):
    L0 = "L0"   # >5.2s 转 PG 全文兜底
    L1 = "L1"   # >5.0s 砍 NL2Query + sparse
    L2 = "L2"   # >4.5s 砍社区摘要保遍历
    L3 = "L3"   # >3.5s 砍重排
    L4 = "L4"   # 正常全链路
