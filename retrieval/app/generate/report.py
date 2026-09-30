"""RPT 可解释报告生成（M-R2）——真流式（OpenAI 兼容 /v1/chat/completions）。

契约：
  - 逐 delta 流式 yield，上层转 SSE token 事件
  - 证据 [n] 引用编号与 results 顺序严格一致（前端点 [n] 要能跳到对应命中）
  - 四要素：相似信号列表 / 关联路径 / 规则触发详情 / 置信度
红线：
  - 社区摘要、LLM 生成内容【无 span，不得当证据】，不进 RAGAS context
  - LLM 熔断 -> 不出正文（命中列表已在 results 事件给出）+ generation_unavailable
  - 严禁为"看起来能跑"而编造正文：无 LLM 就是无正文，这是明确降级不是缺陷
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from ..config import get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import DegradedCollector
from ..fuse.rrf import Hit

log = logging.getLogger("retrieval.generate")

SYSTEM = (
    "你是电磁信号情报分析助手。只能依据【证据】作答，禁止推测补全；"
    "每条结论必须带证据编号 [n]，编号与给定证据顺序一致；"
    "证据未覆盖的内容必须明说'依据不足'，不得编造参数值。"
)


def build_prompt(
    query: str,
    hits: list[Hit],
    parents: dict[str, dict],
    rule_hits: list[dict] | None = None,
    memories: list[str] | None = None,
    summary: str | None = None,
    window: list[dict] | None = None,
) -> list[dict]:
    """组装 messages：父块全文进 context，子块给编号 [n]；附置信度 + 规则触发情况。

    置信度（需求(8)「置信度评分」）：每条证据行尾标 kg_confidence / rerank / rrf，
    三者语义不同（实体抽取置信 / 检索匹配度 / 融合票分），不混算，只并列供参考。
    规则触发（需求(7)/(8)「规则触发详情」）：rule_hits 由确定性规则引擎算好注入，
    LLM 只转述，不得自行判定规则是否命中。
    历史上下文（记忆管理）：summary/window/memories 为连续性上下文，非证据；
    结论仍须由证据 [n] 支撑，历史只用于理解指代与背景。
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        p = parents.get(h.parent_id, {})
        ctx = p.get("content") or h.content
        conf = []
        kg = p.get("kg_confidence")
        if kg is not None:
            conf.append(f"kg_confidence={kg:.3f}")
        if h.meta.get("rerank_score") is not None:
            conf.append(f"rerank={h.meta.get('rerank_score'):.3f}")
        if h.meta.get("rrf_score") is not None:
            conf.append(f"rrf={h.meta.get('rrf_score'):.4f}")
        conf_s = f"（置信度：{'，'.join(conf)}）" if conf else ""
        lines.append(f"[{i}] 来源 kb={h.meta.get('kb_id','')} child_id={h.child_id}{conf_s}\n{ctx}")
    evidence = "\n\n".join(lines) if lines else "（无命中证据）"

    rule_block = ""
    if rule_hits:
        rule_lines = []
        for r in rule_hits:
            hint = f"；{r['hint']}" if r.get("hint") else ""
            rule_lines.append(
                f"- 【{r['label']}】规则「{r['name']}」（{r['rule_id']}，severity={r['severity']}）"
                f"已触发，{r['reason']}{hint}"
            )
        rule_block = "预设规则触发情况：\n" + "\n".join(rule_lines)

    history_block = ""
    hist_parts = []
    if summary:
        hist_parts.append(f"【会话摘要】{summary}")
    if window:
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in window)
        hist_parts.append(f"【近期对话】\n{turns}")
    if memories:
        hist_parts.append("【长期记忆事实】\n" + "\n".join(f"- {m}" for m in memories))
    if hist_parts:
        history_block = "历史上下文：\n" + "\n\n".join(hist_parts)

    user = (
        f"用户问题：{query}\n\n"
        f"证据（共 {len(hits)} 条，编号 [1]~[{len(hits)}]）：\n{evidence}\n\n"
    )
    if history_block:
        user += f"{history_block}\n\n"
    if rule_block:
        user += f"{rule_block}\n\n"
    user += (
        "请按四要素输出：1) 相似信号列表（带 [n]）2) 关联路径 3) 规则触发详情 "
        "4) 置信度与依据不足说明。结论必须由证据支撑；"
        "历史上下文仅用于理解问题背景与指代，非证据；"
        "规则触发情况与置信度仅供参考，仍须以证据为准，禁止编造参数值。"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


async def stream_report(
    query: str,
    hits: list[Hit],
    parents: dict[str, dict],
    collector: DegradedCollector,
    rule_hits: list[dict] | None = None,
    memories: list[str] | None = None,
    summary: str | None = None,
    window: list[dict] | None = None,
) -> AsyncIterator[str]:
    """yield 报告 token（真实流式）；任何失败 -> 记录降级后静默结束（不出正文）。"""
    s = get_settings()
    if not s.graphrag_llm_base_url:
        collector.record(DegradedCode.GENERATION_UNAVAILABLE,
                         "LLM endpoint not configured", "generate")
        return

    messages = build_prompt(query, hits, parents, rule_hits, memories, summary, window)
    try:
        from ..store.fake import guard
        guard("llm")          # 故障注入（跑降级矩阵用）
        from ..llm import chat_stream
        async for tok in chat_stream(messages, temperature=0.2):
            yield tok
    except Exception as e:
        log.warning("generate failed: %s", e)
        collector.record(DegradedCode.GENERATION_UNAVAILABLE, str(e)[:200], "generate")
