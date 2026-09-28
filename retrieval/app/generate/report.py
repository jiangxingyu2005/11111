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


def build_prompt(query: str, hits: list[Hit], parents: dict[str, dict]) -> list[dict]:
    """组装 messages：父块全文进 context，子块给编号 [n]。"""
    lines = []
    for i, h in enumerate(hits, start=1):
        p = parents.get(h.parent_id, {})
        ctx = p.get("content") or h.content
        lines.append(f"[{i}] 来源 kb={h.meta.get('kb_id','')} child_id={h.child_id}\n{ctx}")
    evidence = "\n\n".join(lines) if lines else "（无命中证据）"
    user = (
        f"用户问题：{query}\n\n"
        f"证据（共 {len(hits)} 条，编号 [1]~[{len(hits)}]）：\n{evidence}\n\n"
        "请按四要素输出：1) 相似信号列表（带 [n]）2) 关联路径 3) 规则触发详情 "
        "4) 置信度与依据不足说明。结论必须由证据支撑。"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


async def stream_report(
    query: str,
    hits: list[Hit],
    parents: dict[str, dict],
    collector: DegradedCollector,
) -> AsyncIterator[str]:
    """yield 报告 token（真实流式）；任何失败 -> 记录降级后静默结束（不出正文）。"""
    s = get_settings()
    if not s.graphrag_llm_base_url:
        collector.record(DegradedCode.GENERATION_UNAVAILABLE,
                         "LLM endpoint not configured", "generate")
        return

    messages = build_prompt(query, hits, parents)
    url = s.graphrag_llm_base_url.rstrip("/") + "/chat/completions"
    try:
        from ..store.fake import guard
        guard("llm")          # 故障注入（跑降级矩阵用）
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as cli:
            async with cli.stream(
                "POST", url,
                headers={"Authorization": f"Bearer {s.graphrag_llm_api_key}"},
                json={"model": s.graphrag_llm_model, "messages": messages,
                      "stream": True, "temperature": 0.2},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    tok = delta.get("content")
                    if tok:
                        yield tok
    except Exception as e:
        log.warning("generate failed: %s", e)
        collector.record(DegradedCode.GENERATION_UNAVAILABLE, str(e)[:200], "generate")
