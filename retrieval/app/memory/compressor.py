"""压缩流水线：token 触发 + 递归摘要 + 封顶 + 事实兜底（fail-closed）。

策略（已对齐）：
  - 按 token 用量触发（记忆区占用 ≥ 预算 × trigger_ratio），不按条数
  - 压缩次数封顶 max_compressions，压满后只做窗口裁剪不再摘要
  - 丢弃原文前【先把事实 add 进 mem0】（长期事实兜底，事实不丢）
fail-closed：任何 LLM/mem0/DB 失败 -> 保留原文、放弃本次压缩，绝不影响检索主链。
"""
from __future__ import annotations

import logging
import re

from ..config import get_settings

log = logging.getLogger("retrieval.memory.compressor")

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z0-9]+")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算：中文≈1 token/字，英数≈1 token/词，其余字符≈0.5。

    不引 tiktoken/transformers 重依赖；与真实 tokenizer 有偏差，budget 已留 20% 裕量。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    rest = len(re.sub(r"[一-鿿A-Za-z0-9]+", "", text))
    return cjk + words + max(0, rest // 2)


async def recursive_summarize(prev_summary: str, turns: list[dict]) -> str:
    """把「旧摘要 + 新增轮次」压缩成新摘要（调本地 LLM，OpenAI 兼容 /chat/completions）。

    失败抛异常，由调用方 fail-closed（保留旧摘要/原文）。
    """
    s = get_settings()
    if not s.graphrag_llm_base_url or not s.graphrag_llm_model:
        raise RuntimeError("LLM 未配置，无法压缩摘要")
    from ..store.fake import guard

    guard("llm")
    import httpx

    user_text = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns)
    messages = [
        {"role": "system",
         "content": "你是对话记忆压缩器。把【既有摘要】与【新增对话】合并成一段更紧凑的摘要，"
                    "保留关键事实、用户关注点、约束与结论，丢弃寒暄与冗余。"},
        {"role": "user",
         "content": f"既有摘要：{prev_summary or '(空)'}\n\n新增对话：\n{user_text}\n\n"
                    "请输出压缩后的摘要（只输出摘要正文，不要解释）。"},
    ]
    url = s.graphrag_llm_base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as cli:
        r = await cli.post(
            url,
            headers={"Authorization": f"Bearer {s.graphrag_llm_api_key}"},
            json={"model": s.graphrag_llm_model, "messages": messages,
                  "stream": False, "temperature": 0.2},
        )
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


async def maybe_compress(store, memory, conv: dict, cfg: dict) -> None:
    """压缩流水线入口（后台调用）。

    conv: get_or_create_conversation 返回的会话字典（含 id/user_id/summary/compress_count）。
    """
    try:
        window_size = int((cfg or {}).get("window_size", 10))
        budget = int((cfg or {}).get("budget_tokens", 4096))
        trigger = float((cfg or {}).get("trigger_ratio", 0.8))
        max_comp = int((cfg or {}).get("max_compressions", 5))

        window = await store.load_window(conv["id"], 10_000)
        tokens = estimate_tokens(conv.get("summary", "")) + sum(
            estimate_tokens(m["content"]) for m in window)
        if tokens < budget * trigger:
            return  # 未到阈值，不压

        # 确定保留条数 n_keep，溢出 overflow = 最旧 (len-n_keep) 条：
        # 窗口已满 -> 留 window_size 条；未满但超 token（单条超长）-> 挤最旧 1 条保证有进展。
        # 关键：trim 必须用 n_keep（= 实际要保留的条数），不能用 window_size，否则窗口未满时
        # overflow 挤出的消息不会被真正删除（trim(window_size) > 当前条数 = 不删），导致重复抽取/反复触发。
        if len(window) > window_size:
            n_keep = window_size
        elif len(window) >= 2:
            n_keep = len(window) - 1
        else:
            return
        overflow = window[: len(window) - n_keep]

        # ① 先把被挤出的原文抽进 mem0（失败则整体放弃、保留原文——「事实不丢」的前置条件）
        await memory.add(conv["user_id"], overflow)

        # ② 递归摘要（best-effort，受压缩次数封顶；失败保留旧摘要，不阻断裁剪）
        if conv.get("compress_count", 0) < max_comp:
            try:
                summary = await recursive_summarize(conv.get("summary", ""), overflow)
                await store.update_summary(conv["id"], summary, conv.get("compress_count", 0) + 1)
            except Exception as e:
                log.warning("summary failed (保留旧摘要): %s", e)

        # ③ 裁剪窗口到 n_keep 条（事实已进 mem0，可安全丢弃原文）
        await store.trim_messages(conv["id"], n_keep)
    except Exception as e:
        log.warning("memory compress skipped (fail-closed，保留原文): %s", e)
