"""GP-12 QRY-02:NL2Query(自然语言 → 结构化查询条件,条款(6))。

规则式意图理解(离线可验证、无外部依赖):关键词 → 调制方式 / 频段等结构化条件。
示例:"跳频信号" → {modulation_type: "FSK"}。纯 `re`,无 I/O,保持同步。
"""
from __future__ import annotations

import re

# 调制方式关键词 → 枚举值(封闭词表,可扩展)
_MODULATION_MAP: list[tuple[tuple[str, ...], str]] = [
    (("qpsk",), "QPSK"),
    (("bpsk",), "BPSK"),
    (("8psk",), "8PSK"),
    (("跳频", "fhss"), "跳频"),
    (("相移键控", "psk"), "PSK"),
    (("正交幅度", "qam"), "QAM"),
    (("线性调频", "lfm"), "LFM"),
    (("调频", "fm"), "FM"),
    (("调幅", "am"), "AM"),
]

_FREQ_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:GHz|MHz|kHz|KHz)", re.IGNORECASE)


def nl2query(query: str) -> dict:
    """返回 {intent, modulation_type, freq_band, raw}。"""
    q = query.lower()
    modulation = None
    for kws, mod in _MODULATION_MAP:
        if any(kw in q for kw in kws):
            modulation = mod
            break

    freq_band = None
    m = _FREQ_RE.search(query)
    if m:
        freq_band = m.group(0)

    return {
        "intent": "local_fact",
        "modulation_type": modulation,
        "freq_band": freq_band,
        "raw": query,
    }
