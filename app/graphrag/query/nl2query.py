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
# 长单位优先,避免 500km 被收成 500k + m,或 2.4GHz 收成 2.4g + hz
_QTY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[-~～至到]\s*(\d+(?:\.\d+)?))?\s*"
    r"(ghz|mhz|khz|km|kg|kw|kn|ma|hz|t|m)(?![A-Za-z])",
    re.IGNORECASE,
)
_UNIT_FAM = {
    "ghz": ("hz", 1e9), "mhz": ("hz", 1e6), "khz": ("hz", 1e3), "hz": ("hz", 1.0),
    "km": ("km", 1.0), "m": ("m", 1.0),
    "kn": ("kn", 1.0), "ma": ("ma", 1.0),
    "t": ("t", 1.0), "kg": ("kg", 1.0), "kw": ("kw", 1.0),
}


_GAP_RE = re.compile(r"^\s*[-~～至到—－]+\s*$")


def _qty_of(m: re.Match) -> tuple[float, float, str]:
    fam, mul = _UNIT_FAM[m.group(3).lower()]
    lo = float(m.group(1)) * mul
    hi = float(m.group(2)) * mul if m.group(2) else lo
    return (min(lo, hi), max(lo, hi), fam)


def parse_quantities(text: str) -> list[tuple[float, float, str]]:
    """正文/问句 → [(下限, 上限, 量纲族)]。

    同一量纲、中间只有 到/-/~ 的两个带单位数字合成一个区间:
    500km到1000km、2.4GHz-3GHz、2400MHz到3GHz。
    """
    text = text or ""
    ms = list(_QTY_RE.finditer(text))
    out: list[tuple[float, float, str]] = []
    i = 0
    while i < len(ms):
        a = _qty_of(ms[i])
        if i + 1 < len(ms):
            gap = text[ms[i].end():ms[i + 1].start()]
            b = _qty_of(ms[i + 1])
            if a[2] == b[2] and _GAP_RE.match(gap):
                out.append((min(a[0], b[0]), max(a[1], b[1]), a[2]))
                i += 2
                continue
        out.append(a)
        i += 1
    return out



def nl2query(query: str) -> dict:
    """返回 {intent, modulation_type, freq_band, quantities, raw}。"""
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
        "quantities": parse_quantities(query),
        "raw": query,
    }
