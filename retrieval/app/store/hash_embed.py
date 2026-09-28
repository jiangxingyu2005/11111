"""确定性哈希嵌入（dev/fake 通道专用）——不依赖 GPU、不依赖模型权重。

用途与边界（重要）：
  - 仅用于【本地无模型环境】跑通全链路（seed 建库 + fake 后端检索 + 契约测试）
  - 生产/验收必须用 BGE-M3：RETRIEVAL_EMBED_MODE=local 或 embed 服务 EMBED_BACKEND=bge_m3
  - 同一份算法同时被 ① 离线 seed 建库 ② 在线 query 向量化 使用，
    保证 fake 通道下 query 与 doc 在同一向量空间（否则检索结果无意义）
  - 语义能力仅限"字面/词面重叠"，不得用它评估召回率（M-D5 消融必须换真模型）

算法：hashing trick + 符号哈希 + sublinear TF + L2 归一（与 BGE-M3 同为 1024 维 COSINE）
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

DIM = 1024                       # 与 BGE-M3 dense 维度对齐
_WORD = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中文按字 + 相邻二元组（保留部分词序信息），英数按词并小写。"""
    text = text or ""
    toks: list[str] = []
    for m in _WORD.finditer(text):
        toks.append("w:" + m.group(0).lower())
    # 中文：单字 + bigram
    buf: list[str] = []
    for ch in text:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            if buf:
                toks.extend("c:" + c for c in buf)
                toks.extend("b:" + buf[i] + buf[i + 1] for i in range(len(buf) - 1))
                buf = []
    if buf:
        toks.extend("c:" + c for c in buf)
        toks.extend("b:" + buf[i] + buf[i + 1] for i in range(len(buf) - 1))
    return toks


def _h(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def hash_embedding(text: str, dim: int = DIM) -> list[float]:
    """text -> L2 归一化 dense 向量（长度 dim）。"""
    vec = [0.0] * dim
    tf = Counter(tokenize(text))
    for tok, n in tf.items():
        h = _h(tok)
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(n))          # sublinear TF
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def hash_sparse(text: str, dim: int = 30000) -> dict[int, float]:
    """text -> Milvus SPARSE_FLOAT_VECTOR 形态 {token_id: weight}（模拟 lexical_weights）。

    权重 = sublinear TF，不做 IDF（IDF 需要全量语料统计，fake 通道不需要）。
    """
    out: dict[int, float] = {}
    tf = Counter(tokenize(text))
    for tok, n in tf.items():
        out[_h(tok) % dim] = round(1.0 + math.log(n), 4)
    return out


def cosine(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度（两者均已归一化时即点积）。"""
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


def lex_overlap(query_sparse: dict[int, float], doc_sparse: dict[int, float]) -> float:
    """稀疏打分（模拟 BM25 的内积排序效果）：Σ q_w * d_w。"""
    if not query_sparse or not doc_sparse:
        return 0.0
    small, large = (query_sparse, doc_sparse) if len(query_sparse) < len(doc_sparse) else (doc_sparse, query_sparse)
    return sum(w * large.get(k, 0.0) for k, w in small.items())


# ────────────────────────────────────────────────────────────
# 数值/单位归一化信号参数嵌入（①路 B 副向量，real/fake 通用）
# ------------------------------------------------------------
# 与 A 路（字面哈希）正交：A 路捕字面/词面重叠，B 路捕「信号参数语义」。
# 载频/脉宽/重频/方位/信噪比五要素 -> 归一基准单位 -> 对数/线性分桶 -> 软 one-hot -> L2 归一。
# 效果：9GHz ≡ 9000MHz（同桶全等）、9.0 vs 9.2GHz（邻桶部分相似）、参数无关文档（近正交）。
# 这是【确定性算法，非占位】：real 通道同样用它产出 embedding_v2（不依赖模型/GPU）。
# ────────────────────────────────────────────────────────────
NUM_DIM = 256                       # B 路维度（独立于 A 路 DIM=1024）
_NUM_FIELDS = ["rf", "pw", "prf", "doa", "snr"]
_NUM_BUCKETS = 50                   # 每参数 50 桶；5 参数 x 50 = 250 <= 256
# (min, max, use_log) 与 app/understand/query_parser.py 的 RANGE_LIMITS 对齐（同一量纲口径）
_NUM_RANGES = {
    "rf":  (0.01, 300.0, True),      # GHz
    "pw":  (0.01, 10000.0, True),    # us
    "prf": (1.0, 2_000_000.0, True), # Hz
    "doa": (0.0, 360.0, False),      # deg
    "snr": (-40.0, 80.0, False),     # dB
}


def hash_numeric_embedding(params: dict[str, float], dim: int = NUM_DIM) -> list[float]:
    """把已归一化到基准单位的信号参数 {rf,pw,prf,doa,snr} 转成 B 路向量。

    params 值为 None / 键缺失 -> 该参数不占桶；空入参 -> 零向量（调用方视为 B 路缺席）。
    """
    vec = [0.0] * dim
    for i, f in enumerate(_NUM_FIELDS):
        v = params.get(f)
        if v is None:
            continue
        lo, hi, use_log = _NUM_RANGES[f]
        if use_log:
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(max(float(v), 1e-9))
        pos = (float(v) - lo) / (hi - lo)
        pos = min(max(pos, 0.0), 1.0 - 1e-9)
        bucket = int(pos * _NUM_BUCKETS)
        base = i * _NUM_BUCKETS + bucket
        # 软 one-hot：主桶 1.0、相邻桶 0.5（相邻参数值保留部分相似，而非硬正交）
        vec[base] = max(vec[base], 1.0)
        if bucket > 0:
            vec[base - 1] = max(vec[base - 1], 0.5)
        if bucket < _NUM_BUCKETS - 1:
            vec[base + 1] = max(vec[base + 1], 0.5)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]
