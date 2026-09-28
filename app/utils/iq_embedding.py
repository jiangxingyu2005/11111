"""IQ 嵌入:用 WavesFM 重导出的 256 维 CLS 特征模型,把归一化 IQ 片段编码为向量。

模型 = model/iq/wavesfm_embed.onnx(从 dc `wavesfm_rml.onnx` 重导出,新增 256 维 `select` 输出)。
同步 CPU 密集逻辑,由 API 层丢 run_in_threadpool 执行;onnxruntime 会话进程级缓存。
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from ..config import settings

_EMBED_OUTPUT = "select"   # 重导出时固定的 256 维 CLS 特征输出名
_onnx_session = None


def _load_model():
    global _onnx_session
    if _onnx_session is not None:
        return _onnx_session
    import onnxruntime as ort

    model_path = os.path.realpath(settings.iq_embedding_model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"IQ 嵌入模型不存在: {model_path}")
    _onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _onnx_session


def _run(iq_batch: np.ndarray) -> np.ndarray:
    """(N,128,2) float32 -> (N,256),不做归一化(由调用方统一 L2 归一)。"""
    session = _load_model()
    x = iq_batch.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (N,128,2) -> (N,2,1,128)
    input_name = session.get_inputs()[0].name
    return session.run([_EMBED_OUTPUT], {input_name: x})[0]


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return emb / norm


def iq_batch_to_embedding(iq_batch: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """批量嵌入:(N,128,2) -> (N,256),L2 归一化。"""
    iq_batch = np.asarray(iq_batch, dtype=np.float32)
    if iq_batch.ndim != 3 or iq_batch.shape[1:] != (128, 2):
        raise ValueError(f"IQ 批量形状应为 (N,128,2),实际为 {iq_batch.shape}")
    all_emb = []
    for start in range(0, len(iq_batch), batch_size):
        all_emb.append(_run(iq_batch[start:start + batch_size]))
    return _l2_normalize(np.concatenate(all_emb, axis=0))


def iq_to_embedding(iq: np.ndarray) -> np.ndarray:
    """单段嵌入:(128,2) 或 (2,128) -> (256,),L2 归一化。"""
    iq = np.asarray(iq, dtype=np.float32)
    if iq.shape == (128, 2):
        iq = iq.T  # (2,128)
    elif iq.shape != (2, 128):
        raise ValueError(f"IQ 形状应为 (128,2) 或 (2,128),实际为 {iq.shape}")
    x = iq[np.newaxis, :, np.newaxis, :]  # (1,2,1,128)
    session = _load_model()
    input_name = session.get_inputs()[0].name
    emb = session.run([_EMBED_OUTPUT], {input_name: x})[0]  # (1,256)
    return _l2_normalize(emb)[0]


def embed_segments(segments: list[dict]) -> np.ndarray:
    """归一化片段列表 [{I:[...],Q:[...]},...] -> 单个 (256,) 文档向量(逐段嵌入后 mean-pool)。"""
    if not segments:
        raise ValueError("归一化后无有效 128 点片段,无法嵌入")
    iq_batch = np.asarray(
        [np.column_stack([s["I"], s["Q"]]) for s in segments], dtype=np.float32
    )  # (N,128,2)
    embs = iq_batch_to_embedding(iq_batch)  # (N,256)
    emb = embs.mean(axis=0)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb


def embed_document(h5_content: bytes) -> np.ndarray:
    """整段 IQ 信号 -> 单个 (256,) 文档向量(内部重跑 IQ 管道取归一化片段,供检索用)。"""
    from .iq import process_iq_h5

    result = process_iq_h5(h5_content)
    return embed_segments(result.get("iq_norm") or [])
