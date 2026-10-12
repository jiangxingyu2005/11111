"""音频转文字(ASR):faster-whisper 本地模型。

模型单独放在项目内 model/ 文件夹(见 config.asr_model_dir),需先运行
`python scripts/download_asr_model.py` 下载。转写是同步、CPU 密集操作,
由 API 层丢 run_in_threadpool 调用。
"""

import os
import tempfile
from pathlib import Path

import av
from faster_whisper import WhisperModel

from ..config import settings

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """懒加载 faster-whisper 模型单例,从本地 model/ 目录加载。"""
    global _model
    if _model is None:
        model_dir = Path(settings.asr_model_dir)
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(
                f"ASR 模型未下载:{model_dir},请先运行 python scripts/download_asr_model.py"
            )
        _model = WhisperModel(
            str(model_dir),
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
    return _model


def validate_audio(content: bytes, filename: str) -> None:
    """音频完整性校验:不过抛清晰 ValueError,不让损坏/不完整文件进转写管道。

    用 PyAV(av,faster-whisper 的既有依赖)探测容器:空文件 / 无法解码 / 无音频流 /
    采样率无效均拒收。时长不做硬性判定(不少合法容器不写 duration 元数据)。
    """
    if not content:
        raise ValueError("音频文件为空,无法解析")
    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        try:
            container = av.open(tmp.name)
        except Exception as e:
            raise ValueError("音频文件无法解码,可能已损坏或不是音频") from e
        try:
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("音频文件无音频流")
            rate = getattr(audio_stream, "rate", None) or getattr(audio_stream, "sample_rate", None)
            if not rate or rate <= 0:
                raise ValueError("音频采样率无效")
        finally:
            container.close()
    finally:
        os.unlink(tmp.name)


def transcribe_audio(content: bytes, filename: str) -> str:
    """音频字节 -> 文字(全文拼接)。

    写带原扩展名的临时文件让 PyAV 识别容器格式,识别后清理临时文件。
    识别失败抛异常,由上层兜底。
    """
    validate_audio(content, filename)
    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.close()
        segments, _ = _get_model().transcribe(
            tmp.name,
            language=settings.asr_language or None,
        )
        return "".join(seg.text for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
