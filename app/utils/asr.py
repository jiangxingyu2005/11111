"""音频转文字(ASR):faster-whisper 本地模型。

模型单独放在项目内 model/ 文件夹(见 config.asr_model_dir),需先运行
`python scripts/download_asr_model.py` 下载。转写是同步、CPU 密集操作,
由 API 层丢 run_in_threadpool 调用。
"""

import os
import tempfile
from pathlib import Path

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


def transcribe_audio(content: bytes, filename: str) -> str:
    """音频字节 -> 文字(全文拼接)。

    写带原扩展名的临时文件让 PyAV 识别容器格式,识别后清理临时文件。
    识别失败抛异常,由上层兜底。
    """
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
