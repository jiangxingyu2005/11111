"""预下载 faster-whisper 模型到项目内 model/ 文件夹。

用法:python scripts/download_asr_model.py
"""

import os
import sys
from pathlib import Path

# 国内网络直连 HuggingFace 会超时、hf-xet 会报 401,统一走镜像并禁用 xet(可用环境变量覆盖)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper.utils import download_model  # noqa: E402

from app.config import settings  # noqa: E402


if __name__ == "__main__":
    print(f"下载 ASR 模型 {settings.asr_model_size} -> {settings.asr_model_dir}")
    path = download_model(settings.asr_model_size, output_dir=settings.asr_model_dir)
    print(f"完成:{path}")
