"""预下载英文->中文翻译模型到项目内 model/mt-en-zh/ 文件夹。

用法:python scripts/download_translator_model.py
"""

import os
import sys
from pathlib import Path

# 国内网络直连 HuggingFace 会超时、hf-xet 会报 401,统一走镜像并禁用 xet(可用环境变量覆盖)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402

from app.config import settings  # noqa: E402


if __name__ == "__main__":
    print(f"下载翻译模型 {settings.translator_model_id} -> {settings.translator_model_dir}")
    path = snapshot_download(settings.translator_model_id, local_dir=settings.translator_model_dir)
    print(f"完成:{path}")
