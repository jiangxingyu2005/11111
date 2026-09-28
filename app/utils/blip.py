"""图片语义理解(BLIP):生成图片描述 caption。

模型单独放在项目内 model/blip/ 文件夹(见 config.blip_model_dir),需先运行
`python scripts/download_blip_model.py` 下载。caption 生成是同步、CPU 密集操作,
由 API 层丢 run_in_threadpool 调用。transformers 为懒加载(首次调用才 import),
未安装时抛错由上层降级为纯文字,不影响主流程。

注:最初计划用 BLIP2(blip2-opt-2.7b),但其 float32 权重约 14GB,本机内存不足,
改用同源的 BLIP-base(Salesforce/blip-image-captioning-base,247M),功能一致。
"""

import io
from pathlib import Path

from PIL import Image

from ..config import settings

_processor = None
_model = None


def _get_captioner():
    """懒加载 BLIP processor + model 单例,从本地 model/blip 目录加载。"""
    global _processor, _model
    if _model is None:
        model_dir = Path(settings.blip_model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"BLIP 模型未下载:{model_dir},请先运行 python scripts/download_blip_model.py"
            )
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _processor = BlipProcessor.from_pretrained(str(model_dir))
        _model = BlipForConditionalGeneration.from_pretrained(str(model_dir)).eval()
    return _processor, _model


def caption_image(content: bytes) -> str:
    """图片字节 -> 语义描述 caption(BLIP-base 为英文 decoder,默认英文描述)。"""
    processor, model = _get_captioner()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=settings.blip_max_new_tokens)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()
