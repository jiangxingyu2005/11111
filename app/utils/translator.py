"""英文 -> 中文翻译(MarianMT 本地模型),把 BLIP 英文 caption 翻成中文。

模型单独放在项目内 model/mt-en-zh/ 文件夹(见 config.translator_model_dir),需先运行
`python scripts/download_translator_model.py` 下载。翻译是同步、CPU 密集操作,由 API
层丢 run_in_threadpool 调用。transformers 懒加载,未安装/未下载时抛错由上层降级
(保留英文 caption 或纯文字),不影响主流程。
"""

from pathlib import Path

from ..config import settings

_tokenizer = None
_model = None


def _get_translator():
    """懒加载 MarianMT 翻译 tokenizer + model 单例。"""
    global _tokenizer, _model
    if _model is None:
        model_dir = Path(settings.translator_model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"翻译模型未下载:{model_dir},请先运行 python scripts/download_translator_model.py"
            )
        from transformers import MarianMTModel, MarianTokenizer

        _tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
        _model = MarianMTModel.from_pretrained(str(model_dir)).eval()
    return _tokenizer, _model


def translate_en_zh(text: str) -> str:
    """英文文本 -> 中文翻译。"""
    tokenizer, model = _get_translator()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    out = model.generate(**inputs, max_new_tokens=settings.translator_max_new_tokens)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
