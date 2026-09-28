"""文档脱敏:把源文件里的军事/装备敏感字段替换为分类标签(文本)或涂黑(图片/PDF)。

规则集中在 SENSITIVE_RULES,按顺序替换;图片/PDF 按「整框/整行」命中即遮蔽。
装备类型名称(驱逐舰/护卫舰等公开分类)不算敏感,只脱敏具体参数/坐标/番号/编成/机构。
"""

import io
import re

from .paddle_ocr import recognize_with_boxes

# 每条 = (分类标签, 正则)。替换为标签,图片/PDF 则涂黑命中区域。
SENSITIVE_RULES: list[tuple[str, str]] = [
    # 坐标:经纬度 / 度分秒
    ("[坐标]", r"(?:北纬|南纬|东经|西经)\s*[0-9]+(?:\.[0-9]+)?\s*[°度]?"),
    ("[坐标]", r"[0-9]{1,3}°[0-9]{1,2}[′'分][0-9]{1,2}[″\"秒]?(?:[NSEWnsew])?"),
    # 参数:字段名[单位]:数值
    (
        "[参数]",
        r"(?:排水量|航速|最大航速|巡航速度|射程|口径|舰长|舰宽|吃水|吨位|乘员|满载|服役|续航|速度|航程|年份)"
        r"\s*[^\n:：]{0,3}?[：:]\s*[0-9][^\n]*",
    ),
    # 参数:数值 + 单位
    ("[参数]", r"[0-9]+(?:\.[0-9]+)?\s*(?:吨|节|海里|公里|千米|米|马赫|kn|kt|nm|km|m)\b"),
    # 番号:部队番号
    ("[番号]", r"第?[一二三四五六七八九十百0-9]+(?:军|师|旅|团|营|连|排|班)"),
    ("[番号]", r"[0-9]{2,4}部队"),
    # 编制:编队 / 序列 / 舰队
    ("[编制]", r"(?:编队|支队|大队|舰队|序列|建制|编成)"),
    # 机构:涉密机构
    ("[机构]", r"(?:国防部|火箭军|战略支援部队|军事科学院|军工|兵工厂|设计局|研究院)"),
]


def _is_sensitive(text: str) -> bool:
    return any(re.search(pattern, text) for _, pattern in SENSITIVE_RULES)


def desensitize_text(text: str) -> str:
    """文本脱敏:命中的片段替换为分类标签。"""
    for label, pattern in SENSITIVE_RULES:
        text = re.sub(pattern, label, text)
    return text


_IMG_SAVE = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "bmp": "BMP",
    "tif": "TIFF",
    "tiff": "TIFF",
}


def desensitize_image(content: bytes, ext: str = "png") -> bytes:
    """图片脱敏:OCR 定位敏感文字框并涂黑,返回同格式图片字节。"""
    from PIL import Image, ImageDraw

    boxes = recognize_with_boxes(content)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for text, (x1, y1, x2, y2) in boxes:
        if _is_sensitive(text):
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format=_IMG_SAVE.get(ext, "PNG"))
    return buf.getvalue()


def desensitize_pdf(content: bytes) -> bytes:
    """PDF 脱敏:按行聚合文本,命中敏感规则的行涂黑。"""
    import pymupdf as fitz

    doc = fitz.open(stream=content, filetype="pdf")
    for page in doc:
        lines: dict[tuple, dict] = {}
        for w in page.get_text("words"):
            x0, y0, x1, y1, word, block, line = w[0], w[1], w[2], w[3], w[4], w[5], w[6]
            key = (block, line)
            entry = lines.setdefault(key, {"text": "", "rect": [float("inf"), float("inf"), -1.0, -1.0]})
            entry["text"] += word
            entry["rect"][0] = min(entry["rect"][0], x0)
            entry["rect"][1] = min(entry["rect"][1], y0)
            entry["rect"][2] = max(entry["rect"][2], x1)
            entry["rect"][3] = max(entry["rect"][3], y1)
        for entry in lines.values():
            if _is_sensitive(entry["text"]):
                page.draw_rect(fitz.Rect(entry["rect"]), color=(0, 0, 0), fill=(0, 0, 0))
    return doc.tobytes()


_MIME = {
    "md": "text/markdown; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "htm": "text/html; charset=utf-8",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
}

# 视频/音频:下载时直接返回原文件,不做脱敏(只补正确 Content-Type)。
_VIDEO_MIME = {
    "mp4": "video/mp4",
    "m4v": "video/x-m4v",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "wmv": "video/x-ms-wmv",
    "flv": "video/x-flv",
}

_AUDIO_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "wma": "audio/x-ms-wma",
}


def desensitize_file(filename: str, content: bytes) -> tuple[bytes, str]:
    """按扩展名分派:文本类(md/txt/csv/图片/PDF)脱敏;视频/音频直接原样返回。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 视频/音频:直接返回原文件(下载时不做脱敏)
    if ext in _VIDEO_MIME:
        return content, _VIDEO_MIME[ext]
    if ext in _AUDIO_MIME:
        return content, _AUDIO_MIME[ext]

    if ext in ("md", "txt", "csv", "html", "htm"):
        text = content.decode("utf-8", errors="replace")
        return desensitize_text(text).encode("utf-8"), _MIME[ext]
    if ext in _IMG_SAVE:
        return desensitize_image(content, ext), _MIME[ext]
    if ext == "pdf":
        return desensitize_pdf(content), _MIME["pdf"]
    return content, "application/octet-stream"
