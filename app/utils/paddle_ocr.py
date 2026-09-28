"""飞桨 PaddleOCR 降级封装:图片 OCR 提取文本(可选带坐标,用于脱敏打码)。

可选依赖:paddleocr(及其依赖 paddlepaddle)。未安装或不可用时抛 PaddleError,
由调用方决定如何兜底。懒加载:模型加载较慢,首次调用才初始化。
"""

import os
import tempfile

# paddlepaddle 3.3+ 在 Windows 上 oneDNN 有已知 bug(PIR 属性转换报错),禁用之。
# 必须放在 import paddleocr 之前(本模块懒加载,首次调用前已设置)。
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")


class PaddleError(Exception):
    """飞桨 OCR 调用失败(未安装 / 初始化失败 / 识别失败)。"""


_ocr = None


def _get_ocr():
    """懒加载 PaddleOCR 单例。"""
    global _ocr
    if _ocr is None:
        try:
            from paddleocr import PaddleOCR
        except Exception as e:  # 未安装
            raise PaddleError(f"飞桨 PaddleOCR 未安装: {e}")
        try:
            # 关闭文档预处理(方向分类/去弯曲),否则返回坐标是处理后图的坐标,
            # 与送进来的原图不对应,拿去涂黑会错位。坐标需 1:1 对应原图。
            _ocr = PaddleOCR(
                use_textline_orientation=True,
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as e:
            raise PaddleError(f"飞桨 PaddleOCR 初始化失败: {e}")
    return _ocr


def _to_xyxy(poly) -> tuple[int, int, int, int]:
    """多点坐标 → (x1, y1, x2, y2) 包围盒。"""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _extract_boxes(result) -> list[tuple[str, tuple[int, int, int, int]]]:
    """从 PaddleOCR 结果抽取 [(文本, 矩形框(x1,y1,x2,y2)), ...],兼容 2.x / 3.x。"""
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for item in result or []:
        if isinstance(item, dict):
            # 3.x:predict() 返回 OCRResult(dict 子类),含 rec_texts / rec_boxes / rec_polys
            texts = item.get("rec_texts") or []
            boxes = item.get("rec_boxes")
            is_xyxy = boxes is not None  # rec_boxes 直接是 [x1,y1,x2,y2]
            if boxes is None:
                boxes = item.get("rec_polys")  # 否则退四点多边形
            for i, t in enumerate(texts):
                if not t or boxes is None or i >= len(boxes):
                    continue
                box = boxes[i]
                xyxy = tuple(int(v) for v in box) if is_xyxy else _to_xyxy(box)
                out.append((str(t), xyxy))
            continue
        if isinstance(item, (list, tuple)):
            # 2.x:ocr() 返回 [[box, (text, score)], ...],box 为四点
            for row in item:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    poly, pair = row[0], row[1]
                    text = pair[0] if isinstance(pair, (list, tuple)) and pair else ""
                    if text:
                        out.append((str(text), _to_xyxy(poly)))
    return out


def recognize_with_boxes(content: bytes) -> list[tuple[str, tuple[int, int, int, int]]]:
    """识别图片字节,返回 [(文本, 原图矩形框), ...](坐标相对原图像素)。"""
    ocr = _get_ocr()

    # PaddleOCR 需要文件路径,这里走临时文件最稳妥
    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        try:
            result = ocr.predict(path)
        except AttributeError:
            result = ocr.ocr(path, cls=True)
        return _extract_boxes(result)
    except PaddleError:
        raise
    except Exception as e:
        raise PaddleError(f"飞桨识别失败: {e}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def recognize_text(content: bytes) -> str:
    """识别图片字节,返回按行拼接的纯文本。"""
    return "\n".join(t for t, _ in recognize_with_boxes(content))
