"""视频抽帧与帧去重(同步、CPU 密集,由 API 层丢 run_in_threadpool 执行)。"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..config import settings


def validate_video(content: bytes, filename: str) -> None:
    """视频完整性校验:不过抛清晰 ValueError,不让损坏/不完整文件进抽帧管道。

    只保证「打得开、有帧、能解析」:空文件 / 无法解码 / 读不到第一帧均拒收。
    帧率不硬判(fps 元数据缺失时抽帧已有 25fps 兜底,不算文件不完整)。
    """
    if not content:
        raise ValueError("视频文件为空,无法解析")
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("视频无法解码,可能已损坏或不是视频")
        try:
            ret, _ = cap.read()
            if not ret:
                raise ValueError("视频无有效帧")
        finally:
            cap.release()
    finally:
        os.unlink(tmp_path)


def extract_frames(content: bytes, filename: str) -> list[bytes]:
    """从视频字节抽取帧(每秒 settings.video_frame_rate 帧),返回 jpg 字节列表。

    opencv 的 VideoCapture 只能读文件路径,先把字节写到临时文件,抽完删除。
    """
    validate_video(content, filename)
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            # 每秒抽 video_frame_rate 帧 => 每隔 fps/rate 帧取 1 帧
            interval = max(1, int(fps / settings.video_frame_rate))
            frames: list[bytes] = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        frames.append(buf.tobytes())
                idx += 1
            return frames
        finally:
            cap.release()
    finally:
        os.unlink(tmp_path)


def _gray_hist(frame: bytes):
    """帧 jpg 字节 -> 64x64 灰度直方图(用于相似度比较)。"""
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    return cv2.calcHist([img], [0], None, [64], [0, 256])


def _similarity(frame_a: bytes, frame_b: bytes) -> float:
    """两帧灰度直方图相关性,1.0 表示完全相同。"""
    ha, hb = _gray_hist(frame_a), _gray_hist(frame_b)
    if ha is None or hb is None:
        return 0.0
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def dedupe_frames(frames: list[bytes]) -> list[bytes]:
    """相邻帧去重(相似度 >= 阈值判为重复跳过),去重后仍超上限则均匀采样。"""
    if not frames:
        return []

    threshold = settings.video_frame_similarity_threshold
    keep: list[bytes] = [frames[0]]
    for f in frames[1:]:
        if _similarity(keep[-1], f) < threshold:
            keep.append(f)

    # 超过上限 -> 均匀采样到 max_frames 帧,覆盖整个视频
    max_frames = max(1, settings.video_max_frames)
    if len(keep) > max_frames:
        keep = [keep[i * len(keep) // max_frames] for i in range(max_frames)]

    return keep
