import json
from pathlib import Path

from ..config import settings


def ensure_dirs() -> None:
    """确保上传目录与输出目录存在。"""
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


def save_result(
    doc_id: str,
    raw_md: str,
    raw_json: dict,
    clean_md: str,
    clean_json: dict,
    meta: dict,
) -> dict:
    """把解析与清洗结果落盘到 {OUTPUT_DIR}/{doc_id}/,返回各文件路径。

    doc_id 为文件 uuid 字符串(作为唯一目录名)。
    """
    doc_dir = Path(settings.output_dir) / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # (文件名, 内容),json 内容统一序列化
    files = {
        "raw_md": ("raw.md", raw_md),
        "raw_json": ("raw.json", json.dumps(raw_json, ensure_ascii=False, indent=2)),
        "clean_md": ("clean.md", clean_md),
        "clean_json": ("clean.json", json.dumps(clean_json, ensure_ascii=False, indent=2)),
        "meta": ("meta.json", json.dumps(meta, ensure_ascii=False, indent=2)),
    }

    paths = {}
    for key, (name, content) in files.items():
        path = doc_dir / name
        path.write_text(content, encoding="utf-8")
        paths[key] = str(path)
    return paths
