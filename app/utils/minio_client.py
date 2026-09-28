"""MinIO 对象存储封装:双桶(源文件 / 清洗后),按 知识库/文件 分层存对象。

对象键规则:
- 源文件桶(raw):  {kb_id}/{file_id}/{filename}
- 清洗后桶(clean): {kb_id}/{file_id}/clean.md、meta.json ...
"""

import io
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from ..config import settings

_client: Minio | None = None
_buckets_ready = False


def get_client() -> Minio:
    """懒加载 MinIO 客户端单例。"""
    global _client
    if _client is None:
        # minio SDK 的 endpoint 不带协议前缀(由 secure 决定 http/https)
        endpoint = settings.minio_endpoint
        for prefix in ("http://", "https://"):
            if endpoint.startswith(prefix):
                endpoint = endpoint[len(prefix):]
        endpoint = endpoint.rstrip("/")
        _client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def ensure_buckets() -> None:
    """确保源文件桶、清洗后桶存在(幂等)。"""
    global _buckets_ready
    if _buckets_ready:
        return
    client = get_client()
    for bucket in (settings.minio_raw_bucket, settings.minio_clean_bucket):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    _buckets_ready = True


def _object_key(*parts) -> str:
    """拼接对象键,过滤空段。"""
    return "/".join(str(p) for p in parts if p is not None and str(p) != "")


def _put(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    get_client().put_object(bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)


def upload_source(kb_id, file_id, filename: str, data: bytes) -> str:
    """上传源文件到 raw 桶,返回对象键。"""
    ensure_buckets()
    key = _object_key(kb_id, file_id, Path(filename).name)
    _put(settings.minio_raw_bucket, key, data)
    return key


def upload_clean(kb_id, file_id, name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """上传清洗产物到 clean 桶,返回对象键。"""
    ensure_buckets()
    key = _object_key(kb_id, file_id, name)
    _put(settings.minio_clean_bucket, key, data, content_type=content_type)
    return key


def get_object(bucket: str, key: str) -> bytes:
    """下载对象字节。对象不存在抛 S3Error(NoSuchKey)。"""
    resp = get_client().get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def download_source(kb_id, file_id, filename: str) -> bytes:
    """下载源文件(raw 桶),key 与 upload_source 对称:{kb_id}/{file_id}/{basename(filename)}。"""
    ensure_buckets()
    key = _object_key(kb_id, file_id, Path(filename).name)
    return get_object(settings.minio_raw_bucket, key)
