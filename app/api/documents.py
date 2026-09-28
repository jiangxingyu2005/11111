"""文档读取接口:按 uuid 返回脱敏后的源文件(保持原格式)。"""

import io
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from minio.error import S3Error
from starlette.concurrency import run_in_threadpool

from ..utils.database import SessionLocal
from ..utils.desensitize import desensitize_file
from ..utils.document import get_document
from ..utils.minio_client import download_source

router = APIRouter(prefix="/documents", tags=["documents"])


async def _stream_source(doc_id: UUID) -> StreamingResponse:
    """共享实现:查文档 -> MinIO 下载源文件 -> 按类型分派(文本类脱敏 / 视频音频原样)。

    仅实时计算,不落库、不写 MinIO。
    """
    async with SessionLocal() as db:
        doc = await get_document(db, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")

    # MinIO 下载(同步 SDK)丢线程池
    try:
        content = await run_in_threadpool(download_source, doc.kb_id, doc.id, doc.filename)
    except S3Error as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="源文件不存在") from e

    # 文本类脱敏 / 视频音频直接返回(CPU 密集同步操作丢线程池)
    desensitized, media_type = await run_in_threadpool(desensitize_file, doc.filename, content)

    name = Path(doc.filename).name or doc.filename
    disposition = f"attachment; filename*=UTF-8''{quote(name)}"
    return StreamingResponse(
        io.BytesIO(desensitized),
        media_type=media_type,
        headers={"Content-Disposition": disposition},
    )


@router.get("/{doc_id}/download")
async def download(doc_id: UUID) -> StreamingResponse:
    """下载源文件:文本类(md/txt/csv/图片/PDF)下载时脱敏,视频/音频直接返回原文件。"""
    return await _stream_source(doc_id)


@router.get("/{doc_id}/desensitize")
async def desensitize(doc_id: UUID) -> StreamingResponse:
    """按文档 uuid 返回脱敏后的源文件(保持原格式)。视频/音频直接返回原文件。"""
    return await _stream_source(doc_id)
