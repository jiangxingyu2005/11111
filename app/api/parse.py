import csv
import hashlib
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..schemas import (
    CleanStats,
    FileArtifacts,
    FileResult,
    HealthResponse,
    IqParseResponse,
    IqResult,
    MinerUHealth,
    ParseResponse,
)
from ..utils.asr import transcribe_audio
from ..utils.blip import caption_image
from ..utils.translator import translate_en_zh
from ..utils.cleaning import CLEAN_RULESET_VERSION, clean, strip_image_refs
from ..utils.database import SessionLocal
from ..utils.document import create_document, mark_document_in_kb
from ..graphrag.ingest import ingest_document
from ..utils.kb import get_kb
from ..utils.mineru_client import MinerUError, extract_page_count, mineru_client
from ..utils.minio_client import upload_clean, upload_source
from ..utils.paddle_ocr import recognize_text as paddle_recognize_text
from ..utils.scoring import noise_ratio, score_markdown_total
from ..utils.storage import ensure_dirs, save_result
from ..utils.html_parser import parse_html_to_markdown
from ..utils.iq import process_iq_h5
from ..utils.video import dedupe_frames, extract_frames

router = APIRouter()

logger = logging.getLogger(__name__)

_PLAIN_EXTS = {".txt", ".md", ".csv"}


def _decode_text(content: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


_CSV_TITLE_COLS = ("型号", "名称", "name")


def _csv_to_markdown(text: str) -> str:
    """CSV -> 每行一节:标题为型号/名称,正文为「字段：值」单行(只留非空字段)。

    原样 CSV 里表头和数据相隔很远,宽表(几十列、大半为空)时 LLM 会把值对错列
    (实测把 弹头kg 的 480-700 当成射程)。转成 字段：值 后每个值紧挨字段名。
    整行写成一行:clean() 会删全局重复的短行与重复行,逐字段分行(如「国别：美国」)会被误删。
    """
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return text
    header = [h.strip() for h in rows[0]]
    title_col = next((i for i, h in enumerate(header) if h in _CSV_TITLE_COLS), 0)
    out: list[str] = []
    for n, row in enumerate(rows[1:], 1):
        pairs = [(header[i], v.strip()) for i, v in enumerate(row)
                 if i < len(header) and header[i] and v.strip()]
        if not pairs:
            continue
        title = (row[title_col].strip() if title_col < len(row) else "") or f"第{n}行"
        out.append(f"## {title}")
        out.append("；".join(f"{k}：{v}" for k, v in pairs) + "。")
    return "\n".join(out) + "\n" if out else text


async def _require_kb(kb_id: UUID) -> None:
    """校验知识库存在(未逻辑删除),不存在抛 404。"""
    async with SessionLocal() as db:
        if await get_kb(db, kb_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")


async def _finalize(
    kb_id: UUID,
    filename: str,
    content: bytes,
    raw_md: str,
    raw_entry: dict,
    page_count: int | None,
    engine: str,
    return_content: bool,
) -> FileResult:
    """清洗 -> 打分 -> 落盘 -> 入 MinIO -> 登记 documents,返回 FileResult。"""
    clean_md, stats = clean(
        raw_md,
        header_threshold=settings.clean_header_threshold,
        header_max_len=settings.clean_header_max_len,
    )
    file_id = uuid4()  # 文件 uuid:唯一标识 + 落盘目录名

    file_score = score_markdown_total(
        clean_md,
        min_chunk_len=settings.score_min_chunk_len,
        max_chunk_len=settings.score_max_chunk_len,
        w_len=settings.score_w_len,
        w_info=settings.score_w_info,
        w_content=settings.score_w_content,
        w_hits=settings.score_w_hits,
    )

    meta = {
        "filename": filename,
        "doc_id": str(file_id),
        "kb_id": str(kb_id),
        "engine": engine,
        "parsed_at": datetime.now().isoformat(),
        "page_count": page_count,
        "score": file_score,
        "job_id": raw_entry.get("job_id"),
        "job_status": raw_entry.get("job_status"),
        "stats": stats,
        "content_sha256": hashlib.sha256((raw_md or "").encode("utf-8")).hexdigest(),
        "clean_ruleset_version": CLEAN_RULESET_VERSION,
    }

    # clean.json 存清洗后产物,与 raw.json(原始数据)区分开
    clean_entry = {
        "markdown": clean_md,
        "page_count": page_count,
        "stats": stats,
    }

    paths = save_result(str(file_id), raw_md, raw_entry, clean_md, clean_entry, meta)

    # 入 MinIO(同步 SDK 丢线程池):源文件→raw 桶,清洗产物→clean 桶
    await run_in_threadpool(upload_source, kb_id, file_id, filename, content)
    clean_md_key = await run_in_threadpool(
        upload_clean, kb_id, file_id, "clean.md", clean_md.encode("utf-8"), "text/markdown"
    )
    meta_key = await run_in_threadpool(
        upload_clean, kb_id, file_id, "meta.json",
        json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"), "application/json",
    )

    # 登记 documents 表(路径指向 MinIO 对象键;上传成功才登记)
    # id 用 file_id,保证 documents.id == 落盘目录名 == MinIO 对象键里的文件 uuid
    async with SessionLocal() as db:
        await create_document(
            db,
            kb_id=kb_id,
            filename=filename,
            meta_path=meta_key,
            clean_md_path=clean_md_key,
            score=file_score,
            id=file_id,
        )

    # GraphRAG 入库(ems_graphrag 融合:清洗 md -> PG 父/子块 -> Milvus -> Neo4j)
    # IQ 不入图谱;失败仅记日志,不影响 parse 结果(documents.in_kb 保持 False)
    # 低质/乱码:仍切块进 Milvus 可检索,不抽实体(in_kb 保持 False)
    if settings.graphrag_enabled and engine != "iq":
        noise = noise_ratio(clean_md)
        q = file_score if isinstance(file_score, (int, float)) else 0.0
        graph = q >= settings.score_q_quarantine and noise < settings.score_noise_quarantine
        if not graph:
            logger.info("噪声/低质隔离不抽图谱: file=%s score=%.4f noise=%.4f", filename, q, noise)
        try:
            await ingest_document(kb_id, str(file_id), clean_md, engine, filename, graph=graph)
            if graph:
                async with SessionLocal() as db:
                    await mark_document_in_kb(db, file_id)
        except Exception as e:
            logger.warning("GraphRAG 入库失败(doc_id=%s engine=%s): %s", file_id, engine, e)

    return FileResult(
        filename=filename,
        doc_id=str(file_id),
        success=True,
        page_count=page_count,
        clean_md=clean_md if return_content else None,
        files=FileArtifacts(**paths),
        stats=CleanStats(**stats),
    )


async def _extract_image(filename: str, content: bytes) -> tuple[str, dict, int | None, str]:
    """图片提取:BLIP 语义描述(英文,翻译成中文) + MinerU 文字合并;MinerU 失败降级飞桨 PaddleOCR。

    返回 (markdown, raw_entry, page_count, engine)。
    """
    # 1. BLIP 语义描述(英文 caption,再翻成中文;失败/空则降级为纯文字)
    caption = None
    try:
        cap = await run_in_threadpool(caption_image, content)
        if cap and cap.strip():
            caption = await run_in_threadpool(translate_en_zh, cap.strip())
        else:
            caption = None
    except Exception:
        caption = None

    # 2. MinerU 文字(优先)
    text = None
    raw_entry: dict = {}
    page_count: int | None = None
    text_engine = "mineru"
    try:
        markdown, raw_entry = await mineru_client.parse_file(filename, content)
        if markdown and markdown.strip():
            # 剥离图片引用(base64 内嵌 / 本地 / 远程),图片只保留文字
            text = strip_image_refs(markdown)
            page_count = extract_page_count(raw_entry)
    except Exception:
        pass  # MinerU 抛错 → 降级

    # 3. 飞桨兜底(MinerU 无文字时)
    if not text:
        try:
            ocr_text = await run_in_threadpool(paddle_recognize_text, content)
        except Exception:
            ocr_text = ""
        if ocr_text and ocr_text.strip():
            text = ocr_text
            raw_entry = {"engine": "paddleocr", "recognized_text": ocr_text}
            text_engine = "paddleocr"
            page_count = 1

    # 4. 合并 caption + 文字
    if not caption and not text:
        raise MinerUError("BLIP / MinerU / 飞桨均未提取到内容")
    raw_md = "\n\n".join(p for p in (caption, text) if p and p.strip())

    engine = f"blip+{text_engine}" if caption else text_engine
    raw_entry["caption"] = caption
    raw_entry["pipeline"] = engine

    return raw_md, raw_entry, page_count, engine


async def _extract_video(filename: str, content: bytes) -> tuple[str, dict, int, str]:
    """视频提取:抽帧 -> 去重 -> 逐帧复用图片链路(_extract_image)-> 合并成一条。

    返回 (raw_md, raw_entry, page_count, engine="video")。
    """
    # 1. 抽帧 + 去重(同步/CPU 密集,丢线程池)
    frames = await run_in_threadpool(extract_frames, content, filename)
    frames = await run_in_threadpool(dedupe_frames, frames)
    if not frames:
        raise MinerUError("未能从视频中抽取到可用帧")

    # 2. 逐帧提取(串行;复用 BLIP + MinerU + PaddleOCR 兜底)
    parts: list[str] = []
    frame_metas: list[dict] = []
    for i, frame_bytes in enumerate(frames):
        # 帧文件名必须带 .jpg 后缀,否则 MinerU 按 octet-stream 上传会拒绝
        frame_name = f"{filename}_frame_{i + 1}.jpg"
        try:
            markdown, raw_entry, page_count, engine = await _extract_image(frame_name, frame_bytes)
        except Exception:
            continue  # 单帧失败跳过,其余帧继续
        if not markdown or not markdown.strip():
            continue
        parts.append(markdown)
        frame_metas.append(
            {
                "index": i + 1,
                "engine": engine,
                "page_count": page_count,
                "caption": raw_entry.get("caption"),
            }
        )

    if not parts:
        raise MinerUError("视频各帧均未提取到内容")

    raw_md = "\n\n".join(parts)
    raw_entry = {
        "engine": "video",
        "extracted_frames": len(frames),
        "frame_count": len(frame_metas),
        "frames": frame_metas,
    }
    return raw_md, raw_entry, len(frame_metas), "video"


@router.post("/parse", response_model=ParseResponse)
async def parse(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> ParseResponse:
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[FileResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                FileResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            suffix = Path(filename).suffix.lower()
            if suffix in _PLAIN_EXTS:
                raw_md = _decode_text(content)
                if suffix == ".csv":
                    raw_md = _csv_to_markdown(raw_md)
                raw_entry = {"engine": "plain"}
                results.append(
                    await _finalize(
                        kb_id, filename, content, raw_md, raw_entry,
                        None, "plain", return_content,
                    )
                )
                continue
            raw_md, raw_entry = await mineru_client.parse_file(filename, content)
            results.append(
                await _finalize(
                    kb_id, filename, content, raw_md, raw_entry,
                    extract_page_count(raw_entry), "mineru", return_content,
                )
            )
        except MinerUError as e:
            results.append(FileResult(filename=filename, success=False, error=str(e)))
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(FileResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return ParseResponse(results=results)


@router.post("/parse-image", response_model=ParseResponse)
async def parse_image(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> ParseResponse:
    """图片解析:MinerU 优先,抛错/空结果降级飞桨 PaddleOCR,再打分入库。"""
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[FileResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                FileResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            markdown, raw_entry, page_count, engine = await _extract_image(filename, content)
            results.append(
                await _finalize(kb_id, filename, content, markdown, raw_entry, page_count, engine, return_content)
            )
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(FileResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return ParseResponse(results=results)


@router.post("/parse-audio", response_model=ParseResponse)
async def parse_audio(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> ParseResponse:
    """音频解析:ASR 转文字 -> 清洗打分入库(清洗步骤与正常文本一致,不过 MinerU)。"""
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[FileResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                FileResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            # 1. ASR 转文字(同步/CPU 密集,丢线程池)
            text = await run_in_threadpool(transcribe_audio, content, filename)
            if not text.strip():
                results.append(FileResult(filename=filename, success=False, error="未从音频中识别到文字"))
                continue

            # 2. ASR 文字直接走 _finalize 里的 clean() 清洗(与正常文本同一步骤),不过 MinerU
            raw_entry = {"engine": "asr", "asr_text": text}
            results.append(
                await _finalize(kb_id, filename, content, text, raw_entry, None, "asr", return_content)
            )
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(FileResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return ParseResponse(results=results)


@router.post("/parse-video", response_model=ParseResponse)
async def parse_video(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> ParseResponse:
    """视频解析:抽帧 -> 每帧图片提取(语义+文字)-> 合并成一条文档 -> 清洗打分入库。"""
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[FileResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                FileResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            raw_md, raw_entry, page_count, engine = await _extract_video(filename, content)
            results.append(
                await _finalize(kb_id, filename, content, raw_md, raw_entry, page_count, engine, return_content)
            )
        except MinerUError as e:
            results.append(FileResult(filename=filename, success=False, error=str(e)))
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(FileResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return ParseResponse(results=results)


@router.post("/parse-html", response_model=ParseResponse)
async def parse_html(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> ParseResponse:
    """HTML 解析:提取正文(标题/段落/列表/表格转 markdown)-> 清洗打分入库。"""
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[FileResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                FileResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            markdown = await run_in_threadpool(parse_html_to_markdown, content)
            if not markdown.strip():
                results.append(FileResult(filename=filename, success=False, error="未从 HTML 提取到内容"))
                continue

            raw_entry = {"engine": "html"}
            results.append(
                await _finalize(kb_id, filename, content, markdown, raw_entry, None, "html", return_content)
            )
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(FileResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return ParseResponse(results=results)


@router.post("/parse-iq", response_model=IqParseResponse)
async def parse_iq(
    files: list[UploadFile] = File(...),
    kb_id: UUID = Form(..., description="所属知识库 id"),
    return_content: bool = Query(True, description="是否在响应里返回清洗后的 markdown 正文"),
) -> IqParseResponse:
    """IQ 信号解析:H5 -> SNR 清洗 -> IQ 补全 -> 信号处理 -> 26 特征中文描述入库(不落中间 IQ 数据)。"""
    await _require_kb(kb_id)
    ensure_dirs()
    results: list[IqResult] = []

    for f in files:
        filename = f.filename or "unknown"
        content = await f.read()

        if len(content) > settings.max_upload_mb * 1024 * 1024:
            results.append(
                IqResult(filename=filename, success=False, error=f"文件超过大小限制 {settings.max_upload_mb}MB")
            )
            continue

        try:
            iq = await run_in_threadpool(process_iq_h5, content)
            fr = await _finalize(
                kb_id, filename, content, iq["md"], iq["raw_entry"], None, "iq", return_content
            )
            extra = {
                "snr": iq["snr"],
                "attenuation_db": iq["attenuation_db"],
                "charts": iq["charts"],
                "features": iq["features"],
                "prediction": iq["prediction"],
                "param_guess": iq["param_guess"],
                "iq_raw": iq["iq_raw"],
                "iq_clean": iq["iq_clean"],
                "iq_masked": iq["iq_masked"],
                "iq_recovered": iq["iq_recovered"],
                "iq_meta": iq["iq_meta"],
            }
            results.append(IqResult(**fr.model_dump(), **extra))
        except Exception as e:  # 兜底:任何异常都不让整个请求崩溃
            results.append(IqResult(filename=filename, success=False, error=f"处理失败: {e}"))

    return IqParseResponse(results=results)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        detail = await mineru_client.health()
        return HealthResponse(status="ok", mineru=MinerUHealth(ok=True, detail=json.dumps(detail, ensure_ascii=False)))
    except Exception as e:
        return HealthResponse(status="degraded", mineru=MinerUHealth(ok=False, detail=str(e)))


@router.get("/healthz")
async def healthz() -> dict:
    """存活探针（与 retrieval/model_stub 的 /healthz 对齐，纯进程存活，不查 mineru）。"""
    return {"status": "ok", "service": "qingxi"}
