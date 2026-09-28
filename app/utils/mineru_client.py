"""MinerU v1 API 客户端。

协议(OpenAI Files 风格,异步):
1. 上传:POST /v1/uploads -> PUT /v1/uploads/{id}/content -> POST /v1/uploads/{id}/complete
2. 解析:POST /v1/parse/jobs(引用 file_id)-> 轮询 GET /v1/parse/jobs/{job_id}
3. 下载:GET /v1/files/{file_id}/content
"""

import asyncio
import hashlib
import json
import time

import httpx

from ..config import settings


class MinerUError(Exception):
    """MinerU 调用失败(连接/超时/HTTP 错误/业务错误)。"""


_MIME_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _mime_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MIME_TYPES.get(ext, "application/octet-stream")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_page_count(raw_entry: dict) -> int | None:
    """从 middle_json 里提取页数,拿不到返回 None。"""
    middle = raw_entry.get("middle_json") or {}
    if not isinstance(middle, dict):
        return None
    doc = (middle.get("metadata") or {}).get("document") or {}
    for key in ("page_count", "page_cnt"):
        v = doc.get(key)
        if isinstance(v, int):
            return v
    pages = middle.get("pages")
    if isinstance(pages, list):
        return len(pages)
    return None


class MinerUClient:
    def __init__(self) -> None:
        self.base_url = settings.mineru_api_base_url.rstrip("/")
        self.prefix = settings.mineru_api_prefix.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.prefix}{path}"

    async def _upload(self, client: httpx.AsyncClient, filename: str, content: bytes) -> str:
        """三步上传:create -> put content -> complete,返回 file_id。"""
        r = await client.post(
            self._url("/uploads"),
            json={
                "filename": filename,
                "bytes": len(content),
                "mime_type": _mime_type(filename),
                "purpose": "parse",
            },
        )
        r.raise_for_status()
        upload_id = r.json()["id"]

        r = await client.put(self._url(f"/uploads/{upload_id}/content"), content=content)
        r.raise_for_status()

        r = await client.post(
            self._url(f"/uploads/{upload_id}/complete"),
            json={"sha256sum": _sha256(content)},
        )
        r.raise_for_status()
        return r.json()["file"]["id"]

    async def _create_job(self, client: httpx.AsyncClient, file_id: str) -> str:
        body: dict = {
            "files": [{"source": {"type": "file_id", "file_id": file_id}}],
            "output_formats": settings.output_formats,
        }
        if settings.mineru_tier:
            body["tier"] = settings.mineru_tier
        if settings.mineru_ocr_mode:
            body["ocr_mode"] = settings.mineru_ocr_mode

        r = await client.post(self._url("/parse/jobs"), json=body)
        r.raise_for_status()
        return r.json()["job_id"]

    async def _wait_job(self, client: httpx.AsyncClient, job_id: str) -> dict:
        deadline = time.time() + settings.mineru_max_wait_seconds
        while time.time() < deadline:
            r = await client.get(self._url(f"/parse/jobs/{job_id}"))
            r.raise_for_status()
            job = r.json()
            if job["status"] in ("completed", "partial", "failed", "canceled"):
                return job
            await asyncio.sleep(settings.mineru_poll_interval)
        raise MinerUError(f"解析任务超时(超过 {settings.mineru_max_wait_seconds}s)")

    async def _download(self, client: httpx.AsyncClient, file_id: str) -> str:
        r = await client.get(self._url(f"/files/{file_id}/content"))
        r.raise_for_status()
        # MinerU 返回的 markdown/middle_json 是 UTF-8,但响应头是 application/octet-stream
        # (无 charset),httpx 的 r.text 会猜测编码(中文环境易误判成 GBK 等),强制按 UTF-8 解码。
        return r.content.decode("utf-8")

    async def parse_file(self, filename: str, content: bytes) -> tuple[str, dict]:
        """完整流程:上传 -> 建任务 -> 轮询 -> 下载,返回 (markdown, raw_entry)。"""
        async with httpx.AsyncClient(timeout=settings.mineru_timeout) as client:
            file_id = await self._upload(client, filename, content)
            job_id = await self._create_job(client, file_id)
            job = await self._wait_job(client, job_id)

            markdown = ""
            middle_json: dict = {}
            errors: list[str] = []
            for f in job.get("files", []):
                if f.get("status") != "completed":
                    err = f.get("error") or {}
                    errors.append(f"{f.get('name', '')}: {err.get('message', f.get('status'))}")
                    continue
                outputs = f.get("output_files") or {}
                if outputs.get("markdown"):
                    markdown = await self._download(client, outputs["markdown"]["file_id"])
                if outputs.get("middle_json"):
                    raw = await self._download(client, outputs["middle_json"]["file_id"])
                    try:
                        middle_json = json.loads(raw)
                    except json.JSONDecodeError:
                        middle_json = {"raw": raw}

            raw_entry = {
                "job_id": job_id,
                "job_status": job.get("status"),
                "files": job.get("files", []),
                "middle_json": middle_json,
            }

            if job.get("status") == "failed":
                raise MinerUError(f"解析任务失败: {'; '.join(errors) or '未知错误'}")

            return markdown, raw_entry

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(self._url("/health"))
            r.raise_for_status()
            return r.json()


mineru_client = MinerUClient()
