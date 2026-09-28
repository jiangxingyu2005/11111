"""GP-02 CHUNK:父子双层切分(ems chunk/parent_child.py 移植,I/O 改 async)。

H1/H2/H3 切父块 -> 512/10% 切子块 -> 确定性发号 {doc_id}_{章节序号}#{子块序号}。
父块入 PG parent_document、子块入 PG chunk_relation(此时不写 Milvus)。
无标题文档回退段落聚合。融合后:kb_id 贯穿、front-matter 三项置空、免 CSV 直对象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ChunkRelation, ParentDocument
from . import db
from .config import get
from .parse_md import content_hash, split_front_matter

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass
class _Section:
    level: int
    title: str
    chapter_no: str
    content: str


def _split_sections(text: str) -> list[_Section]:
    """把 md 按 H1/H2/H3 切分为章节,含章节序号与正文。"""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections: list[_Section] = []
    counters = {1: 0, 2: 0, 3: 0}
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        counters[level] += 1
        for lv in range(level + 1, 4):
            counters[lv] = 0
        chapter_no = ".".join(str(counters[lv]) for lv in range(1, level + 1))
        sections.append(_Section(level=level, title=title, chapter_no=chapter_no, content=content))
    return sections


def _paragraph_fallback(text: str) -> list[_Section]:
    """无标题文档:段落聚合回退。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    content = "\n\n".join(paras) if paras else text.strip()
    return [_Section(level=1, title="无标题", chapter_no="1", content=content)]


def chunk_text(text: str, chunk_size: int, overlap: float) -> list[str]:
    """固定字符数 + 比例重叠切子块。"""
    text = text.strip()
    if not text:
        return []
    overlap_chars = int(chunk_size * overlap)
    if overlap_chars >= chunk_size:
        overlap_chars = 0
    step = chunk_size - overlap_chars
    if step <= 0:
        step = chunk_size
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def build_parent_chunks(doc_id: str, text: str) -> list[dict]:
    sections = _split_sections(text) or _paragraph_fallback(text)
    parents: list[dict] = []
    for s in sections:
        parents.append({
            "parent_id": f"{doc_id}_{s.chapter_no}",
            "doc_id": doc_id, "title": s.title, "level": s.level,
            "chapter_no": s.chapter_no, "content": s.content,
        })
    return parents


def build_child_chunks(parents: list[dict], chunk_size: int, overlap: float) -> list[dict]:
    children: list[dict] = []
    for p in parents:
        for idx, piece in enumerate(chunk_text(p["content"], chunk_size, overlap), start=1):
            children.append({
                "child_id": f"{p['parent_id']}#{idx}",
                "parent_id": p["parent_id"],
                "chunk_index": idx,
                "content": piece,
            })
    return children


def chunk_document(doc_id: str, text: str, chunk_size: Optional[int] = None,
                   overlap: Optional[float] = None) -> tuple[list[dict], list[dict]]:
    chunk_size = chunk_size if chunk_size is not None else int(get("chunk.chunk_size", 512))
    overlap = overlap if overlap is not None else float(get("chunk.overlap", 0.10))
    parents = build_parent_chunks(doc_id, text)
    children = build_child_chunks(parents, chunk_size, overlap)
    return parents, children


async def write_to_pg(session: AsyncSession, doc_id: str, kb_id, text: str) -> tuple[list[dict], list[dict]]:
    """切分 + 元数据装配 + 写入 PG(幂等 upsert)。返回 (parents, children) 直对象。"""
    meta, _ = split_front_matter(text)
    file_hash = content_hash(text)
    parents, children = chunk_document(doc_id, text)
    for p in parents:
        p["kb_id"] = kb_id
        p["event_time"] = str(meta.get("event_time") or "")
        p["azimuth_range"] = str(meta.get("azimuth_range") or "")
        p["freq_band"] = str(meta.get("freq_band") or "")
        p["file_hash"] = file_hash
        await db.upsert(session, ParentDocument, p, "parent_id")
    for c in children:
        await db.upsert(session, ChunkRelation, c, "child_id")
    return parents, children
