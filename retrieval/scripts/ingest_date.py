"""date/ 真实数据 → fake 后端 seed（数据接入层，多域多模态）。

覆盖需求文档「(1) 多源异构数据接入」所列格式（PDF 因标注 SYNTHETIC 除外）：
  - 雷达域：txt（230 情报报告）+ csv（1140 信号记录）
  - 装备域：html（106 装备网页档案）+ docx（38 装备文档档案）
  - 媒体域：images（168 高清图 1920×1080）+ audio（27 信号音频）+ video（15 装备视频）

产出 data/seed_date/ 的 parents/chunks/entities/edges/communities/meta + records.json
（结构化解析，供 eval_date.py 构造评测集 gold）。

边界（重要）：
  - 嵌入走 hash（fake 通道无模型），召回绝对值仍是字面/词面重叠量级，禁止当生产召回率；
  - 图片/音频/视频为【元数据接入】（当前程序为纯文本检索，无图像/音频/视频向量化能力），
    视频与装备档案通过 eq_ 编号关联；图片/音频无文本关联，仅证明"已接入可检索"。
  - 每个文档切成 1 个自包含子块，gold 由实体共现 / 标量容差客观判定。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.store.hash_embed import hash_embedding, hash_sparse, hash_numeric_embedding  # noqa: E402

DATE = ROOT.parent / "date"
OUT = ROOT / "data" / "seed_date"


def ent_id(typ: str, name: str) -> str:
    """稳定实体 id（md5，含类型，不依赖 PYTHONHASHSEED）。"""
    return "ent_" + hashlib.md5(f"{typ}:{name}".encode("utf-8")).hexdigest()[:8]


def _tof(s: str) -> float | None:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────
# 雷达域：txt（情报报告）
# ─────────────────────────────────────────────────────────────
def parse_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def field(pat: str) -> str:
        m = re.search(pat, text)
        return m.group(1).strip() if m else ""

    country_manu = field(r"国别/制造商[:：]\s*(.+)")
    country = country_manu.split("/")[0].strip() if country_manu else ""
    manufacturer = country_manu.split("/")[-1].strip() if "/" in country_manu else ""

    prf = pw = None
    m = re.search(r"脉冲重复频率\s*([\d.]+)\s*([kKmM]?Hz)", text)
    if m:
        v = float(m.group(1))
        u = m.group(2)
        prf = v * (1000 if u[:1] in "kK" else (1_000_000 if u[:1] in "mM" else 1))
    m = re.search(r"脉宽约?\s*([\d.]+)\s*(μ?s|us|ms|微秒|毫秒|纳秒|ns)", text)
    if m:
        v = float(m.group(1))
        u = m.group(2)
        pw = v * (1000 if u in ("ms", "毫秒") else (0.001 if u in ("ns", "纳秒") else 1))
    # [：\s]+ 排除正文里"…(脉宽/重频/调制方式)与部署动态"的误匹配
    mod = field(r"调制方式[:：\s]+([^\s,，。；;()（）\n]+)")

    return {
        "source": "txt", "domain": "radar", "modality": "txt", "doc_id": path.stem,
        "model": field(r"装备型号[:：]\s*(.+)"),
        "category": field(r"装备类别[:：]\s*(.+)"),
        "country": country, "manufacturer": manufacturer,
        "system": field(r"体制[:：]\s*(.+)"),
        "band": field(r"频段[:：]\s*(.+)"),
        "platform": field(r"平台[:：]\s*(.+)"),
        "function": field(r"用途[:：]\s*(.+)"),
        "rf_ghz": None, "prf_hz": prf, "pw_us": pw, "mod": mod,
        "content": text,
    }


# ─────────────────────────────────────────────────────────────
# 雷达域：csv（信号记录）
# ─────────────────────────────────────────────────────────────
def parse_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = (row.get("型号") or "").strip()
            if not model:
                continue
            rf_mhz = _tof(row.get("合成_载频MHz"))
            rf_ghz = (rf_mhz / 1000.0) if rf_mhz else None
            prf_hz = _tof(row.get("合成_PRFHz"))
            pw_us = _tof(row.get("合成_脉宽us"))
            band = (row.get("频段") or "").strip()
            mod = (row.get("合成_调制") or "").strip()
            platform = (row.get("平台") or "").strip()
            function = (row.get("用途") or "").strip()
            country = (row.get("国别") or "").strip()
            manufacturer = (row.get("制造商") or "").strip()
            system = (row.get("体制") or "").strip()
            category = (row.get("类别") or "").strip()

            parts = [f"型号 {model}", f"类别 {category}", f"国别 {country}",
                     f"制造商 {manufacturer}", f"体制 {system}", f"频段 {band}"]
            if rf_ghz is not None:
                parts.append(f"载频约 {rf_ghz}GHz")
            if prf_hz is not None:
                parts.append(f"脉冲重复频率 {prf_hz}Hz")
            if pw_us is not None:
                parts.append(f"脉宽 {pw_us}us")
            if mod:
                parts.append(f"调制方式 {mod}")
            if platform:
                parts.append(f"平台 {platform}")
            if function:
                parts.append(f"用途 {function}")
            content = "，".join(p for p in parts if p) + "。"

            rows.append({
                "source": "csv", "domain": "radar", "modality": "csv",
                "doc_id": row.get("id") or path.stem,
                "model": model, "category": category,
                "country": country, "manufacturer": manufacturer, "system": system,
                "band": band, "platform": platform, "function": function,
                "rf_ghz": rf_ghz, "prf_hz": prf_hz, "pw_us": pw_us, "mod": mod,
                "content": content,
            })
    return rows


# ─────────────────────────────────────────────────────────────
# 装备域：html（装备网页档案）
# ─────────────────────────────────────────────────────────────
def parse_html(path: Path) -> dict:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    name = (soup.h1.get_text(strip=True) if soup.h1 else path.stem)
    tags = [t.get_text(strip=True) for t in soup.find_all("span", class_="tag")]
    category = tags[0] if len(tags) > 0 else ""
    country = tags[1] if len(tags) > 1 else ""
    manufacturer = tags[2] if len(tags) > 2 else ""
    fields = {}
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            fields[tds[0].get_text(strip=True)] = tds[1].get_text(strip=True)
    radar = fields.get("雷达", "")
    archive_id = ""
    src = soup.find("div", class_="src")
    if src:
        m = re.search(r"档案编号[:：]\s*([A-Za-z0-9_-]+)", src.get_text())
        if m:
            archive_id = m.group(1)
    text = soup.get_text(" ", strip=True)
    return {
        "source": "html", "domain": "equipment", "modality": "html",
        "doc_id": archive_id or path.stem,
        "model": name, "category": category, "country": country,
        "manufacturer": manufacturer, "system": "", "band": "",
        "platform": name, "function": "", "radar": radar,
        "rf_ghz": None, "prf_hz": None, "pw_us": None, "mod": "",
        "content": text,
    }


# ─────────────────────────────────────────────────────────────
# 装备域：docx（装备文档档案，无 python-docx 用 zip+xml 抽取）
# ─────────────────────────────────────────────────────────────
def _docx_text(path: Path) -> str:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \n]+", "\n", text)
    return text.strip()


def parse_docx(path: Path) -> dict:
    text = _docx_text(path)
    flat = re.sub(r"\s+", "", text)  # 无空格版本，便于跨换行正则
    # 名称取首行（保留空格），"装备档案 EQ-XXXXX" 在第二行
    name = text.split("\n", 1)[0].strip()
    if "装备档案" in name:
        name = re.sub(r"装备档案.*$", "", name).strip()
    category = ""
    m = re.search(r"装备类别[:：](.+?)国别[:：]", flat)
    if m:
        category = m.group(1).strip()
    country = ""
    m = re.search(r"国别[:：](.+?)制造商[:：]", flat)
    if m:
        country = m.group(1).strip()
    manufacturer = ""
    m = re.search(r"制造商[:：](.+?)(?:二、技术参数|$)", flat)
    if m:
        manufacturer = m.group(1).strip()
    archive_id = ""
    m = re.search(r"档案编号[:：]?\s*([A-Za-z0-9_-]+)", text)
    if m:
        archive_id = m.group(1)
    return {
        "source": "docx", "domain": "equipment", "modality": "docx",
        "doc_id": archive_id or path.stem,
        "model": name, "category": category, "country": country,
        "manufacturer": manufacturer, "system": "", "band": "",
        "platform": name, "function": "", "radar": "",
        "rf_ghz": None, "prf_hz": None, "pw_us": None, "mod": "",
        "content": text,
    }


# ─────────────────────────────────────────────────────────────
# 媒体域：images / audio / video（元数据接入）
# ─────────────────────────────────────────────────────────────
def parse_media(path: Path, kind: str, eq_names: dict[str, str]) -> dict:
    stem = path.stem
    name = ""
    if kind == "video":
        num = stem.split("_")[-1].zfill(5)
        name = eq_names.get(num, "")
        content = f"装备视频：{name} 的演示视频（文件 {path.name}）" if name else f"装备视频文件 {path.name}"
    elif kind == "image":
        content = f"高清图像文件 {path.name}（分辨率 1920×1080）"
    else:  # audio
        content = f"雷达信号音频文件 {path.name}"
    return {
        "source": kind, "domain": "media", "modality": kind,
        "doc_id": stem, "model": "", "category": "", "country": "",
        "manufacturer": "", "system": "", "band": "", "platform": name,
        "function": "", "radar": "", "rf_ghz": None, "prf_hz": None,
        "pw_us": None, "mod": "", "content": content,
    }


# ─────────────────────────────────────────────────────────────
# 建库：records -> seed JSON
# ─────────────────────────────────────────────────────────────
def build_seed(records: list[dict]) -> dict:
    registry: dict[tuple[str, str], str] = {}

    def reg(typ: str, name: str):
        name = (name or "").strip()
        if not name:
            return None
        key = (typ, name)
        if key not in registry:
            registry[key] = ent_id(typ, name)
        return registry[key]

    def ents_of(r: dict) -> list[tuple[str, str, str]]:
        """返回 (entity_id, type, name) 列表（含主实体）。"""
        out = []
        if r["domain"] == "equipment":
            for typ, f in (("Platform", "model"), ("Radar", "radar"), ("Country", "country")):
                nm = (r.get(f) or "").strip()
                if nm:
                    eid = reg(typ, nm)
                    out.append((eid, typ, nm))
        elif r["domain"] == "media":
            pass
        else:  # radar
            for typ, f in (("Radar", "model"), ("Band", "band"), ("Modulation", "mod"),
                           ("Platform", "platform"), ("Function", "function"),
                           ("Country", "country")):
                nm = (r.get(f) or "").strip()
                if nm:
                    eid = reg(typ, nm)
                    out.append((eid, typ, nm))
        return out

    def edges_of(r: dict, doc_ents: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
        """返回 (rel, src_entity_id, dst_entity_id) 实体-实体边。"""
        ids = {t: eid for eid, t, _n in doc_ents}
        out = []
        if r["domain"] == "equipment":
            if "Radar" in ids and "Platform" in ids:
                out.append(("MOUNTED_ON", ids["Radar"], ids["Platform"]))
            if "Platform" in ids and "Country" in ids:
                out.append(("BELONGS_TO", ids["Platform"], ids["Country"]))
        elif r["domain"] == "radar":
            rl = [("MOUNTED_ON", "Radar", "Platform"), ("USES_MODULATION", "Radar", "Modulation"),
                  ("OPERATES_IN", "Radar", "Band"), ("HAS_FUNCTION", "Radar", "Function"),
                  ("BELONGS_TO", "Radar", "Country")]
            for rel, a, b in rl:
                if a in ids and b in ids:
                    out.append((rel, ids[a], ids[b]))
        return out

    parents, chunks, entities, edges, binds = [], [], [], [], []

    for i, r in enumerate(records):
        pid = f"p_{i:05d}"
        cid = f"c_{i:05d}"
        doc_ents = ents_of(r)

        parents.append({
            "parent_id": pid, "doc_id": r["doc_id"], "title": r.get("model", ""),
            "content": r["content"], "summary": r["content"][:200],
            "keywords": [n for _, _, n in doc_ents],
            "content_version": 1, "kb_id": "kb_public", "source_type": r["source"],
            "toa_text": "",
        })

        scalar = {k: r[k] for k in ("rf_ghz", "prf_hz", "pw_us") if r.get(k) is not None}
        chunks.append({
            "child_id": cid, "parent_id": pid, "content": r["content"],
            "kb_id": "kb_public", "content_version": 1,
            "kg_entity_ids": [e for e, _, _ in doc_ents],
            "section": r.get("model", ""),
            "span": [0, len(r["content"])],
            "scalar": scalar,
            "embedding_v2": hash_numeric_embedding({
                "rf": r.get("rf_ghz"), "pw": r.get("pw_us"), "prf": r.get("prf_hz")}),
            "embedding": hash_embedding(r["content"]),
            "sparse": hash_sparse(r["content"]),
        })
        for eid, _t, _n in doc_ents:
            binds.append({"src": eid, "dst": cid, "rel": "BINDS", "weight": 1.0,
                          "kb_id": "kb_public", "source_parent_id": pid,
                          "evidence": {"child_id": cid, "span": [0, len(r["content"])],
                                       "parent_id": pid}})
        for rel, src, dst in edges_of(r, doc_ents):
            edges.append({"src": src, "dst": dst, "rel": rel, "weight": 1.0,
                          "kb_id": "kb_public", "source_parent_id": pid,
                          "evidence": {"child_id": cid, "span": [0, len(r["content"])],
                                       "parent_id": pid}})

    for (typ, name), eid in registry.items():
        entities.append({
            "entity_id": eid, "name": name, "type": typ, "aliases": [],
            "kb_id": "kb_public", "embedding": hash_embedding(f"{name} {typ}"),
            "desc": f"{name}（{typ}）",
        })

    return {"parents": parents, "chunks": chunks, "entities": entities,
            "edges": edges + binds, "communities": []}


def main() -> int:
    records: list[dict] = []
    for p in sorted((DATE / "txt").glob("*.txt")):
        records.append(parse_txt(p))
    for p in sorted((DATE / "csv").glob("*.csv")):
        records.extend(parse_csv(p))

    # 装备档案：html + docx
    eq_records = []
    for p in sorted((DATE / "html").glob("*.html")):
        eq_records.append(parse_html(p))
    for p in sorted((DATE / "docx").glob("*.docx")):
        eq_records.append(parse_docx(p))
    # 档案编号(如 EQ-00000) → 装备名，供视频关联
    eq_names = {r["doc_id"].split("-")[-1].zfill(5): r["model"]
                for r in eq_records if r["doc_id"]}
    records.extend(eq_records)

    # 媒体：图片/音频/视频（元数据接入）
    for p in sorted((DATE / "images").glob("*.jpg")):
        records.append(parse_media(p, "image", eq_names))
    for p in sorted((DATE / "audio").glob("*.mp3")):
        records.append(parse_media(p, "audio", eq_names))
    for p in sorted((DATE / "video").glob("*.mp4")):
        records.append(parse_media(p, "video", eq_names))

    seed = build_seed(records)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("parents", "chunks", "entities", "edges", "communities"):
        (OUT / f"{name}.json").write_text(
            json.dumps(seed[name], ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "date/ (txt+csv+html+docx+images+audio+video, 除 PDF/SYNTHETIC)",
        "embed_channel": "hash", "dim": 1024,
        "docs": len(records), "chunks": len(seed["chunks"]),
        "entities": len(seed["entities"]), "edges": len(seed["edges"]),
        "note": "真实数据接入；图片/音频/视频为元数据接入；召回绝对值受 hash 嵌入限制",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    dc = Counter(r["domain"] for r in records)
    mc = Counter(r["modality"] for r in records)
    print(f"[ingest_date] docs={len(records)} chunks={len(seed['chunks'])} "
          f"entities={len(seed['entities'])} edges={len(seed['edges'])}")
    print(f"[ingest_date] 域分布={dict(dc)}")
    print(f"[ingest_date] 模态分布={dict(mc)}")
    print(f"[ingest_date] out={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
