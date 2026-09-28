# -*- coding: utf-8 -*-
"""
装备知识图谱构建与 Neo4j 批量导入脚本
=====================================

数据源: C:/Users/A/Desktop/数据/data
  csv/    500 个 CSV(57 个 cat_* 分类表 + 442 个 subset_* 子集表), 统一 77 列, 8195 行
  txt/    5000 份装备情报报告 (report_00000.txt ~ report_04999.txt)
  html/   2000 份装备档案 (eq_00000.html ~ eq_01999.html)
  docx/   600 份装备档案  (eq_00000.docx ~ eq_00599.docx)
  images/ 300 张装备配图 (eq_00000.png ~ eq_00299.png)
  audio/  200 条语音资料 (eq_00000.mp3 ~ eq_00199.mp3)

图模型
------
节点:
  (:Category     {name})            装备类别, 57 个
  (:Country      {name})            国别, 按 "/" 拆分为多国
  (:Manufacturer {name})            制造商(保留原始字符串, 不拆分, 防止破坏联名名称)
  (:Equipment    {id, 型号, ...})   装备, id 全局唯一(已验证 id<->型号 一一对应, 无属性冲突)
  (:Scenario     {name})            评估场景
  (:Unit         {name})            报告编制单位
  (:Report       {no, ...})         情报报告, no=报告编号(INT-xxxxx)
  (:Archive      {no, ...})         HTML 装备档案, no=档案编号(EQ-xxxxx), 经内嵌 id 关装备
  (:DocxArchive  {no, ...})         DOCX 装备档案, 编号体系与 html 相互独立(实测 EQ-00001
                                    在两种载体中对应不同装备), 经标题型号关装备
  (:Image        {file})            配图, 按编号关 HTML 档案 (eq_00000.png -> EQ-00000)
  (:Audio        {file})            语音, 按编号关 HTML 档案 (eq_00000.mp3 -> EQ-00000)

关系:
  (Equipment)-[:归属类别]->(Category)
  (Equipment)-[:所属国家]->(Country)
  (Manufacturer)-[:制造]->(Equipment)
  (Report)-[:评估对象]->(Equipment)
  (Report)-[:所属场景]->(Scenario)
  (Unit)-[:编制]->(Report)
  (Archive)-[:档案记录]->(Equipment)
  (DocxArchive)-[:档案记录]->(Equipment)
  (Image)-[:配图]->(Archive)
  (Audio)-[:语音资料]->(Archive)

说明:
  * images/hd_*.jpg、images_hd/、video/、video_hd/ 与 iq/ 的编号体系与档案编号
    无法可靠对应, 为保证严谨不入库(脚本中留有统计输出)。
  * 全部写入使用 MERGE, 幂等, 可重复执行; 先建唯一约束再批量 UNWIND 导入。

用法:
  # 1. 仅解析校验, 不连接 Neo4j (无需安装 neo4j 驱动)
  python build_kg.py --dry-run

  # 2. 导入 Neo4j
  pip install neo4j
  python build_kg.py --uri bolt://localhost:7687 --user neo4j --password 你的密码

  # 可选: --database neo4j  --batch-size 1000  --data-dir "C:/Users/A/Desktop/数据/data"
"""

import argparse
import csv
import glob
import html as html_mod
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 配置

DEFAULT_DATA_DIR = r"C:/Users/A/Desktop/数据/data"

# CSV 中除以下固定列外, 其余非空列全部作为 Equipment 属性
FIXED_COLS = {"id", "类别", "型号", "国别", "制造商", "来源"}

REPORT_RE = {
    "no":    re.compile(r"^报告编号:\s*(\S+)\s*$", re.M),
    "model": re.compile(r"^装备型号:\s*(.+?)\s*$", re.M),
    "cat":   re.compile(r"^装备类别:\s*(.+?)\s*$", re.M),
    "cm":    re.compile(r"^国别/制造商:\s*(.+?)\s*$", re.M),
    "scene": re.compile(r"^评估场景:\s*(.+?)\s*$", re.M),
    "unit":  re.compile(r"^编制单位:\s*(.+?)\s*$", re.M),
    "eval":  re.compile(r"^综合评估:\s*(.+?)\s*$", re.M),
    "src":   re.compile(r"^资料来源:\s*(.+?)\s*$", re.M),
}
PARAM_BLOCK_RE = re.compile(
    r"主要参数:\s*\n(.*?)\n?-{10,}", re.S)
PARAM_LINE_RE = re.compile(r"^\s{1,}([^\s:：]+)\s*[:：]\s*(.+?)\s*$", re.M)

HTML_ID_RE = re.compile(r"<td>id</td><td>([^<]+)</td>")
HTML_ARC_RE = re.compile(r"档案编号:\s*(EQ-\d+)")
DOCX_ARC_RE = re.compile(r"装备档案\s*(EQ-\d+)")
DOCX_MODEL_RE = re.compile(r"<w:t[^>]*>([^<]+)</w:t>")
EQ_FILE_RE = re.compile(r"^eq_(\d+)\.(png|mp3)$")


class KGError(Exception):
    """数据校验失败时抛出"""


# ---------------------------------------------------------------- 解析: CSV -> Equipment / Category / Country / Manufacturer

def parse_csv_dir(csv_dir):
    """解析全部 CSV, 按 id 去重合并装备, 校验属性冲突."""
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise KGError(f"未找到 CSV 文件: {csv_dir}")

    equipment = {}          # id -> dict(属性)
    sources = defaultdict(list)   # id -> [来源文件名]
    n_rows = 0
    conflicts = []

    for fp in files:
        fname = os.path.basename(fp)
        with open(fp, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames:
                raise KGError(f"CSV 缺少 id 列: {fname}")
            for ln, row in enumerate(reader, start=2):
                n_rows += 1
                row = {k: (v or "").strip() for k, v in row.items()}
                eid = row.get("id", "")
                if not eid:
                    raise KGError(f"{fname} 第{ln}行 id 为空")
                if not row.get("型号") or not row.get("类别"):
                    raise KGError(f"{fname} 第{ln}行 型号/类别 为空 (id={eid})")

                if eid not in equipment:
                    equipment[eid] = {k: v for k, v in row.items() if v}
                else:
                    base = equipment[eid]
                    for k, v in row.items():
                        if not v:
                            continue
                        if k in base and base[k] != v:
                            conflicts.append((eid, k, base[k], v, fname))
                        else:
                            base.setdefault(k, v)
                if fname not in sources[eid]:
                    sources[eid].append(fname)

    if conflicts:
        sample = "; ".join(f"{c[0]}.{c[1]}: {c[2]}!={c[3]}({c[4]})" for c in conflicts[:5])
        raise KGError(f"同一 id 属性冲突 {len(conflicts)} 处, 示例: {sample}")

    # id <-> 型号 一一对应校验
    m2id = defaultdict(set)
    for eid, e in equipment.items():
        m2id[e["型号"]].add(eid)
    bad = {m: s for m, s in m2id.items() if len(s) > 1}
    if bad:
        raise KGError(f"型号对应多个 id: {list(bad.items())[:5]}")

    for eid, e in equipment.items():
        e["csv来源文件"] = sorted(sources[eid])

    stats = {"csv文件数": len(files), "csv总行数": n_rows, "装备数": len(equipment)}
    return equipment, stats


# ---------------------------------------------------------------- 解析: TXT -> Report / Scenario / Unit

def parse_report(fp):
    text = open(fp, encoding="utf-8").read()
    fname = os.path.basename(fp)

    def grab(key, required=True):
        m = REPORT_RE[key].search(text)
        if not m:
            if required:
                raise KGError(f"{fname}: 缺少字段 [{key}]")
            return ""
        return m.group(1).strip()

    rep = {
        "no":    grab("no"),
        "model": grab("model"),
        "cat":   grab("cat"),
        "scene": grab("scene"),
        "unit":  grab("unit"),
        "eval":  grab("eval"),
        "src":   grab("src"),
        "文件":   fname,
    }
    cm = grab("cm")  # 仅用于交叉校验, 格式: "国别 / 制造商"
    rep["_cm"] = cm

    # 主要参数块
    params = {}
    bm = PARAM_BLOCK_RE.search(text)
    if bm:
        for k, v in PARAM_LINE_RE.findall(bm.group(1)):
            params[k.strip()] = v.strip()
    rep["params"] = params
    return rep


def parse_txt_dir(txt_dir, equipment):
    files = sorted(glob.glob(os.path.join(txt_dir, "report_*.txt")))
    reports = []
    seen_no = set()
    model2eq = {e["型号"]: eid for eid, e in equipment.items()}

    for fp in files:
        rep = parse_report(fp)
        if rep["no"] in seen_no:
            raise KGError(f"报告编号重复: {rep['no']}")
        seen_no.add(rep["no"])

        # 交叉校验: 报告型号必须存在于装备库, 且类别一致
        eid = model2eq.get(rep["model"])
        if eid is None:
            raise KGError(f"{rep['文件']}: 型号[{rep['model']}]不在装备库中")
        if equipment[eid]["类别"] != rep["cat"]:
            raise KGError(
                f"{rep['文件']}: 类别不一致 报告[{rep['cat']}] vs 装备[{equipment[eid]['类别']}]")
        rep["equipment_id"] = eid
        reports.append(rep)

    return reports, {"报告数": len(reports)}


# ---------------------------------------------------------------- 解析: HTML / DOCX -> Archive

def parse_html_dir(html_dir, equipment):
    files = sorted(glob.glob(os.path.join(html_dir, "eq_*.html")))
    archives = {}   # EQ-xxxxx -> dict
    for fp in files:
        fname = os.path.basename(fp)
        text = open(fp, encoding="utf-8").read()
        m_id = HTML_ID_RE.search(text)
        m_arc = HTML_ARC_RE.search(text)
        if not m_id or not m_arc:
            raise KGError(f"{fname}: 无法解析 id 或档案编号")
        eid, arc = m_id.group(1).strip(), m_arc.group(1).strip()
        if eid not in equipment:
            raise KGError(f"{fname}: 装备 id[{eid}] 不在装备库中")
        if arc in archives:
            raise KGError(f"{fname}: 档案编号重复 {arc}")
        archives[arc] = {"no": arc, "equipment_id": eid,
                         "html文件": f"html/{fname}"}
    return archives, {"html档案数": len(archives)}


def parse_docx_dir(docx_dir, equipment):
    """解析 DOCX 档案。注意: docx 的 EQ 编号与 html 是相互独立的序列
    (实测 EQ-00001 在 html=A07 / docx=AS01), 故单独成库, 通过标题型号关联装备。"""
    files = sorted(glob.glob(os.path.join(docx_dir, "eq_*.docx")))
    model2eq = {e["型号"]: eid for eid, e in equipment.items()}
    docx_archives = {}
    for fp in files:
        fname = os.path.basename(fp)
        try:
            with zipfile.ZipFile(fp) as z:
                xml = z.read("word/document.xml").decode("utf-8")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise KGError(f"{fname}: docx 读取失败 ({exc})")

        m_arc = DOCX_ARC_RE.search(xml)
        if not m_arc:
            raise KGError(f"{fname}: 未找到档案编号")
        arc = m_arc.group(1)
        if arc in docx_archives:
            raise KGError(f"{fname}: docx 档案编号重复 {arc}")

        # 标题(第一个 w:t)即型号
        texts = [html_mod.unescape(t) for t in DOCX_MODEL_RE.findall(xml)]
        if not texts:
            raise KGError(f"{fname}: 未找到任何文本")
        model = texts[0].strip()
        eid = model2eq.get(model)
        if eid is None:
            raise KGError(f"{fname}: 型号[{model}]不在装备库中")

        docx_archives[arc] = {"no": arc, "equipment_id": eid,
                              "docx文件": f"docx/{fname}"}
    return docx_archives, {"docx档案数": len(docx_archives)}


# ---------------------------------------------------------------- 解析: 媒体 -> Image / Audio (按编号关档案)

def parse_media(images_dir, audio_dir, archives):
    images, audios, skipped = [], [], 0
    for d, kind in ((images_dir, "png"), (audio_dir, "mp3")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = EQ_FILE_RE.match(fn)
            if not m or m.group(2) != kind:
                skipped += 1
                continue
            arc = f"EQ-{int(m.group(1)):05d}"
            if arc not in archives:
                skipped += 1
                continue
            rec = {"file": f"{os.path.basename(d)}/{fn}", "archive_no": arc}
            (images if kind == "png" else audios).append(rec)
    return images, audios, {"配图数": len(images), "语音数": len(audios),
                            "跳过媒体文件数": skipped}


# ---------------------------------------------------------------- 图数据装配

def split_countries(raw):
    """国别按 '/' 拆分(多国联合研制), 去空白去重."""
    return [c.strip() for c in raw.split("/") if c.strip()]


def assemble(equipment, reports, archives, docx_archives, images, audios):
    """把解析结果装配成节点/关系字典(全部为可 JSON 序列化的 dict 列表)."""
    nodes = defaultdict(list)
    rels = defaultdict(list)

    cat_set, cty_set, mfr_set = set(), set(), set()
    scn_set, unit_set = set(), set()

    for eid, e in sorted(equipment.items()):
        props = {"id": eid}
        for k, v in e.items():
            if k in ("id", "类别", "国别", "制造商") or not v:
                continue
            props[k] = v
        nodes["Equipment"].append(props)

        cat_set.add(e["类别"])
        rels["归属类别"].append({"eid": eid, "cat": e["类别"]})
        for c in split_countries(e.get("国别", "")):
            cty_set.add(c)
            rels["所属国家"].append({"eid": eid, "country": c})
        mfr = e.get("制造商", "")
        if mfr:
            mfr_set.add(mfr)
            rels["制造"].append({"mfr": mfr, "eid": eid})

    nodes["Category"] = [{"name": c} for c in sorted(cat_set)]
    nodes["Country"] = [{"name": c} for c in sorted(cty_set)]
    nodes["Manufacturer"] = [{"name": m} for m in sorted(mfr_set)]

    for rep in reports:
        props = {"no": rep["no"], "综合评估": rep["eval"],
                 "资料来源": rep["src"], "文件": rep["文件"]}
        for k, v in rep["params"].items():
            props[f"参数_{k}"] = v
        nodes["Report"].append(props)
        scn_set.add(rep["scene"])
        unit_set.add(rep["unit"])
        rels["评估对象"].append({"rno": rep["no"], "eid": rep["equipment_id"]})
        rels["所属场景"].append({"rno": rep["no"], "scene": rep["scene"]})
        rels["编制"].append({"unit": rep["unit"], "rno": rep["no"]})

    nodes["Scenario"] = [{"name": s} for s in sorted(scn_set)]
    nodes["Unit"] = [{"name": u} for u in sorted(unit_set)]

    for arc in archives.values():
        nodes["Archive"].append({"no": arc["no"], "html文件": arc["html文件"]})
        rels["档案记录"].append({"ano": arc["no"], "eid": arc["equipment_id"]})

    for arc in docx_archives.values():
        nodes["DocxArchive"].append({"no": arc["no"], "docx文件": arc["docx文件"]})
        rels["DOCX档案记录"].append({"ano": arc["no"], "eid": arc["equipment_id"]})

    for im in images:
        nodes["Image"].append({"file": im["file"]})
        rels["配图"].append({"file": im["file"], "ano": im["archive_no"]})
    for au in audios:
        nodes["Audio"].append({"file": au["file"]})
        rels["语音资料"].append({"file": au["file"], "ano": au["archive_no"]})

    return nodes, rels


# ---------------------------------------------------------------- Neo4j 导入

CONSTRAINTS = [
    ("Category", "name"), ("Country", "name"), ("Manufacturer", "name"),
    ("Equipment", "id"), ("Scenario", "name"), ("Unit", "name"),
    ("Report", "no"), ("Archive", "no"), ("DocxArchive", "no"),
    ("Image", "file"), ("Audio", "file"),
]

NODE_KEY = {"Equipment": "id", "Category": "name", "Country": "name",
            "Manufacturer": "name", "Scenario": "name", "Unit": "name",
            "Report": "no", "Archive": "no", "DocxArchive": "no",
            "Image": "file", "Audio": "file"}

# 关系 -> (语句模板); 键均来自上面的唯一约束
REL_CYPHER = {
    "归属类别": ("MATCH (a:Equipment {id: r.eid}), (b:Category {name: r.cat}) "
                 "MERGE (a)-[:归属类别]->(b)"),
    "所属国家": ("MATCH (a:Equipment {id: r.eid}), (b:Country {name: r.country}) "
                 "MERGE (a)-[:所属国家]->(b)"),
    "制造":   ("MATCH (a:Manufacturer {name: r.mfr}), (b:Equipment {id: r.eid}) "
               "MERGE (a)-[:制造]->(b)"),
    "评估对象": ("MATCH (a:Report {no: r.rno}), (b:Equipment {id: r.eid}) "
                 "MERGE (a)-[:评估对象]->(b)"),
    "所属场景": ("MATCH (a:Report {no: r.rno}), (b:Scenario {name: r.scene}) "
                 "MERGE (a)-[:所属场景]->(b)"),
    "编制":   ("MATCH (a:Unit {name: r.unit}), (b:Report {no: r.rno}) "
               "MERGE (a)-[:编制]->(b)"),
    "档案记录": ("MATCH (a:Archive {no: r.ano}), (b:Equipment {id: r.eid}) "
                 "MERGE (a)-[:档案记录]->(b)"),
    "DOCX档案记录": ("MATCH (a:DocxArchive {no: r.ano}), (b:Equipment {id: r.eid}) "
                     "MERGE (a)-[:档案记录]->(b)"),
    "配图":   ("MATCH (a:Image {file: r.file}), (b:Archive {no: r.ano}) "
               "MERGE (a)-[:配图]->(b)"),
    "语音资料": ("MATCH (a:Audio {file: r.file}), (b:Archive {no: r.ano}) "
                 "MERGE (a)-[:语音资料]->(b)"),
}


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def import_to_neo4j(nodes, rels, uri, user, password, database, batch_size):
    from neo4j import GraphDatabase  # 延迟导入, dry-run 不依赖驱动

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        # 1. 唯一约束
        for label, key in CONSTRAINTS:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_{key}_uniq IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
        print("[约束] 已创建/确认 10 个唯一约束")

        # 2. 节点 (顺序无关, MERGE 幂等)
        for label in ("Category", "Country", "Manufacturer", "Scenario", "Unit",
                      "Equipment", "Report", "Archive", "DocxArchive",
                      "Image", "Audio"):
            rows = nodes.get(label, [])
            if not rows:
                continue
            key = NODE_KEY[label]
            cypher = (f"UNWIND $rows AS r MERGE (n:{label} {{{key}: r.{key}}}) "
                      f"SET n = r")
            n_done = 0
            for chunk in batched(rows, batch_size):
                session.execute_write(
                    lambda tx, c=chunk: tx.run(cypher, rows=c).consume())
                n_done += len(chunk)
            print(f"[节点] {label:<12} {n_done}")

        # 3. 关系 (节点已就位)
        for rtype, cypher_body in REL_CYPHER.items():
            rows = rels.get(rtype, [])
            if not rows:
                continue
            cypher = f"UNWIND $rows AS r {cypher_body}"
            n_done = 0
            for chunk in batched(rows, batch_size):
                session.execute_write(
                    lambda tx, c=chunk: tx.run(cypher, rows=c).consume())
                n_done += len(chunk)
            print(f"[关系] {rtype:<10} {n_done}")

        # 4. 导入后校验
        print("\n[校验] Neo4j 实际计数:")
        for label in NODE_KEY:
            cnt = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  (:{label}) = {cnt}")
        for rtype in REL_CYPHER:
            cnt = session.run(
                f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS c").single()["c"]
            print(f"  -[:{rtype}]-> = {cnt}")
    driver.close()


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="装备知识图谱构建与 Neo4j 批量导入")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="")
    ap.add_argument("--database", default="neo4j")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true",
                    help="仅解析校验并打印统计, 不连接 Neo4j")
    args = ap.parse_args()

    d = args.data_dir
    print("=" * 60)
    print("阶段 1/3  解析 CSV (装备/类别/国别/制造商)")
    equipment, st = parse_csv_dir(os.path.join(d, "csv"))
    print(" ", st)

    print("阶段 2/3  解析 TXT/HTML/DOCX/媒体 (报告/档案/配图/语音)")
    reports, st1 = parse_txt_dir(os.path.join(d, "txt"), equipment)
    archives, st2 = parse_html_dir(os.path.join(d, "html"), equipment)
    docx_archives, st3 = parse_docx_dir(os.path.join(d, "docx"), equipment)
    images, audios, st4 = parse_media(os.path.join(d, "images"),
                                      os.path.join(d, "audio"), archives)
    print(" ", st1, st2, st3, st4)

    nodes, rels = assemble(equipment, reports, archives, docx_archives,
                           images, audios)

    print("\n解析结果统计:")
    for label, rows in nodes.items():
        print(f"  节点 {label:<12} {len(rows)}")
    for rtype, rows in rels.items():
        print(f"  关系 {rtype:<10} {len(rows)}")

    # 期望计数硬性校验(基于对数据源的实测, 防解析静默丢数据)
    expect = {"Equipment": 130, "Category": 57, "Report": 5000}
    for label, n in expect.items():
        assert len(nodes[label]) == n, f"{label} 数量异常: {len(nodes[label])} != {n}"

    if args.dry_run:
        print("\n[dry-run] 校验全部通过, 未连接 Neo4j。")
        return

    if not args.password:
        sys.exit("错误: 导入模式必须提供 --password")

    print("\n阶段 3/3  导入 Neo4j")
    import_to_neo4j(nodes, rels, args.uri, args.user, args.password,
                    args.database, args.batch_size)
    print("\n完成。")


if __name__ == "__main__":
    main()
