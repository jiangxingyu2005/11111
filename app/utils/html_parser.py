"""HTML -> Markdown 提取器(beautifulsoup4 + lxml 解析器)。

把 HTML 字节解析为 markdown 文本:去噪音标签,保留标题/段落/列表/表格等结构。
编码由 bs4 对 bytes 自动检测(utf-8 / gbk / gb2312 等)。
"""

import re

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from ..config import settings

_HEADING_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# 这些块级标签渲染为独立段落(前后空行)
_BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "main"}


def _decode(content: bytes) -> str:
    """优先 UTF-8 解码;失败用 UnicodeDammit 检测(gbk/gb18030/big5 等)。

    bs4 对 bytes 的自动检测在无 <meta charset> 的纯中文页上会误判,故先严格 UTF-8。
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    from bs4 import UnicodeDammit

    return UnicodeDammit(content, ["gb18030", "gbk", "big5", "utf-8"]).unicode_markup


def parse_html_to_markdown(content: bytes) -> str:
    """HTML 字节 -> markdown 文本(去掉噪音标签,保留结构)。"""
    soup = BeautifulSoup(_decode(content), "lxml")

    # 去掉 <head>(title/meta/style 等)与配置的噪音标签
    for node in soup.find_all(["head"]):
        node.decompose()
    noise_tags = [t.strip().lower() for t in settings.html_noise_tags.split(",") if t.strip()]
    for node in soup.find_all(noise_tags):
        node.decompose()

    body = soup.body or soup
    md = _render_children(body)
    return _collapse_blank(md).strip()


def _render_children(node) -> str:
    """渲染节点下所有子节点并拼接。"""
    return "".join(_render_node(child) for child in node.children)


def _render_node(node) -> str:
    """把单个节点渲染为 markdown 片段。"""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = (node.name or "").lower()

    # 标题 h1-h6
    if name in _HEADING_LEVEL:
        return f"\n\n{'#' * _HEADING_LEVEL[name]} {_render_children(node).strip()}\n\n"

    # 预格式代码块
    if name == "pre":
        code = node.get_text().strip()
        return f"\n\n```\n{code}\n```\n\n" if code else ""

    # 块级元素
    if name in _BLOCK_TAGS:
        text = _render_children(node).strip()
        return f"\n\n{text}\n\n" if text else ""

    # 换行 / 分隔线
    if name == "br":
        return "\n"
    if name == "hr":
        return "\n\n---\n\n"

    # 列表 ul/ol
    if name in ("ul", "ol"):
        items = []
        for li in node.find_all("li", recursive=False):
            text = _render_children(li).strip()
            if text:
                items.append("- " + text)
        return ("\n\n" + "\n".join(items) + "\n\n") if items else ""

    # 表格
    if name == "table":
        return _render_table(node)

    # 行内强调
    if name in ("strong", "b"):
        return f"**{_render_children(node).strip()}**"
    if name in ("em", "i"):
        return f"*{_render_children(node).strip()}*"

    # 链接保留文字、图片取 alt
    if name == "a":
        return _render_children(node).strip()
    if name == "img":
        return (node.get("alt") or "").strip()

    # 默认递归子节点
    return _render_children(node)


def _render_table(table) -> str:
    """<table> -> markdown 表格(| 单元格 | 加 --- 分隔行)。"""
    rows = table.find_all("tr")
    if not rows:
        return ""
    lines = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        cell_texts = [
            _render_children(c).strip().replace("\n", " ").replace("|", "\\|")
            for c in cells
        ]
        lines.append("| " + " | ".join(cell_texts) + " |")
    if not lines:
        return ""
    ncols = max(len(r.find_all(["th", "td"])) for r in rows)
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    lines.insert(1, sep)
    return "\n\n" + "\n".join(lines) + "\n\n"


def _collapse_blank(text: str) -> str:
    """清理多余空白:行首尾空白、3 个以上连续换行压成 2 个。"""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
