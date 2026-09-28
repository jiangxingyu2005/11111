"""文本块质量打分模块(独立于解析/清洗流程)。

对一段文本算一个 0~1 的质量分,四维加权:
- f_len   长度
- info    信息密度:字母/数字/中文字符占比
- content 内容词占比:jieba.posseg 词性为 n/v/a 的词占比,未装 jieba 时退化为信息密度近似
- hits    参数实体命中:装备领域正则

阈值与权重均通过参数覆盖(实际值由 config/.env 提供),模块内 DEFAULT_* 仅为 fallback。
纯函数模块,不 import 项目内其它模块,可独立运行。
"""

import re

# ---- 默认参数(fallback;实际配置由 config/.env 经调用方传入)----
DEFAULT_MIN_CHUNK_LEN = 20    # 长度维度下界(字符),块短于此则长度分记 0
DEFAULT_MAX_CHUNK_LEN = 512   # 长度维度上界(字符),块长于此则长度分记满
DEFAULT_W_LEN = 0.25          # 长度维度权重
DEFAULT_W_INFO = 0.30         # 信息密度维度权重
DEFAULT_W_CONTENT = 0.25      # 内容词占比维度权重
DEFAULT_W_HITS = 0.20         # 参数实体命中维度权重
DEFAULT_Q_QUARANTINE = 0.4    # 质量分隔离阈值(低于此分不分级保留)
DEFAULT_Q_KEEP = 0.6          # 质量分保留阈值(高于此分按高权重)

# 装备领域实体正则(可扩展,替换掉原雷达/电子战正则)
_ENTITY_PATTERNS = [
    re.compile(r"[A-Za-z]{1,6}[-_/]?\d{1,4}"),  # 型号: Iskander-M / MGM-140 / MK30-2 / Type 45
    re.compile(r"\d+(?:\.\d+)?\s*(?:mm|m|km|t|kg|kn|Ma|Mach|节|米|千米|吨|毫米)"),  # 数值+单位
    re.compile(r"导弹|驱逐舰|护卫舰|舰艇|战车|坦克|航母|潜艇|战斗机|轰炸机|火炮|雷达|鱼雷"),  # 装备类型词
]

# 按 markdown 标题(#{1,3})切块,标题保留在下一个块的开头
_HEADING_SPLIT_RE = re.compile(r"\n(?=#{1,3}\s)")
# 内联 base64 图片(如 ![](data:image/jpeg;base64,...)),切块前剥离
_INLINE_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")

# jieba 可选依赖:未安装时 content 维度降级为信息密度近似
try:
    import jieba.posseg as pseg  # noqa: F401
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 环境无 jieba 时降级
    pseg = None
    _HAS_JIEBA = False


def _f_len(text: str, min_chunk_len: int, max_chunk_len: int) -> float:
    """长度分:块越长越接近 max_chunk_len 分越高,<= min_chunk_len 为 0。"""
    n = len(text)
    if n <= min_chunk_len:
        return 0.0
    return min(max((n - min_chunk_len) / (max_chunk_len - min_chunk_len), 0.0), 1.0)


def _info(text: str) -> float:
    """信息密度:字母/数字/中文字符占比,过滤纯标点/空格水。"""
    n = len(text)
    if n == 0:
        return 0.0
    meaningful = sum(1 for ch in text if ch.isalnum() or "一" <= ch <= "鿿")
    return meaningful / n


def _content(text: str) -> float:
    """内容词占比:名词/动词/形容词(n/v/a 词性开头)占比。"""
    if _HAS_JIEBA:
        words = list(pseg.lcut(text))
        if not words:
            return 0.0
        content = sum(1 for w in words if w.flag and w.flag[0] in "nva")
        return content / len(words)
    # 降级:无 jieba 时用信息密度近似内容占比
    return _info(text)


def _hits(text: str) -> tuple[int, list[str]]:
    """参数实体命中数 + 命中明细。"""
    found: list[str] = []
    for pat in _ENTITY_PATTERNS:
        found.extend(pat.findall(text))
    return len(found), found


def _is_image_block(text: str) -> bool:
    """兜底识别图片/超长 base64 块(内联图片已提前剥离,这里防漏网)。"""
    t = text.strip()
    if not t:
        return False
    if re.search(r"!\[[^\]]*\]\((?:data:image/|https?://)", t):
        return True
    # 超长连续字母数字(典型 base64):> 500 字符且基本无空格/中文
    if len(t) > 500:
        alnum = sum(1 for ch in t if ch.isalnum())
        if alnum / len(t) > 0.95:
            return True
    return False


def _is_table_separator(text: str) -> bool:
    """判断块是否只是 markdown 表格分隔行(| --- | --- | 之类)。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    return all(re.fullmatch(r"[|:\- ]+", l) for l in lines)


def _strip_inline_images(md_text: str) -> str:
    """移除内联 base64 图片(无文本价值,且会污染长度/信息密度维度)。"""
    return _INLINE_IMAGE_RE.sub("", md_text)


def score(
    text: str,
    min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN,
    max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
    w_len: float = DEFAULT_W_LEN,
    w_info: float = DEFAULT_W_INFO,
    w_content: float = DEFAULT_W_CONTENT,
    w_hits: float = DEFAULT_W_HITS,
) -> dict:
    """对单块文本打分。图片/base64 块返回 skipped=True、q=0。"""
    if _is_image_block(text):
        return {
            "q": 0.0,
            "f_len": 0.0,
            "info": 0.0,
            "content": 0.0,
            "hits": 0,
            "hits_list": [],
            "length": len(text),
            "skipped": True,
            "content_approx": not _HAS_JIEBA,
        }

    fl = _f_len(text, min_chunk_len, max_chunk_len)
    info = _info(text)
    content = _content(text)
    hits, hits_list = _hits(text)
    q = w_len * fl + w_info * info + w_content * content + w_hits * min(hits / 3.0, 1.0)
    q = round(min(max(q, 0.0), 1.0), 4)

    return {
        "q": q,
        "f_len": round(fl, 4),
        "info": round(info, 4),
        "content": round(content, 4),
        "hits": hits,
        "hits_list": hits_list,
        "length": len(text),
        "skipped": False,
        "content_approx": not _HAS_JIEBA,
    }


def split_chunks(md_text: str) -> list[str]:
    """按 markdown 标题(#{1,3})切块,去掉空块和纯表格分隔行。"""
    text = md_text or ""
    chunks: list[str] = []
    for part in _HEADING_SPLIT_RE.split(text):
        p = part.strip()
        if not p:
            continue
        if _is_table_separator(p):
            continue
        chunks.append(p)
    return chunks


def score_markdown(md_text: str, **score_kwargs) -> list[dict]:
    """分块 + 逐块打分,返回结果列表(每个元素即 score() 的返回)。

    score_kwargs 透传给 score(),如 min_chunk_len/max_chunk_len/w_len/... 。
    """
    text = _strip_inline_images(md_text or "")
    return [score(c, **score_kwargs) for c in split_chunks(text)]


def score_markdown_total(md_text: str, **score_kwargs) -> float:
    """文件级质量分:各非跳过块按长度加权平均 q,空文件/全跳过返回 0.0。"""
    results = score_markdown(md_text, **score_kwargs)
    total_len = 0
    weighted = 0.0
    for r in results:
        if r.get("skipped"):
            continue
        length = r.get("length", 0)
        weighted += r["q"] * length
        total_len += length
    if total_len == 0:
        return 0.0
    return round(weighted / total_len, 4)
