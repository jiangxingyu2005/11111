"""文档内指代消解:把「该雷达」「此型雷达」这类提及对应回前文已抽出的实体。

用在入库的两个位置:
  - 实体侧(merge_mentions):实体抽取后,把「雷达」这类代称实体并回前文全名实体,不单独成节点;
  - 关系侧(DocCoref.resolve / head_in_text):关系校验前,把关系端点里的代称换成前文实体。

规则(只在单篇文档内生效,不写全局别名、不改原文):
  1. 去掉指示词(该/此/这/本/上述…)与量词(型/部/台…),得到中心词;
  2. 候选的名字以中心词结尾,且比它长(中心词匹配);
  3. 只往前找:候选首次出现在当前章节或之前;
  4. 唯一命中才采用,零个或多个都不猜。
「其」「它」这类没有中心词的代词不在规则覆盖范围内。
"""

from __future__ import annotations

from typing import Optional

from .ids import normalize_name

_DEMONSTRATIVES = ("上述", "前述", "所述", "该", "此", "这", "本", "那")
_CLASSIFIERS = ("型", "部", "台", "架", "艘", "套", "个", "种", "类", "款")
_MIN_HEAD_LEN = 2   # 中心词至少 2 个字,「机」「站」这类单字不当中心词


def strip_mention(name: str) -> str:
    """「该型雷达」->「雷达」;没有指示词的名字原样返回(已归一化)。"""
    s = normalize_name(name)
    for d in _DEMONSTRATIVES:
        if s.startswith(d) and len(s) > len(d):
            s = s[len(d):]
            if s[:1] in _CLASSIFIERS and len(s) - 1 >= _MIN_HEAD_LEN:
                s = s[1:]
            break
    return s


def _suffix_of(head: str, name: str) -> bool:
    """head 是 name 的真后缀(name 更长,且以 head 结尾)。"""
    return len(head) >= _MIN_HEAD_LEN and len(name) > len(head) and name.endswith(head)


class DocCoref:
    """单篇文档的指代消解器:记住每个实体首次出现在第几节。"""

    def __init__(self, entities: list, parents: list[dict]):
        self.order = {p["parent_id"]: i for i, p in enumerate(parents)}
        self.texts = [normalize_name(f"{p.get('title') or ''}{p.get('content') or ''}") for p in parents]
        self.entities = list(entities)
        self.first_pos: dict[str, int] = {}
        for e in self.entities:
            key = normalize_name(e.name)
            pos = next((i for i, t in enumerate(self.texts) if key and key in t), None)
            if pos is None:   # 原文里找不到字面全名(LLM 改写过),退回抽出它的章节
                pos = self.order.get(e.source_parent_id, len(parents))
            self.first_pos[e.entity_id] = pos

    def resolve(self, mention: str, pos: int, allowed_types: Optional[set] = None) -> tuple:
        """提及 -> ("ok", 实体) | ("ambiguous", [候选]) | ("none", None)。"""
        head = strip_mention(mention)
        cands = [e for e in self.entities
                 if _suffix_of(head, normalize_name(e.name))
                 and self.first_pos.get(e.entity_id, 10 ** 9) <= pos
                 and (not allowed_types or e.entity_type in allowed_types)]
        if len(cands) == 1:
            return "ok", cands[0]
        if cands:
            return "ambiguous", cands
        return "none", None

    def head_in_text(self, entity, pos: int) -> Optional[str]:
        """该实体名的某个真后缀(中心词)是否出现在第 pos 节,返回最长的那个。

        用来证明「这一节确实提到了前文的这个实体」(如本节有「该雷达」,实体是对空搜索雷达)。
        """
        name = normalize_name(entity.name)
        text = self.texts[pos] if 0 <= pos < len(self.texts) else ""
        for i in range(1, len(name) - _MIN_HEAD_LEN + 1):
            suffix = name[i:]
            if suffix in text:
                return suffix
        return None


def merge_mentions(entities: list, parents: list[dict]) -> tuple[list, list[dict]]:
    """实体侧:代称实体并回前文全名实体。返回 (保留的实体, 待审留痕)。

    触发条件:实体名(去指示词后)是本文档另一个更长实体名的结尾,如「雷达」之于「对空搜索雷达」。
      - 唯一回指:并入目标(补属性,类型以目标为准),本身不建节点;
      - 多个候选 / 前文没有全名:不建节点(否则各文档的「雷达」会合成一个全局枢纽),进待审;
      - 没有更长的名字:不是代称,正常保留(如文中只出现「干扰机」)。
    不按类型过滤:代称常被标错类型(实测「雷达」被标成平台,真身是辐射源)。
    并入时不把代称写进目标别名:别名会进全局注册表,用于跨文档匹配。
    """
    coref = DocCoref(entities, parents)
    kept = {e.entity_id: e for e in entities}
    notes: list[dict] = []
    bare_heads: set[str] = set()
    # 长名先处理:「该型雷达」先并入「对空搜索雷达」,「雷达」才不会并进一个随后被删掉的中间实体
    for e in sorted(entities, key=lambda x: -len(normalize_name(x.name))):
        head = strip_mention(e.name)
        longer = [o for o in kept.values() if o.entity_id != e.entity_id
                  and _suffix_of(head, normalize_name(o.name))]
        if not longer:
            continue
        coref.entities = [o for o in kept.values() if o.entity_id != e.entity_id]
        pos = coref.order.get(e.source_parent_id, len(parents))
        status, res = coref.resolve(e.name, pos)
        kept.pop(e.entity_id)
        bare_heads.add(head)
        if status == "ok":
            for k, v in (e.properties or {}).items():
                res.properties.setdefault(k, v)
            reason = f"指代合并:「{e.name}」→「{res.name}」(并入,不单独建节点)"
        elif status == "ambiguous":
            reason = (f"指代未消解:「{e.name}」有多个前文候选("
                      + "、".join(c.name for c in res) + "),不建节点")
        else:
            reason = f"指代未消解:「{e.name}」在前文没有对应的全名实体,不建节点"
        notes.append({"parent_id": e.source_parent_id, "stage": "entity",
                      "payload": {"name": e.name, "entity_type": e.entity_type,
                                  "confidence": e.confidence},
                      "reason": reason, "attempts": 1})
    for e in kept.values():   # 目标实体自带的裸中心词别名(如「雷达」)同样不能进全局注册表
        e.aliases = [a for a in (e.aliases or []) if strip_mention(a) not in bare_heads]
    return list(kept.values()), notes
