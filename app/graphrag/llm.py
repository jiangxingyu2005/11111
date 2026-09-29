"""LLM 客户端(ems common/llm.py 移植,配置源改 qingxi .env;关系类型由调用方传入)。

两阶段抽取 + 生成回答的统一通道。provider=mock 时用确定性规则抽取,供离线跑通全管线;
provider=openai/qwen 时走 OpenAI 兼容接口(function calling 约束 JSON)——qwen 为国产模型,
本地 vLLM 起 OpenAI 兼容端点,base_url 指向内网,数据不出网。
同步实现:mock 纯计算;openai/qwen 走网络,由调用方经 asyncio.to_thread 委托。
"""

from __future__ import annotations

import re
from typing import Optional

from .config import build_settings


class LLMClient:
    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or build_settings()
        self.provider = self.settings["llm"]["provider"]
        # openai 与 qwen(国产,本地 vLLM)都走 OpenAI 兼容接口;mock 走确定性正则
        self.is_llm = self.provider in ("openai", "qwen")

    # ---- 阶段一:实体抽取(返回结构化实体列表,未归一化/未发号)----
    def extract_entities(self, text: str, entity_types: Optional[list] = None,
                         entity_defs: Optional[list] = None) -> list[dict]:
        if self.is_llm:
            return self._extract_entities_openai(text, entity_types, entity_defs)
        return _mock_extract_entities(text, entity_types or [])

    # ---- 阶段二:关系抽取(实体对 + 证据句)----
    def extract_relations(self, text: str, entities: list[dict],
                          relation_types: Optional[list] = None,
                          fewshot: Optional[list] = None) -> list[dict]:
        if self.is_llm:
            return self._extract_relations_openai(text, entities, relation_types or [], fewshot)
        return _mock_extract_relations(text, entities, relation_types or [])

    def generate_answer(self, question: str, contexts: list[dict]) -> str:
        if self.is_llm:
            return self._answer_openai(question, contexts)
        return _mock_answer(question, contexts)

    # ---- 消歧灰区裁判(匹配器③,provider=openai/qwen 才激活)----
    def judge_same_entity(self, name_a: str, name_b: str, entity_type: str) -> bool:
        """判断两个实体名是否同一实体。mock 恒 False(灰区不并,确定性优先)。"""
        if self.is_llm:
            return self._judge_same_openai(name_a, name_b, entity_type)
        return False

    def _judge_same_openai(self, name_a: str, name_b: str, entity_type: str) -> bool:
        schema = {
            "type": "object",
            "properties": {"same": {"type": "boolean"}},
            "required": ["same"],
        }
        result = _call_function(
            self._client(), self.settings["llm"], "judge_same_entity",
            "判断两个实体名是否指向同一电磁信号实体，输出布尔值",
            f"实体类型：{entity_type}\n名称 A：{name_a}\n名称 B：{name_b}", schema)
        return bool(result.get("same", False))

    # ---------- openai/qwen 实现(OpenAI 兼容) ----------
    def _client(self):
        from openai import OpenAI  # 延迟导入
        s = self.settings["llm"]
        return OpenAI(base_url=s.get("base_url"), api_key=s.get("api_key") or "sk-dummy")

    def _extract_entities_openai(self, text, entity_types, entity_defs):
        schema = {
            "type": "object",
            "properties": {"entities": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "entity_type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "properties": {"type": "object"}, "confidence": {"type": "number"},
                }, "required": ["name", "entity_type", "confidence"]},
            }},
            "required": ["entities"],
        }
        return _call_function(self._client(), self.settings["llm"], "extract_entities",
                              _entity_instruction(entity_types, entity_defs), text, schema)["entities"]

    def _extract_relations_openai(self, text, entities, relation_types, fewshot):
        schema = {
            "type": "object",
            "properties": {"relations": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "head": {"type": "string"}, "tail": {"type": "string"},
                    "rel_type": {"type": "string"}, "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                }, "required": ["head", "tail", "rel_type", "confidence"]},
            }},
            "required": ["relations"],
        }
        entity_names = [e["name"] for e in entities]
        instruction = _relation_instruction(entity_names, relation_types)
        if fewshot:
            instruction += f"\n\n关系判定示例：\n{fewshot}"
        return _call_function(self._client(), self.settings["llm"], "extract_relations",
                              instruction, text, schema)["relations"]

    def _answer_openai(self, question, contexts):
        ctx = "\n".join(f"[{i}] {c.get('content', '')}" for i, c in enumerate(contexts))
        resp = self._client().chat.completions.create(
            model=self.settings["llm"]["model"],
            messages=[{"role": "system", "content": "你是电磁信号情报助手，回答需引用来源编号。"},
                      {"role": "user", "content": f"参考资料：\n{ctx}\n\n问题：{question}"}],
            temperature=0.0,
            max_tokens=int(self.settings["llm"].get("max_tokens", 4096)),
        )
        return resp.choices[0].message.content


def _call_function(client, s: dict, name: str, instruction: str, text: str, schema: dict) -> dict:
    """抽取统一入口:重试 + 解析容错(准确度优先)。

    两条通道:
    - use_function_calling=True(默认):function calling(tools/tool_choice)强约束 JSON。
      Qwen(vLLM 本地)偶发网络抖动或返回非纯 JSON,这里重试并回退从 content 提取。
    - use_function_calling=False:小模型(如 ollama qwen2.5:3b)不支持 function calling,
      把 schema 内嵌进 prompt 要求输出 JSON,再走 _parse_json 解析。两者复用同一解析容错。
    """
    import time
    use_fc = bool(s.get("use_function_calling", True))
    max_retries = int(s.get("max_retries", 2))
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if use_fc:
                resp = client.chat.completions.create(
                    model=s["model"], temperature=0.0,
                    max_tokens=int(s.get("max_tokens", 4096)),
                    messages=[{"role": "system", "content": instruction},
                              {"role": "user", "content": text}],
                    tools=[{"type": "function", "function": {"name": name, "parameters": schema}}],
                    tool_choice={"type": "function", "function": {"name": name}},
                )
                msg = resp.choices[0].message
                raw = None
                if getattr(msg, "tool_calls", None):
                    raw = msg.tool_calls[0].function.arguments
                if not raw and getattr(msg, "content", None):
                    raw = msg.content
            else:
                # 小模型(如 ollama qwen2.5:3b)对裸 JSON Schema 理解不稳:把
                # {"type":"object","properties":{...}} 当输出格式照抄,生成嵌套结构导致解析失败。
                # 这里改用「具体扁平示例」作格式说明,小模型照抄示例即可输出合法扁平 JSON。
                top_key = (schema.get("required") or [name])[0]
                sys_prompt = (instruction
                              + f"\n\n只输出一个 JSON 对象,不要输出任何其它文字。输出格式示例:\n"
                              + _output_example(name))
                resp = client.chat.completions.create(
                    model=s["model"], temperature=0.0,
                    max_tokens=int(s.get("max_tokens", 4096)),
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": text}],
                )
                raw = resp.choices[0].message.content
            result = _parse_json(raw)
            # 兜底:小模型偶发只输出数组、漏掉 {key:[...]} 包装,统一包回对象,避免调用方 KeyError
            if isinstance(result, list):
                top_key = (schema.get("required") or [name])[0]
                result = {top_key: result}
            return result
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))  # 指数退避 0.5s / 1s
                continue
    raise last_err  # pragma: no cover - last_err 在异常路径必非空


def _output_example(name: str) -> str:
    """给 JSON-prompt 通道一个具体扁平输出示例(小模型照抄示例比理解 JSON Schema 稳)。

    示例字段与 _extract_*_openai 的 schema 一一对应;小模型按示例的扁平结构输出即可。
    """
    if name == "extract_entities":
        return ('{"entities": [{"name": "AN/TPQ-50", "entity_type": "辐射源", '
                '"aliases": [], "properties": {}, "confidence": 0.9}]}')
    if name == "extract_relations":
        return ('{"relations": [{"head": "AN/TPQ-50", "tail": "地面车载平台", '
                '"rel_type": "EQUIPPED_ON", "confidence": 0.9, "evidence": "装备于地面车载平台"}]}')
    if name == "judge_same_entity":
        return '{"same": true}'
    return '{"' + name + '": []}'


def _parse_json(raw: str) -> dict:
    """把 LLM 返回解析成 dict:剥 ```json 围栏,截取首个平衡 {} 块。"""
    import json
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("LLM 返回空结果")
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 返回非 JSON: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


def _entity_instruction(entity_types: Optional[list], entity_defs: Optional[list]) -> str:
    """实体抽取指令:有完整定义(entity_defs)时讲清每类含义,否则退回类型名列表。"""
    if entity_defs:
        lines = []
        for d in entity_defs:
            name, desc = d.get("name", ""), d.get("description", "")
            lines.append(f"- {name}：{desc}" if desc else f"- {name}")
        return "抽取电磁信号领域实体，类型限定如下：\n" + "\n".join(lines)
    return f"抽取电磁信号领域实体，类型限定：{entity_types}"


def _relation_instruction(entity_names: list, relation_types: list) -> str:
    """关系抽取指令:讲清每类关系含义,rel_type 字段用英文标识(与校验/Neo4j 边一致)。"""
    lines = []
    for r in relation_types:
        name, en, desc = r.get("name", ""), r.get("en", ""), r.get("description", "")
        label = f"{name}（{en}）" if name and en else (name or en)
        lines.append(f"- {label}：{desc}" if desc else f"- {label}")
    return (f"从实体 {entity_names} 中抽取关系，关系类型限定如下"
            f"（rel_type 字段用括号内英文标识）：\n" + "\n".join(lines))


# ====================================================================
# mock 实现:确定性规则抽取(无网络、可离线验证)
# ====================================================================

def _mock_extract_entities(text: str, entity_types: list) -> list[dict]:
    """mock 实体抽取:恒返回空。

    当前受控本体 10 类(原始信号/信号样本/信号簇/辐射源/平台/组织/工作状态/目标/事件/情报产品)
    均无确定性正则,无法离线规则抽取,实体抽取需真 LLM(provider=openai/qwen)。
    保留此 stub 仅为维持 extract_entities 非 LLM 分支的调用签名一致。
    """
    return []


def _mock_extract_relations(text: str, entities: list[dict], relation_types: list) -> list[dict]:
    sentences = re.split(r"[。！？\n；;]", text)
    out: list[dict] = []
    seen: set = set()
    for sent in sentences:
        if not sent.strip():
            continue
        # 命中哪个关系类型(含触发词)
        for rt in relation_types:
            matched = [w for w in rt["trigger_words"] if w in sent]
            if not matched:
                continue
            in_sent = [e["name"] for e in entities if e["name"] in sent]
            if len(in_sent) < 2:
                continue
            head, tail = in_sent[0], in_sent[1]
            key = (head, tail, rt["en"])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "head": head, "tail": tail, "rel_type": rt["en"],
                "confidence": 0.85, "evidence": sent.strip(),
            })
    return out


def _mock_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "未检索到相关资料。"
    lines = [f"根据检索结果（{len(contexts)} 条）："]
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] {c.get('content', '')[:80]}")
    lines.append(f"针对问题「{question}」，以上片段提供支撑。")
    return "\n".join(lines)


def get_llm() -> LLMClient:
    return LLMClient()
