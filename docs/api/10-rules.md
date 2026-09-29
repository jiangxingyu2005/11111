# 预设推理规则 `/rules`

> 对应需求 (7)「触发预设推理规则」与 (8)「规则触发详情」。规则是用户自定义的「条件 → 动作」评判项：
> 检索完成后由**确定性规则引擎**求值（先算后写），命中的规则作为结构化事实注入 LLM 流式提示词；
> LLM 只负责翻译、解释、下结论，**不负责计算规则**（红线：规则命中必须可复现，不能由模型"猜"）。

## 通用约定

| 项 | 说明 |
|---|---|
| 主键 `id` | 语义字符串（非 UUID），形如 `"R_001"`，由调用方显式指定，重复 → 409 |
| 逻辑删除 | `DELETE` 置 `deleted_at`，不物理删除；查询均过滤已删除 |
| 作用域 `kb_id` | 空 = 全局规则；非空 = 仅对该知识库生效 |
| 优先级 `priority` | 越大越靠前输出（检索侧规则引擎按 priority 降序取规则） |
| 生效 | 规则增删改后**即时生效**（检索侧每次现查，无缓存） |

## 条件 DSL（`condition`）

`condition` 是一个 JSONB 对象，支持 `all`（AND，须全部满足）与 `any`（OR，至少一个满足）：

```json
{
  "all": [ { "field": "rerank_score", "op": "lt", "value": 0.5 } ],
  "any": [
    { "field": "content", "op": "contains", "value": "火控雷达" },
    { "field": "kg_confidence", "op": "lt", "value": 0.7 }
  ]
}
```

- `all` / `any` 可只写其一；**两者都空（或缺省）时规则不触发**（fail-closed，避免误标全量）。
- 每个子句 `{field, op, value}`；`value` 按字段取 number 或 string。

### 可用字段（`field`）

| 字段 | 类型 | 取值来源 |
|---|---|---|
| `rerank_score` | number | 重排分数（0~1，匹配度） |
| `rrf_score` | number | 外层 RRF 融合分数 |
| `kg_confidence` | number | 入库时写入 `parent_document.kg_confidence`（实体抽取置信度；无则子句恒不满足） |
| `kb_id` | string | 命中所属知识库 id |
| `section` | string | 命中所在章节标题 |
| `content` | string | 父块正文 |

> 说明：`kg_confidence` 是**入库时自设置的置信度表**（存 PG），只有规则里显式提到时才被读取对比，
> 与需求「只有在用户规则中提到置信度时才拿置信度表对比」一致。

### 可用算子（`op`）

| op | 适用字段 | 含义 |
|---|---|---|
| `lt` / `lte` / `gt` / `gte` | number | 数值小于/小于等于/大于/大于等于 |
| `eq` / `ne` | number, string | 等于 / 不等于 |
| `contains` | string | 字段值包含 `value` |

> 字段取不到值（None）时子句**恒为 False**（fail-closed），不会因"无置信度"而误触发。

## 动作 DSL（`action`）

`action` 描述命中后要做的标记（只做标记，不产生副作用——不改变召回序、不过滤、不提权）：

```json
{ "type": "mark", "label": "疑似优先", "severity": "high", "hint": "建议优先核查该辐射源" }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 动作类型，当前固定 `mark`（标记） |
| `label` | string | 标记文案（报告里展示） |
| `severity` | string | 严重度：`info` / `warning` / `high` |
| `hint` | string | 给 LLM 的提示语（可选） |

---

## `POST /rules` — 新建

**URL**：`POST http://192.168.31.15:8001/rules`

**请求数据**（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 语义主键，形如 `"R_001"`（重复 → 409） |
| `name` | string | 是 | 规则名（≤200） |
| `description` | string | 否 | 描述，默认 `""` |
| `condition` | object | 是 | 谓词（见上 DSL） |
| `action` | object | 是 | 命中动作（见上 DSL） |
| `priority` | int | 否 | 优先级，默认 `0` |
| `enabled` | bool | 否 | 是否启用，默认 `true` |
| `kb_id` | UUID | 否 | 作用域知识库，空 = 全局 |

**请求示例**：

```json
{
  "id": "R_001",
  "name": "火控雷达疑似优先",
  "description": "重排分偏低但正文命中火控雷达且 KG 置信度不足时标记疑似优先",
  "condition": {
    "all": [ { "field": "rerank_score", "op": "lt", "value": 0.5 } ],
    "any": [
      { "field": "content", "op": "contains", "value": "火控雷达" },
      { "field": "kg_confidence", "op": "lt", "value": 0.7 }
    ]
  },
  "action": { "type": "mark", "label": "疑似优先", "severity": "high", "hint": "建议优先核查该辐射源" },
  "priority": 10,
  "enabled": true,
  "kb_id": null
}
```

**响应示例**（201）：返回 `RuleRead`（含 `created_at` / `updated_at`）。

## `GET /rules` — 列表

**URL**：`GET http://192.168.31.15:8001/rules`

按 `priority` 降序返回未删除规则数组。示例：

```json
[
  { "id": "R_001", "name": "火控雷达疑似优先", "description": "…",
    "condition": {"all": [], "any": []},
    "action": {"type": "mark", "label": "疑似优先", "severity": "high", "hint": "…"},
    "priority": 10, "enabled": true, "kb_id": null,
    "created_at": "2026-09-29T10:00:00Z", "updated_at": "2026-09-29T10:00:00Z" }
]
```

## `GET /rules/{rule_id}` — 单个

**URL**：`GET http://192.168.31.15:8001/rules/R_001`

返回单个 `RuleRead`；不存在（或已删除）→ 404。

## `PUT /rules/{rule_id}` — 更新

**URL**：`PUT http://192.168.31.15:8001/rules/R_001`

**请求数据**（`application/json`，仅传要改字段；`None` = 不修改）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 否 | 规则名 |
| `description` | string | 否 | 描述 |
| `condition` | object | 否 | 谓词 |
| `action` | object | 否 | 动作 |
| `priority` | int | 否 | 优先级 |
| `enabled` | bool | 否 | 是否启用 |
| `kb_id` | UUID | 否 | 作用域知识库 |

**请求示例**：

```json
{ "enabled": false }
```

**响应示例**：更新后的 `RuleRead`；不存在 → 404。

## `DELETE /rules/{rule_id}` — 逻辑删除

**URL**：`DELETE http://192.168.31.15:8001/rules/R_001`

**响应示例**：204 无 body（置 `deleted_at`，不物理删除）。

---

## 与检索链路的衔接（retrieval 侧）

1. 检索完成（rerank → 父块回捞）后，规则引擎现查 `rules` 表（`enabled=true AND deleted_at IS NULL`）。
2. 对每条规则、每条命中证据独立判条件，产出 `rule_hits`（命中规则 + 命中证据编号 `[n]`）。
3. `rule_hits` 通过 SSE 的 `rules` 事件与 `done.rule_hits` 返回，并注入 LLM 提示词的「预设规则触发情况」段。
4. 规则读取失败 → 记 warning 并视为"无规则"（fail-closed），**不影响**检索主链路。
