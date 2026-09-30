# 前端对接快速上手(整体流程)

> **先读这份。** 它串起前端从「上传文档」到「提问拿答案」的完整链路。其余文件(01–10)是单个接口的字段级说明,用到再查。

Base URL:`http://<host>:8001`。无认证(内网服务)。所有查询均按 `kb_id`(知识库 id)隔离。

## 1. 整体流程

```
① 建知识库        POST /knowledge-bases         → 拿 kb_id
② 上传文档并解析  POST /parse                   → 自动清洗+打分+入图谱(同步),拿 doc_id
③ 提问(流式)     POST /graph/query/stream      → SSE 逐步返回检索步骤 + 流式答案   ← 前端主交互
③' 提问(一次性)   POST /graph/query             → 一次性返回 answer + hits + 图事实
④ 溯源(可选)     GET /graph/trace/files?kb_id= → 文件召回次数
```

三个关键约定:

1. **② 是同步的**:`/parse` 会等「清洗 → 打分 → 入 MinIO → 入 PG → 入 Neo4j/Milvus」**全部完成后才返回**。响应里 `success=true` 即代表该文档已可查询,无需轮询。
2. **GraphRAG 入库失败不影响 parse 成功**:若入库失败,`success` 仍为 `true`,但该文档 `in_kb=false`(不可查)。前端可经 `GET /knowledge-bases/{kb_id}/documents` 的 `in_kb` 字段判断。
3. **kb_id 是查询开关**:提问时传 `kb_id` 只检索该库;不传 = 全库。

---

## 2. 各步骤接口

### ① 建知识库

```http
POST /knowledge-bases
Content-Type: application/json

{"name": "雷达情报库", "description": "某演习雷达数据"}
```

响应 `201`:

```json
{"id": "dc0443e1-6950-4c68-832f-40b809321131", "name": "雷达情报库", "description": "...", "created_at": "...", "updated_at": "..."}
```

> 前端记住返回的 `id`,它就是后续所有接口的 `kb_id`。

### ② 上传文档并解析(自动入图谱)

```http
POST /parse
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | file[] | 是 | 一个或多个文件(支持多选) |
| `kb_id` | string(UUID) | 是 | 归属知识库 |
| `return_content` | query | 否 | 是否返回清洗后正文,默认 `true`;前端可传 `false` 省流量 |

响应 `200`:

```json
{
  "results": [
    {"filename": "rd_subset_02699.csv", "doc_id": "8f14e45f-...", "success": true, "page_count": null, "error": null}
  ]
}
```

> 一次传多个文件会逐文件返回结果;`success=false` 时看 `error`。**该请求耗时 = 文件清洗 + 向量化 + LLM 抽取**,大文件/多文件可能较久,前端需设较长超时或做上传进度。

文件类型与对应解析入口:`/parse`(文档/PDF/表格)、`/parse-image`、`/parse-audio`、`/parse-video`、`/parse-html`、`/parse-iq`(IQ 信号,不入图谱)。前端按文件类型路由即可。

### ③ 提问(流式,前端主交互)

```http
POST /graph/query/stream
Content-Type: application/json

{"query": "A国装备了哪些武器", "kb_id": "dc0443e1-..."}
```

响应 `Content-Type: text/event-stream`。事件序列固定为:

```
meta → progress(entity_match) → progress(recall) → progress(rerank)
     → [progress(graph_expand)] → results → token×N → done
```

| 事件 | `data` 关键字段 | 含义 |
|---|---|---|
| `meta` | `intent` `cond` | 意图分流(`retrieval`/`graph`)+ 结构化条件 |
| `progress` | `step` + 各步骤字段 | 检索每一步进度(见下) |
| `results` | `hits` `graph_facts` `matched_entities` | 最终召回子块 + 图关系事实 |
| `token` | `text` | 流式答案增量(逐 token,直接拼接展示) |
| `done` / `error` | — / `message` | 正常结束 / 失败 |

`progress` 的 `step` 取值与字段:

| step | 字段 | 含义 |
|---|---|---|
| `entity_match` | `matched_entities` `matched_ids` | 从问题里匹配到的实体 |
| `recall` | `route_a` `route_b` | A 路(实体锚定)/ B 路(纯向量)召回命中数 |
| `rerank` | `top` | 重排后保留条数 |
| `graph_expand` | `expanded` `facts` | 图扩展实体数 + 一跳关系事实(仅关系型问题) |

#### 前端对接代码(fetch 读流)

> 接口是 **POST**,浏览器 `EventSource` 只支持 GET,所以必须用 `fetch` + `ReadableStream` 读流。

```js
async function askStream(query, kbId, { onToken, onStep, onResults, onError } = {}) {
  const res = await fetch('/graph/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, kb_id: kbId }),
  });
  if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {   // SSE 以空行分隔事件
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleSse(block);
    }
  }
  buf += decoder.decode();  // 收尾
  if (buf.trim()) handleSse(buf);

  function handleSse(block) {
    let event = 'message', data = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event: ')) event = line.slice(7);
      else if (line.startsWith('data: ')) data = line.slice(6);
    }
    const p = JSON.parse(data);
    switch (event) {
      case 'token':    onToken?.(p.text); break;      // 流式答案
      case 'progress': onStep?.(p); break;            // 检索步骤
      case 'results':  onResults?.(p); break;         // 命中子块
      case 'error':    onError?.(p.message); break;
      case 'done':     break;
    }
  }
}
```

### ③' 提问(一次性,不流式)

```http
POST /graph/query
Content-Type: application/json

{"query": "A国装备了哪些武器", "kb_id": "dc0443e1-..."}
```

响应 `200`:

```json
{
  "intent": "graph",
  "cond": {"modulation_type": null, "freq_band": null},
  "answer": "A国装备的武器(雷达)包括:1. AN/APG-68 2. 某型相控阵雷达 …",
  "hits": [{"child_id": "...#1", "doc_id": "...", "content": "...", "_score": 0.58}],
  "matched_entities": ["ent_41916512"],
  "expanded_entities": [],
  "graph_facts": ["AN/SPY-1D --[BELONGS_TO]--> A国"]
}
```

| 字段 | 说明 |
|---|---|
| `intent` | `retrieval`(纯检索)/ `graph`(关系型,走图扩展) |
| `answer` | 带 `[n]` 引用的最终答案 |
| `hits` | 召回子块(`content` 是原文片段,`_score` 是相似度分) |
| `graph_facts` | 关系型问题时命中实体的一跳关系事实(`实体 --[关系]--> 实体`) |

### ④ 溯源(可选)

```http
GET /graph/trace/files?kb_id=dc0443e1-...
```

```json
{"files": [{"doc_id": "8f14e45f-...", "file_name": "rd.csv", "recall_count": 2}]}
```

---

## 3. 最小可用流程(前端照着走)

```js
// 1. 建库
const kb = await fetch('/knowledge-bases', {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({name: '雷达情报库'}),
}).then(r => r.json());                       // → kb.id

// 2. 上传文件(自动清洗 + 入图谱,同步完成)
const fd = new FormData();
fd.append('files', fileInput.files[0]);
fd.append('kb_id', kb.id);
const parsed = await fetch('/parse', {method: 'POST', body: fd}).then(r => r.json());
// parsed.results[0].doc_id / .success

// 3. 提问(流式)
let answer = '';
await askStream('A国装备了哪些武器', kb.id, {
  onStep: p => console.log('step', p.step, p),       // 实时展示双路检索步骤
  onToken: t => { answer += t; renderAnswer(answer); }, // 逐 token 渲染答案
  onResults: r => renderHits(r.hits),
  onError: e => console.error(e),
});
```

## 4. 常见问题

- **提问查不到东西?** 先确认文档 `in_kb=true`(`GET /knowledge-bases/{kb_id}/documents`);GraphRAG 入库失败时 `success` 仍为 true 但 `in_kb=false`。
- **`answer` 为空/不相关?** 关系型问题(「X 有哪些 Y」「X 装备了什么」)走图边,依赖图谱里的关系数据;纯检索问题依赖子块向量召回。
- **流式一直不结束?** `done` 事件是结束标志;异常会收到 `error` 事件而非 HTTP 5xx。
