# 清洗解析

统一 multipart/form-data。除 `return_content` 外,参数均同 `/parse`。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | file[] | 是 | 上传文件 |
| `kb_id` | UUID | 是 | 所属知识库 id(须已存在,否则 404) |
| `return_content` | query bool | 否 | 是否返回清洗后 markdown 正文,默认 `true` |

## `POST /parse` — 文档解析

MinerU 解析 → 清洗 → 打分 → 落盘 → MinIO → 登记 `documents` →(若 `GRAPHRAG_ENABLED=true`)GraphRAG 入库。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse \
  -F "files=@sample.pdf" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529" \
  -F "return_content=false"
```

**传出示例** `ParseResponse`:

```json
{
  "results": [
    {
      "filename": "sample.pdf",
      "doc_id": "8f14e45f-...",
      "success": true,
      "page_count": 12,
      "clean_md": "# ...",
      "files": {
        "raw_md": "output/8f14e45f-.../raw.md",
        "raw_json": "output/8f14e45f-.../raw.json",
        "clean_md": "output/8f14e45f-.../clean.md",
        "clean_json": "output/8f14e45f-.../clean.json",
        "meta": "output/8f14e45f-.../meta.json"
      },
      "stats": {"blocks_before": 120, "blocks_after": 100, "removed_images": 3, "removed_page_numbers": 0, "removed_headers": 0, "removed_short_blocks": 0, "removed_duplicates": 2},
      "error": null
    }
  ]
}
```

**失败示例**(单文件失败不中断整批):

```json
{"filename": "a.pdf", "doc_id": "", "success": false, "error": "处理失败: ..."}
```

## `POST /parse-image` — 图片解析

MinerU 优先,失败/空结果降级 PaddleOCR;BLIP 语义描述(英→中)并入正文。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse-image \
  -F "files=@sample.png" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529"
```

**传出示例**:同 `ParseResponse`(见上)。

## `POST /parse-audio` — 音频解析

ASR(faster-whisper)转文字 → 清洗 → 打分 → 入库(不过 MinerU)。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse-audio \
  -F "files=@sample.wav" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529"
```

**传出示例**:同 `ParseResponse`。

## `POST /parse-video` — 视频解析

抽帧 → 去重 → 逐帧图片链路 → 合并 → 清洗 → 打分 → 入库。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse-video \
  -F "files=@sample.mp4" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529"
```

**传出示例**:同 `ParseResponse`。

## `POST /parse-html` — HTML 解析

提取正文(标题/段落/列表/表格转 markdown)→ 清洗 → 打分 → 入库。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse-html \
  -F "files=@sample.html" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529"
```

**传出示例**:同 `ParseResponse`。

## `POST /parse-iq` — IQ 信号解析

H5 → SNR 清洗 → IQ 补全 → 信号处理 → 26 特征中文描述入库。**不入图谱**(`in_kb` 保持 `false`)。

**传入示例**:

```bash
curl -X POST http://127.0.0.1:8001/parse-iq \
  -F "files=@test.h5" \
  -F "kb_id=2db770f2-7a09-4f04-9f07-fce5e6bf6529" \
  -F "return_content=false"
```

**传出示例** `IqParseResponse`(在 `FileResult` 基础上多 IQ 中间结果):

```json
{
  "results": [
    {
      "filename": "test.h5",
      "doc_id": "97303300-b568-4526-9533-21138a1e0dac",
      "success": true,
      "page_count": null,
      "clean_md": "============================================================\n雷达信号特征描述报告\n【幅度特征】\n  - 幅度均值: 0.792370\n...\n【调制识别】\n  - 调制类型: Unknown\n  - 调制ID:   14\n",
      "files": {"raw_md": "...", "raw_json": "...", "clean_md": "...", "clean_json": "...", "meta": "..."},
      "stats": {"blocks_before": 44, "blocks_after": 34, "removed_images": 0, "removed_page_numbers": 0, "removed_headers": 0, "removed_short_blocks": 0, "removed_duplicates": 2},
      "error": null,
      "snr": {"total_segments": 128, "valid_segments": 128, "removed_segments": 0, "mean_snr": 0.000156294439239755, "min_snr": -0.20274522904308617, "snr_threshold_db": -10.0},
      "attenuation_db": -0.0366,
      "charts": {"time_domain": {"time": [...], "iq": [...]}, "...": "..."},
      "features": {...},
      "prediction": {...},
      "param_guess": {...},
      "iq_raw": [...],
      "iq_clean": [...],
      "iq_masked": [...],
      "iq_recovered": [...],
      "iq_meta": {...}
    }
  ]
}
```

## `GET /health` — 健康检查

**传入示例**:

```bash
curl http://127.0.0.1:8001/health
```

**传出示例**:

```json
{"status": "ok", "mineru": {"ok": true, "detail": "..."}}
```
