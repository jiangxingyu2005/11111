# 文档下载 `/documents`

## `GET /documents/{doc_id}/download` — 下载源文件

返回脱敏后的源文件（文本类脱敏；视频/音频原样）。

**URL**：`GET http://192.168.31.15:8001/documents/{doc_id}/download`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | UUID | 是 | 文档 id |

**请求示例**：`GET http://192.168.31.15:8001/documents/8f14e45f-.../download`

**响应示例**：文件二进制流（`Content-Disposition` 附原文件名）。

404：文档或源文件不存在。

## `GET /documents/{doc_id}/desensitize` — 脱敏下载（别名）

行为同 `/download`。

**URL**：`GET http://192.168.31.15:8001/documents/{doc_id}/desensitize`

**请求数据**（路径参数）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | UUID | 是 | 文档 id |

**请求示例**：`GET http://192.168.31.15:8001/documents/8f14e45f-.../desensitize`

**响应示例**：同上。
