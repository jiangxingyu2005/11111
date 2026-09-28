# 文档下载 `/documents`

## `GET /documents/{doc_id}/download` — 下载源文件

返回脱敏后的源文件(文本类脱敏;视频/音频原样)。

**传入示例**:

```bash
curl -O http://127.0.0.1:8001/documents/8f14e45f-.../download
```

**传出示例**:文件二进制流(`Content-Disposition` 附原文件名)。

404:文档或源文件不存在。

## `GET /documents/{doc_id}/desensitize` — 脱敏下载(别名)

行为同 `/download`。

**传入示例**:

```bash
curl -O http://127.0.0.1:8001/documents/8f14e45f-.../desensitize
```

**传出示例**:同上。
