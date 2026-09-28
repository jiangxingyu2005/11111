# 文档解析 + 清洗入库服务

FastAPI 服务:统一接收 `pdf / txt / csv / docx` 等文件上传,转发给本地 Docker 里的 MinerU 解析,再做一轮清洗,落盘为 md/json。详细设计见 [docs/方案.md](docs/方案.md)。

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # 按实际端口改 MINERU_API_BASE_URL

# 3. 确保 MinerU 容器已启动
curl http://127.0.0.1:8008/v1/health

# 4. 启动本服务
uvicorn app.main:app --reload --port 8001
```

## 调用示例

```bash
# 解析单个文件
curl -F "files=@sample.pdf" http://127.0.0.1:8001/parse

# 只返回落盘路径,不返回正文
curl -F "files=@sample.pdf" "http://127.0.0.1:8001/parse?return_content=false"

# 健康检查
curl http://127.0.0.1:8001/health
```

## 输出

解析结果落盘到 `{OUTPUT_DIR}/{doc_id}/`:

- `raw.md` / `raw.json` — MinerU 原始输出
- `clean.md` / `clean.json` — 清洗后结果
- `meta.json` — 来源文件名、解析时间、页数、清洗统计
