# 文档解析 + 清洗入库服务

FastAPI 服务:统一接收 `pdf / txt / csv / docx` 等文件上传,转发给本地 Docker 里的 MinerU 解析,再做一轮清洗,落盘为 md/json。详细设计见 [docs/方案.md](docs/方案.md)。

## Docker 全栈启动（推荐）

基础设施 + MinerU + qingxi + retrieval：

```bash
docker compose up -d --build
curl http://127.0.0.1:8003/health      # qingxi（宿主机 8001 常被占用，映射 8003）
curl http://127.0.0.1:8100/healthz     # retrieval
curl http://127.0.0.1:8008/v1/health   # MinerU
```

首次构建会下载 Python 依赖和 MinerU 模型，耗时较长。Windows 上 Milvus 映射为 `18530`（系统保留了 19530）。对象存储用 SeaweedFS（MinIO 社区版已停发）。嵌入模型为本地 **BGE-M3**（挂载 `retrieval/model/models/BAAI--bge-m3/snapshots/master`），换模型后需在容器内执行 `python -m app.graphrag.rebuild_vectors`。

## 快速启动（宿主机）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # 按实际端口改 MINERU_API_BASE_URL

# 3. 确保 MinerU 容器已启动
curl http://127.0.0.1:8008/v1/health

# 4. 启动本服务(--host 0.0.0.0 监听所有网卡,前端/其他机器才能通过本机 IP 访问)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## 调用示例

```bash
# 解析单个文件
curl -F "files=@sample.pdf" http://192.168.31.15:8001/parse

# 只返回落盘路径,不返回正文
curl -F "files=@sample.pdf" "http://192.168.31.15:8001/parse?return_content=false"

# 健康检查
curl http://192.168.31.15:8001/health
```

## 输出

解析结果落盘到 `{OUTPUT_DIR}/{doc_id}/`:

- `raw.md` / `raw.json` — MinerU 原始输出
- `clean.md` / `clean.json` — 清洗后结果
- `meta.json` — 来源文件名、解析时间、页数、清洗统计
