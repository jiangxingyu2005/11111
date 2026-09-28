# ============================================================
# retrieval 服务 —— 常用命令（Windows 无 make 时按 README 的等价命令执行）
# ============================================================
PY ?= python

.PHONY: help init seed stub serve smoke test up down init-real seed-real clean

help:
	@echo "init        安装依赖 + 生成种子数据（fake 通道）"
	@echo "seed        重新生成 data/seed/*.json"
	@echo "stub        起模型 stub 服务（8001：embed/rerank/LLM）"
	@echo "serve       起 retrieval 服务（8100）"
	@echo "smoke       一键端到端冒烟（含 5 个降级注入场景）"
	@echo "test        pytest 全量"
	@echo "up/down     docker compose 起/停真三库"
	@echo "init-real   真环境建库（Milvus collection + PG 表 + Neo4j 约束）"
	@echo "seed-real   把 seed 灌进真三库"

init:
	$(PY) -m pip install -r requirements.txt
	$(PY) scripts/gen_seed.py

seed:
	$(PY) scripts/gen_seed.py

stub:
	$(PY) services/model_stub.py --port 8001

serve:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8100

smoke:
	$(PY) scripts/smoke.py

test:
	$(PY) -m pytest -q

# ── 真环境（RETRIEVAL_BACKEND=real）──
up:
	docker compose up -d

down:
	docker compose down

init-real:
	$(PY) scripts/init_milvus.py
	docker compose exec neo4j cypher-shell -u neo4j -p dev_only -f /import/001_constraints.cypher

seed-real:
	$(PY) scripts/seed_real.py

clean:
	rm -rf logs/trace logs/trace_smoke logs/trace_test .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
