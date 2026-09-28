"""配置纪律测试：防"配了没生效 / 缺变量静默启动 / 后端通道串味"。"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_missing_required_var_refuses_startup(monkeypatch):
    """缺必需变量必须拒绝启动（禁止带默认值静默启动）。

    注意：不能靠"删 env"来验证——根目录 .env 会兜底补值。
    必须显式关掉 env_file 再构造 Settings，才是真正测到"缺变量"。
    """
    import pytest as _pt
    from pydantic import ValidationError

    for k in list(os.environ):
        if k.startswith(("POSTGRES_", "MILVUS_", "NEO4J_")):
            monkeypatch.delenv(k, raising=False)
    from app.config import Settings

    with _pt.raises(ValidationError):
        Settings(_env_file=None)


def test_dev_overrides_only_relax_thresholds():
    """dev_overrides 只允许覆盖阈值/超时，不得改算法语义参数（RRF k / top_k）。"""
    from app.config import retrieval_cfg

    cfg = retrieval_cfg()
    ov = cfg.get("dev_overrides", {}).get("fake", {})
    forbidden = {"fusion.k", "fusion.top_k", "hybrid.rrfranker.k", "reranker.top_k_out"}
    assert not (set(ov) & forbidden), f"dev 覆盖动了语义参数: {set(ov) & forbidden}"


def test_fake_backend_uses_lowered_threshold():
    """fake 通道必须把 τ 降下来，否则哈希嵌入的余弦全部被杀成 0 召回。"""
    os.environ.setdefault("RETRIEVAL_BACKEND", "fake")
    os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
    os.environ.setdefault("POSTGRES_USER", "ro")
    os.environ.setdefault("POSTGRES_PASSWORD", "x")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
    os.environ.setdefault("NEO4J_USER", "ro")
    os.environ.setdefault("NEO4J_PASSWORD", "x")
    from app.config import effective_cfg, reset_caches

    reset_caches()
    cfg = effective_cfg()
    assert cfg["_backend"] == "fake"
    assert cfg["milvus"]["dense"]["primary"]["threshold"] < 0.2, "fake 通道 τ 必须下调"
    assert cfg["fusion"]["k"] == 60, "RRF k 在任何通道都不得变（语义参数）"


def test_all_yaml_contracts_parse():
    """config/*.yaml 是契约：任一文件写坏都必须炸在启动前，而不是炸在运行时。"""
    import yaml

    d = ROOT / "config"
    files = sorted(d.glob("*.yaml"))
    assert files, "config 目录为空"
    parsed = []
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            yaml.safe_load(f)        # 写坏的 yaml 在此抛异常（启动前暴露，而非运行时）
            parsed.append(p.name)
    assert len(parsed) == len(files)


def test_api_schemas_are_valid_jsonschema():
    import jsonschema

    for name in ("retrieval_response.v1.schema.json", "subgraph_view.v1.schema.json"):
        s = __import__("json").loads((ROOT / "api" / name).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(s)
