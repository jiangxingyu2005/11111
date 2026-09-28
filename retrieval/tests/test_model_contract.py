"""E1/E2/E3 模型契约位测试 —— 防「参数位配了但没生效」的静默漂移。

背景：模型身份只登记在 config/retrieval.yaml:models。
"yaml 写 BAAI/bge-m3、实际跑 hash"是本项目最危险的漂移形态：
链路全绿、契约合法、零报错，但召回全是噪声。所以声明必须可校验。
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    ALLOWED_BACKENDS, retrieval_cfg, validate_model_contract,
)

VALID = ["hash", "bge_m3"]


def _cfg():
    return copy.deepcopy(retrieval_cfg())


def test_yaml_declares_all_three_model_slots():
    """E1/E2/E3 的参数位必须都在，缺一个就是"到时候临时找地方填"。"""
    m = retrieval_cfg().get("models") or {}
    for kind in ("embed", "rerank", "llm"):
        assert kind in m, f"models.{kind} 缺失"
        assert m[kind].get("id"), f"models.{kind}.id 未登记"
        assert m[kind].get("backend") in ALLOWED_BACKENDS[kind], \
            f"models.{kind}.backend 不在白名单 {sorted(ALLOWED_BACKENDS[kind])}"
    assert m["embed"]["dim"] == 1024, "BGE-M3 dense 必须 1024 维"
    assert m["embed"]["id"] == "BAAI/bge-m3", "与离线 qingxi 侧必须同模型"


def test_real_backend_with_hash_embed_is_rejected():
    """红线：真环境跑哈希嵌入 = 静默噪声召回，宁可拒绝启动也不许放行。"""
    cfg = _cfg()
    cfg["models"]["embed"]["backend"] = "hash"
    problems = validate_model_contract(cfg, "real", allow_placeholder=False)
    assert any(p.startswith("RETRIEVAL_BACKEND=real") for p in problems), \
        "real 通道 + hash 嵌入必须被判为硬性问题"


def test_fake_backend_with_hash_embed_is_allowed():
    """dev 通道允许哈希嵌入（验逻辑用），但不得被当成质量问题放过。"""
    cfg = _cfg()
    cfg["models"]["embed"]["backend"] = "hash"
    assert validate_model_contract(cfg, "fake", allow_placeholder=False) == []


def test_dim_must_match_model_id():
    """dim 与模型 id 绑定：写成 768 会和 collection 维度对不上。"""
    cfg = _cfg()
    cfg["models"]["embed"]["dim"] = 768
    problems = validate_model_contract(cfg, "fake", allow_placeholder=False)
    assert any("必须是 1024" in p for p in problems)


def test_sparse_switches_must_move_together():
    """sparse 两处开关不一致 = "以为开了 hybrid 其实没开"。"""
    cfg = _cfg()
    cfg["models"]["embed"]["sparse_enabled"] = True
    cfg["milvus"]["sparse"]["enabled"] = False
    problems = validate_model_contract(cfg, "fake", allow_placeholder=False)
    assert any("sparse_enabled" in p for p in problems)


def test_reranker_model_id_consistent():
    """reranker.model 是历史字段，必须与 models.rerank.id 保持一致。"""
    cfg = _cfg()
    cfg["reranker"]["model"] = "some-other-model"
    problems = validate_model_contract(cfg, "fake", allow_placeholder=False)
    assert any("models.rerank.id" in p for p in problems)


def test_unknown_backend_rejected():
    cfg = _cfg()
    cfg["models"]["embed"]["backend"] = "openai_clip"
    problems = validate_model_contract(cfg, "fake", allow_placeholder=False)
    assert any("非法" in p for p in problems)
