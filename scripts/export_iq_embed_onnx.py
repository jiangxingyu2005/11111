"""一次性脚本:把 dc 的 IQ 分类模型重导出为 256 维嵌入模型。

背景:dc 的 `wavesfm_rml.onnx` 是 17 类分类器,图只暴露 `logits`(batch,17)。
但它的 ViT 隐藏层有 256 维 CLS 特征(Gather 节点 `select`,shape [batch,256]),
相似度检索需要这个 256 维特征。本脚本把它加为图输出,另存成 `wavesfm_embed.onnx`。

运行:
    python scripts/export_iq_embed_onnx.py
依赖: onnx(可选 onnxruntime 做验证)
"""

from __future__ import annotations

import os
from pathlib import Path

import onnx
from onnx import helper, TensorProto

SRC = Path(r"C:\Users\A\Desktop\dc\robot_model\wavesfm_rml.onnx")
DST_DIR = Path(__file__).resolve().parent.parent / "model" / "iq"
DST = DST_DIR / "wavesfm_embed.onnx"
FEATURE_NAME = "select"  # 256 维 CLS 特征(Gather 节点输出,head Gemm 之前)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"源模型不存在: {SRC}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"加载 {SRC}")
    model = onnx.load(str(SRC), load_external_data=True)

    # 找 256 维特征 tensor 的形状信息
    value_info = None
    for vi in model.graph.value_info:
        if vi.name == FEATURE_NAME:
            value_info = vi
            break

    dims = None
    if value_info is not None:
        dims = [d.dim_value if d.HasField("dim_value") else 0
                for d in value_info.type.tensor_type.shape.dim]
        print(f"找到 value_info: {FEATURE_NAME} 形状 {dims}")
    else:
        # 兜底:手工构造 (batch, 256) 的 value info
        dims = [0, 256]
        print(f"value_info 未直接命中,按 (batch,256) 手工构造 {FEATURE_NAME}")

    if not dims or dims[-1] != 256:
        raise RuntimeError(f"特征维度异常: {dims}(应为 ...,256)")

    # 避免重复加输出
    existing = {o.name for o in model.graph.output}
    if FEATURE_NAME not in existing:
        # dims[0]==0 表示动态 batch,导出为带名字的维度
        shape = [d if d != 0 else "batch_size" for d in dims]
        out = helper.make_tensor_value_info(FEATURE_NAME, TensorProto.FLOAT, shape)
        model.graph.output.append(out)
        print(f"新增图输出: {FEATURE_NAME}")
    else:
        print(f"图输出已含 {FEATURE_NAME},跳过")

    # 权重内联成单个自包含 .onnx(源已 load_external_data 进内存),避免外置数据 sidecar 的解析问题
    onnx.save(model, str(DST))
    print(f"已保存 {DST}(自包含,含全部权重)")

    _verify(str(DST))


def _verify(path: str) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("未装 onnxruntime/numpy,跳过验证")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    print("新模型输出:", names)
    x = np.random.randn(1, 2, 1, 128).astype(np.float32)
    r = sess.run(None, {sess.get_inputs()[0].name: x})
    for n, arr in zip(names, r):
        print(f"  {n}: {arr.shape}")


if __name__ == "__main__":
    main()
