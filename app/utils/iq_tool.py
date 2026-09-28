"""信号特征描述工具"""

from typing import Dict, Union


def describe_signal_features(flat: Dict[str, Union[float, int, str]]) -> str:
    """
    输入信号特征字典，返回中文描述字符串。

    Args:
        flat: 包含 26 个信号特征字段的字典

    Returns:
        结构化的中文描述字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("           雷达信号特征描述报告")
    lines.append("=" * 60)

    # 【幅度特征】
    amplitude_mean = flat.get("amplitude_mean", 0.0)
    amplitude_var = flat.get("amplitude_var", 0.0)
    lines.append("")
    lines.append("【幅度特征】")
    lines.append(f"  - 幅度均值: {amplitude_mean:.6f}")
    lines.append(f"  - 幅度方差: {amplitude_var:.6f}")

    # 【相位特征】
    phase_kurtosis = flat.get("phase_kurtosis", 0.0)
    phase_skewness = flat.get("phase_skewness", 0.0)
    lines.append("")
    lines.append("【相位特征】")
    lines.append(f"  - 相位偏度: {phase_skewness:.6f}")
    lines.append(f"  - 相位峰度: {phase_kurtosis:.6f}")

    # 【频率特征】
    frequency_mean = flat.get("frequency_mean", 0.0)
    frequency_var = flat.get("frequency_var", 0.0)
    frequency_median = flat.get("frequency_median", 0.0)
    frequency_std = flat.get("frequency_std", 0.0)
    frequency_min = flat.get("frequency_min", 0.0)
    frequency_max = flat.get("frequency_max", 0.0)
    lines.append("")
    lines.append("【频率特征】")
    lines.append(f"  - 频率均值:   {frequency_mean:.6f}")
    lines.append(f"  - 频率方差:   {frequency_var:.6f}")
    lines.append(f"  - 频率中位数: {frequency_median:.6f}")
    lines.append(f"  - 频率标准差: {frequency_std:.6f}")
    lines.append(f"  - 频率范围:   [{frequency_min:.6f}, {frequency_max:.6f}]")

    # 【频谱特征】
    psd_peak_frequency = flat.get("psd_peak_frequency", 0.0)
    psd_peak_value = flat.get("psd_peak_value", 0.0)
    spectral_centroid = flat.get("spectral_centroid", 0.0)
    spectral_bandwidth = flat.get("spectral_bandwidth", 0.0)
    spectral_symmetry = flat.get("spectral_symmetry", 0.0)
    lines.append("")
    lines.append("【频谱特征】")
    lines.append(f"  - PSD峰值频率:    {psd_peak_frequency:.6f}")
    lines.append(f"  - PSD峰值:        {psd_peak_value:.6f}")
    lines.append(f"  - 光谱质心:       {spectral_centroid:.6f}")
    lines.append(f"  - 光谱带宽:       {spectral_bandwidth:.6f}")
    lines.append(f"  - 光谱对称性:     {spectral_symmetry:.6f}")

    # 【频带能量特征】
    max_alpha_energy = flat.get("max_alpha_energy", 0.0)
    dominant_alpha = flat.get("dominant_alpha", 0.0)
    symbol_rate_section_peak = flat.get("symbol_rate_section_peak", 0.0)
    lines.append("")
    lines.append("【频带能量特征】")
    lines.append(f"  - 最大α能量:       {max_alpha_energy:.6f}")
    lines.append(f"  - 主导α参数:       {dominant_alpha:.6f}")
    lines.append(f"  - 符号速率段峰值:   {symbol_rate_section_peak:.6f}")

    # 【高阶累积量特征】
    cumulant_c20 = flat.get("cumulant_c20", 0.0)
    cumulant_c21 = flat.get("cumulant_c21", 0.0)
    cumulant_c40 = flat.get("cumulant_c40", 0.0)
    cumulant_c42 = flat.get("cumulant_c42", 0.0)
    cumulant_c60 = flat.get("cumulant_c60", 0.0)
    cumulant_c63 = flat.get("cumulant_c63", 0.0)
    lines.append("")
    lines.append("【高阶累积量特征】")
    lines.append(f"  - 二阶累积量 C20: {cumulant_c20:.6f}")
    lines.append(f"  - 二阶累积量 C21: {cumulant_c21:.6f}")
    lines.append(f"  - 四阶累积量 C40: {cumulant_c40:.6f}")
    lines.append(f"  - 四阶累积量 C42: {cumulant_c42:.6f}")
    lines.append(f"  - 六阶累积量 C60: {cumulant_c60:.6f}")
    lines.append(f"  - 六阶累积量 C63: {cumulant_c63:.6f}")

    # 【调制识别】
    modulation = flat.get("modulation", "")
    mod_id = flat.get("mod_id", -1)
    lines.append("")
    lines.append("【调制识别】")
    lines.append(f"  - 调制类型: {modulation if modulation else '未知'}")
    lines.append(f"  - 调制ID:   {mod_id}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def describe_signal_features_brief(flat: Dict[str, Union[float, int, str]]) -> str:
    """
    输入信号特征字典，返回简洁的一行中文描述。

    Args:
        flat: 包含 26 个信号特征字段的字典

    Returns:
        简洁的中文描述字符串
    """
    modulation = flat.get("modulation", "")
    frequency_mean = flat.get("frequency_mean", 0.0)
    amplitude_mean = flat.get("amplitude_mean", 0.0)
    psd_peak_frequency = flat.get("psd_peak_frequency", 0.0)
    spectral_bandwidth = flat.get("spectral_bandwidth", 0.0)

    mod_str = modulation if modulation else "未知调制"

    return (
        f"该雷达信号为 {mod_str}，幅度均值 {amplitude_mean:.4f}，"
        f"中心频率 {frequency_mean:.4f}，PSD峰值频率 {psd_peak_frequency:.4f}，"
        f"光谱带宽 {spectral_bandwidth:.4f}"
    )