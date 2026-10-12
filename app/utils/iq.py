"""IQ 信号处理编排:读 H5 -> SNR 清洗 -> IQ 补全 -> 信号处理 -> 特征/图表/分类/参数 -> 26 特征中文描述。

整个管道是同步 CPU 密集逻辑,由 API 层丢 run_in_threadpool 执行,不阻塞事件循环。
"""

import json
import math
import os
import tempfile
from contextlib import contextmanager

import h5py
import numpy as np

from ..config import settings
from .iq_processing import (
    remove_dc, fir_filter, resample, remove_dc_post_resample, normalize,
    compute_time_domain, compute_spectrum, compute_spectrogram,
    compute_constellation, compute_eyediagram, compute_envelope,
    compute_phase, compute_instantaneous_frequency, compute_psd,
    compute_cepstrum, compute_stft, compute_wigner_ville,
    clean_by_snr, extract_all_features, estimate_pulse_params,
)
from .iq_buquan import buquan_iq_both
from .iq_predict import predict_weighted
from .iq_param_guess import analyze_iq
from .iq_tool import describe_signal_features


_name_to_id = None


@contextmanager
def _open_h5(content: bytes):
    """字节 -> 临时 .h5 文件 -> 打开的 h5py 句柄(离开上下文自动关闭并清理临时文件)。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        f = h5py.File(tmp.name, "r")
        try:
            yield f
        finally:
            f.close()
    finally:
        os.unlink(tmp.name)


def validate_iq_h5(content: bytes) -> None:
    """输入 H5 完整性校验:不过抛清晰 ValueError,不让损坏/不完整文件进解析管道。

    挡住空文件 / 非 H5 / 缺 IQ 数据(signal/iq 或 iq)/ 形状不对 / 缺采样率 /
    采样率不大于目标采样率(后者正是 fir_filter 在 cutoff>=fs/2 崩掉的那条)。
    """
    if not content:
        raise ValueError("IQ 文件为空,无法解析")
    try:
        with _open_h5(content) as f:
            if "signal" in f and "iq" in f["signal"]:
                ds = f["signal/iq"]
                if ds.ndim != 2 or ds.shape[0] != 2 or ds.shape[1] < settings.iq_segment_len:
                    raise ValueError(
                        f"IQ 双通道数据不完整:形状 {ds.shape},需 (2, >= {settings.iq_segment_len})"
                    )
            elif "iq" in f:
                ds = f["iq"]
                if ds.ndim != 1 or ds.shape[0] < settings.iq_segment_len:
                    raise ValueError(
                        f"IQ 复数数据不完整:形状 {ds.shape},需 (>= {settings.iq_segment_len})"
                    )
            else:
                raise ValueError("缺少 IQ 数据(signal/iq 或 iq),不是有效的 IQ H5 文件")
            if "sample_rate_hz" not in f.attrs:
                raise ValueError("缺少采样率属性 sample_rate_hz")
            fs = float(f.attrs["sample_rate_hz"])
            if not math.isfinite(fs) or fs <= 0:
                raise ValueError(f"采样率无效: {fs}")
            if fs <= settings.iq_target_fs:
                raise ValueError(f"采样率 {fs} Hz 须大于目标采样率 {settings.iq_target_fs} Hz")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("不是有效的 H5 文件") from e


def _read_h5(content: bytes):
    """读 H5 的 IQ 数据、采样率、符号率、脉冲参数。

    兼容两种格式:
      老格式: f["signal/iq"]  shape(2,N) I/Q 两行
      新格式: f["iq"]         shape(N,)  complex64
    脉冲参数:有 f["pulses"] 组则直接读 pri_s / pulse_width_s;
             没有则返回空,由 estimate_pulse_params 从 IQ 估计。
    """
    with _open_h5(content) as f:
        # ---- IQ 数据 ----
        if "signal" in f and "iq" in f["signal"]:
            iq_arr = np.asarray(f["signal/iq"], dtype=np.float64)
            I = iq_arr[0, :].copy()
            Q = iq_arr[1, :].copy()
        elif "iq" in f:
            iq_c = np.asarray(f["iq"])
            if np.iscomplexobj(iq_c):
                I = np.asarray(iq_c.real, dtype=np.float64).copy()
                Q = np.asarray(iq_c.imag, dtype=np.float64).copy()
            else:
                I = np.asarray(iq_c, dtype=np.float64).copy()
                Q = np.zeros_like(I)
        else:
            raise ValueError("H5 中未找到 IQ 数据(signal/iq 或 iq)")

        fs = float(f.attrs["sample_rate_hz"])
        symbol_rate = float(f.attrs.get("symbol_rate_hz", 1e6))

        # ---- 脉冲参数(直接读,不估计) ----
        pulse_params: dict = {}
        if "pulses" in f and isinstance(f["pulses"], h5py.Group):
            pg = f["pulses"]
            if "pri_s" in pg and "pulse_width_s" in pg:
                pri_arr = np.asarray(pg["pri_s"][:], dtype=np.float64)
                pw_arr = np.asarray(pg["pulse_width_s"][:], dtype=np.float64)
                pri_ok = pri_arr[pri_arr > 0]
                pw_ok = pw_arr[pw_arr > 0]
                pulse_params = {
                    "pri_hz": round(float(1.0 / np.median(pri_ok)), 2) if len(pri_ok) else None,
                    "pulse_width_us": round(float(np.median(pw_ok)) * 1e6, 2) if len(pw_ok) else None,
                    "pulse_count": int(len(pri_arr)),
                    "source": "h5_pulses",
                }
    return I, Q, fs, symbol_rate, pulse_params


def _load_name_to_id():
    """label_map(id->name)反向映射 name->id(全局缓存)。"""
    global _name_to_id
    if _name_to_id is None:
        with open(settings.iq_label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        _name_to_id = {v: int(k) for k, v in label_map.items()}
    return _name_to_id


def _slice_iq(I, Q, segment_len):
    """把 I/Q 切成固定长度片段,返回 [{I: [...], Q: [...]}, ...],末尾不足整段的丢弃。"""
    n = min(len(I), len(Q))
    segments = []
    for start in range(0, n - segment_len + 1, segment_len):
        segments.append({
            "I": [float(x) for x in I[start:start + segment_len]],
            "Q": [float(x) for x in Q[start:start + segment_len]],
        })
    return segments


def _flatten_features(features, modulation, mod_id):
    """extract_all_features 嵌套 4 类 -> 平铺 26 字段(喂 describe_signal_features)。"""
    td = features.get("time_domain", {})
    fd = features.get("frequency_domain", {})
    cs = features.get("cyclic_stationary", {})
    ho = features.get("higher_order_cumulant", {})

    return {
        "amplitude_mean": td.get("amplitude_mean", 0.0),
        "amplitude_var": td.get("amplitude_var", 0.0),
        "phase_kurtosis": td.get("phase_kurtosis", 0.0),
        "phase_skewness": td.get("phase_skewness", 0.0),
        "frequency_mean": td.get("frequency_mean", 0.0),
        "frequency_var": td.get("frequency_var", 0.0),
        "frequency_median": td.get("frequency_median", 0.0),
        "frequency_std": td.get("frequency_std", 0.0),
        "frequency_min": td.get("frequency_min", 0.0),
        "frequency_max": td.get("frequency_max", 0.0),
        "psd_peak_frequency": fd.get("psd_peak_frequency", 0.0),
        "psd_peak_value": fd.get("psd_peak_value", 0.0),
        "spectral_centroid": fd.get("spectral_centroid", 0.0),
        "spectral_bandwidth": fd.get("spectral_bandwidth", 0.0),
        "spectral_symmetry": fd.get("spectral_symmetry", 0.0),
        "max_alpha_energy": cs.get("max_alpha_energy", 0.0),
        "dominant_alpha": cs.get("dominant_alpha", 0.0),
        "symbol_rate_section_peak": cs.get("symbol_rate_section_peak", 0.0),
        "cumulant_c20": ho.get("C20", 0.0),
        "cumulant_c21": ho.get("C21", 0.0),
        "cumulant_c40": ho.get("C40", 0.0),
        "cumulant_c42": ho.get("C42", 0.0),
        "cumulant_c60": ho.get("C60", 0.0),
        "cumulant_c63": ho.get("C63", 0.0),
        "modulation": modulation,
        "mod_id": mod_id,
    }


def _clean_param_result(param_result):
    """清理 analyze_iq 返回值:数值四舍五入到 6 位,None 保留,inf/nan 转 None。"""
    clean = {}
    for k, v in param_result.items():
        if v is None:
            clean[k] = None
        elif isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                clean[k] = None
            else:
                clean[k] = round(float(v), 6)
        else:
            clean[k] = v
    return clean


def process_iq_h5(content: bytes, snr_threshold_db: float | None = None) -> dict:
    """处理 H5 IQ 信号,返回全部中间结果(供 API 层组装响应与 md 入库)。

    snr_threshold_db: SNR 清洗阈值(dB);None 回退全局默认 settings.iq_snr_threshold_db。
    """
    validate_iq_h5(content)
    I, Q, fs, symbol_rate, h5_pulse_params = _read_h5(content)
    segment_len = settings.iq_segment_len
    if snr_threshold_db is None:
        snr_threshold_db = settings.iq_snr_threshold_db
    target_fs = settings.iq_target_fs

    if min(len(I), len(Q)) < segment_len:
        raise ValueError(f"IQ 数据长度 {min(len(I), len(Q))} 小于片段长度 {segment_len}")

    # 1. SNR 清洗(记录处理前 I/Q 与处理后 I_work/Q_work)
    I_clean, Q_clean, cleaning_info = clean_by_snr(I, Q, segment_len, snr_threshold_db)
    if cleaning_info.get("removed_segments", 0) > 0:
        I_work, Q_work = I_clean, Q_clean
    else:
        I_work, Q_work = I, Q

    snr_data = {
        "total_segments": cleaning_info.get("total_segments", 0),
        "valid_segments": cleaning_info.get("valid_segments", 0),
        "removed_segments": cleaning_info.get("removed_segments", 0),
        "mean_snr": cleaning_info.get("mean_snr", 0.0),
        "min_snr": cleaning_info.get("min_snr", 0.0),
        "snr_threshold_db": cleaning_info.get("snr_threshold_db", snr_threshold_db),
    }

    # 2. IQ 补全(对 SNR 清洗后 IQ;补全前 = masked,补全后 = recovered)
    iq_complex = I_work + 1j * Q_work
    iq_recovered, iq_masked = buquan_iq_both(iq_complex)

    # 3. 信号处理(对 SNR 清洗后 IQ:去直流 -> FIR -> 重采样 -> 去直流 -> 归一化,与 see 一致)
    I_dc, Q_dc = remove_dc(I_work, Q_work)
    original_power = float(np.mean(I_dc ** 2) + np.mean(Q_dc ** 2))

    I_filt, Q_filt = fir_filter(I_dc, Q_dc, fs, target_fs)
    filtered_power = float(np.mean(I_filt ** 2) + np.mean(Q_filt ** 2))
    attenuation_db = round(10 * math.log10(filtered_power / original_power), 4) if original_power > 0 else 0.0

    I_res, Q_res = resample(I_filt, Q_filt, fs, target_fs)
    I_final, Q_final = remove_dc_post_resample(I_res, Q_res)
    I_norm, Q_norm, _ = normalize(I_final, Q_final)

    if len(I_norm) < 128:
        raise ValueError(f"归一化后 IQ 长度 {len(I_norm)} 不足 128,无法分类")

    # 4. 11 图表
    charts = {
        "time_domain": compute_time_domain(I_norm, Q_norm, target_fs),
        "envelope": compute_envelope(I_norm, Q_norm, target_fs),
        "phase": compute_phase(I_norm, Q_norm, target_fs),
        "instantaneous_frequency": compute_instantaneous_frequency(I_norm, Q_norm, target_fs),
        "spectrum": compute_spectrum(I_norm, Q_norm, target_fs),
        "psd": compute_psd(I_norm, Q_norm, target_fs),
        "cepstrum": compute_cepstrum(I_norm, Q_norm, target_fs),
        "spectrogram": compute_spectrogram(I_norm, Q_norm, target_fs),
        "stft": compute_stft(I_norm, Q_norm, target_fs),
        "wigner_ville": compute_wigner_ville(I_norm, Q_norm, target_fs),
        "constellation": compute_constellation(I_norm, Q_norm),
        "eyediagram": compute_eyediagram(I_norm, Q_norm, target_fs),
    }

    # 5. 特征(嵌套 4 类,照搬 see 返回结构)
    features = extract_all_features(I_norm, Q_norm, target_fs, symbol_rate)

    # 6. 调制分类(前 128 点)
    iq_sample = np.column_stack([I_norm[:128], Q_norm[:128]])
    prediction = predict_weighted(iq_sample)

    # 7. 参数猜测(mod_id 取加权分类结果)
    predicted_class = prediction["weighted"]["class"]
    mod_id = _load_name_to_id().get(predicted_class, 14)
    iq_norm_complex = I_norm + 1j * Q_norm
    param_guess = {
        "modulation": predicted_class,
        "mod_id": mod_id,
        "parameters": _clean_param_result(analyze_iq(iq_norm_complex, mod_id, target_fs)),
    }

    # 8. 平铺 26 特征 -> 中文描述(作为 md 正文)
    flat = _flatten_features(features, predicted_class, mod_id)
    # 8.5 脉冲参数:H5 有 pulses 组直接读,否则从 IQ 估计
    if h5_pulse_params and h5_pulse_params.get("pri_hz") is not None:
        pulse_params = h5_pulse_params
    else:
        pulse_params = estimate_pulse_params(
            I_norm, Q_norm, target_fs,
            threshold_method=settings.iq_pulse_threshold_method,
            threshold_k=settings.iq_pulse_threshold_k,
            threshold_pct=settings.iq_pulse_threshold_pct,
            min_pulse_width_us=settings.iq_pulse_min_width_us,
            min_pri_us=settings.iq_pulse_min_pri_us,
            min_pulses=settings.iq_pulse_min_count,
            max_duty=settings.iq_pulse_max_duty,
            max_pw_cv=settings.iq_pulse_max_pw_cv,
        )
    flat["pri_hz"] = pulse_params.get("pri_hz")
    flat["pulse_width_us"] = pulse_params.get("pulse_width_us")
    flat["pulse_count"] = pulse_params.get("pulse_count", 0)
    md = describe_signal_features(flat)

    # 9. 5 组 IQ 片段(新增 iq_norm:归一化后片段,供 IQ 嵌入用)
    iq_raw = _slice_iq(I, Q, segment_len)
    iq_clean = _slice_iq(I_work, Q_work, segment_len)
    iq_masked_segments = _slice_iq(iq_masked.real, iq_masked.imag, segment_len)
    iq_recovered_segments = _slice_iq(iq_recovered.real, iq_recovered.imag, segment_len)
    iq_norm = _slice_iq(I_norm, Q_norm, segment_len)

    raw_entry = {
        "engine": "iq",
        "sample_rate_hz": fs,
        "symbol_rate_hz": symbol_rate,
        "segment_len": segment_len,
        "snr_threshold_db": snr_threshold_db,
        "target_fs": target_fs,
        "modulation": predicted_class,
        "mod_id": mod_id,
        "attenuation_db": attenuation_db,
        "snr": snr_data,
        "pulse": pulse_params,
    }

    return {
        "snr": snr_data,
        "attenuation_db": attenuation_db,
        "charts": charts,
        "features": features,
        "prediction": prediction,
        "param_guess": param_guess,
        "iq_raw": iq_raw,
        "iq_clean": iq_clean,
        "iq_masked": iq_masked_segments,
        "iq_recovered": iq_recovered_segments,
        "iq_norm": iq_norm,
        "flat": flat,
        "iq_meta": {
            "segment_len": segment_len,
            "iq_raw_segments": len(iq_raw),
            "iq_clean_segments": len(iq_clean),
            "iq_masked_segments": len(iq_masked_segments),
            "iq_recovered_segments": len(iq_recovered_segments),
            "iq_norm_segments": len(iq_norm),
        },
        "md": md,
        "raw_entry": raw_entry,
    }
