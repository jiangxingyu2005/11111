import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Tuple, Dict


def downsample_to_limit(I: np.ndarray, Q: np.ndarray, limit: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """如果数据点超过limit，均匀降采样到limit个点"""
    if len(I) <= limit:
        return I, Q
    indices = np.linspace(0, len(I) - 1, limit, dtype=int)
    return I[indices], Q[indices]


def compute_time_domain(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """时域波形数据"""
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    time = np.arange(len(I_d)) / fs
    return {
        "time": time.tolist(),
        "I": I_d.tolist(),
        "Q": Q_d.tolist(),
    }


def compute_spectrum(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """频谱图数据（单边幅度谱）"""
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    N = len(complex_sig)
    fft_vals = np.fft.fft(complex_sig)
    fft_mag = np.abs(fft_vals) / N
    freqs = np.fft.fftfreq(N, 1 / fs)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_mag = fft_mag[pos_mask] * 2
    return {
        "freqs": freqs.tolist(),
        "magnitude": fft_mag.tolist(),
    }


def compute_spectrogram(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """时频图数据（spectrogram）"""
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    nperseg = min(256, len(complex_sig))
    f, t, Sxx = signal.spectrogram(complex_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    S_db = 10 * np.log10(Sxx + 1e-12)
    return {
        "freqs": f.tolist(),
        "times": t.tolist(),
        "power": S_db.tolist(),
    }


def compute_constellation(I: np.ndarray, Q: np.ndarray, limit: int = 2048) -> Dict:
    """星座图数据"""
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    return {
        "I": I_d.tolist(),
        "Q": Q_d.tolist(),
    }


def compute_eyediagram(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """眼图数据（基于幅度包络，按符号周期折叠）"""
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    envelope = np.sqrt(I_d ** 2 + Q_d ** 2)
    samples_per_symbol = 4
    num_symbols = len(envelope) // samples_per_symbol
    if num_symbols < 4:
        num_symbols = max(4, num_symbols)
        samples_per_symbol = len(envelope) // num_symbols
    trace_len = num_symbols * samples_per_symbol
    trace = envelope[:trace_len].reshape(num_symbols, samples_per_symbol)
    x = np.arange(samples_per_symbol)
    return {
        "x": x.tolist(),
        "traces": trace.tolist(),
        "samples_per_symbol": samples_per_symbol,
    }


def remove_dc(I: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """去直流：减去I/Q信号的均值"""
    I = I - np.mean(I)
    Q = Q - np.mean(Q)
    return I, Q


def fir_filter(
    I: np.ndarray,
    Q: np.ndarray,
    fs: float,
    target_fs: float = 10e6,
    stopband_atten: float = 80,
    transition_width: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """FIR抗混叠滤波

    参数:
        I: I路信号
        Q: Q路信号
        fs: 原始采样率 (Hz)
        target_fs: 目标采样率 (Hz)，默认 10MHz
        stopband_atten: 阻带衰减 (dB)，默认 80
        transition_width: 过渡带宽度（归一化），默认 0.02
    """
    cutoff_freq = target_fs / 2
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist

    if stopband_atten > 50:
        beta = 0.1102 * (stopband_atten - 8.7)
    elif stopband_atten >= 21:
        beta = 0.5842 * (stopband_atten - 21) ** 0.4 + 0.07886 * (stopband_atten - 21)
    else:
        beta = 0.0

    N = int(np.ceil((stopband_atten - 8) / (2.285 * transition_width * 2 * np.pi)))
    N = N if N % 2 == 0 else N + 1

    h = signal.firwin(N + 1, normalized_cutoff, window=("kaiser", beta), pass_zero=True)
    I_filtered = signal.filtfilt(h, 1.0, I)
    Q_filtered = signal.filtfilt(h, 1.0, Q)

    return I_filtered, Q_filtered


def resample(
    I: np.ndarray,
    Q: np.ndarray,
    fs: float,
    target_fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """重采样：根据原始采样率和目标采样率对I/Q信号进行重采样

    参数:
        I: I路信号
        Q: Q路信号
        fs: 原始采样率 (Hz)
        target_fs: 目标采样率 (Hz)
    """
    num_original = len(I)
    num_resampled = int(num_original * target_fs / fs)

    I_resampled = signal.resample(I, num_resampled)
    Q_resampled = signal.resample(Q, num_resampled)

    return I_resampled, Q_resampled


def remove_dc_post_resample(I: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """重采样后的第二次去直流：减去I/Q信号的均值"""
    return remove_dc(I, Q)


def normalize(I: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """归一化到 [-1, 1]，基于I/Q的绝对值全局最大值

    参数:
        I: I路信号
        Q: Q路信号
    返回:
        I_normalized: 归一化后的I路信号
        Q_normalized: 归一化后的Q路信号
        global_max: 归一化因子（I/Q绝对值的全局最大值）
    """
    global_max = max(np.max(np.abs(I)), np.max(np.abs(Q)))
    I_normalized = I / global_max
    Q_normalized = Q / global_max
    return I_normalized, Q_normalized, global_max


# ==================== 信号特征提取 ====================

def extract_time_domain_features(I: np.ndarray, Q: np.ndarray, fs: float) -> Dict:
    """提取时域特征
    
    返回:
        amplitude_mean: 幅度均值
        amplitude_var: 幅度方差
        phase_kurtosis: 相位峰度
        phase_skewness: 相位偏度
        frequency_mean: 瞬时频率均值
        frequency_var: 瞬时频率方差
        frequency_median: 瞬时频率中位数
        frequency_std: 瞬时频率标准差
        frequency_min: 瞬时频率最小值
        frequency_max: 瞬时频率最大值
    """
    complex_sig = I + 1j * Q
    
    # 幅度特征
    amplitude = np.abs(complex_sig)
    amplitude_mean = float(np.mean(amplitude))
    amplitude_var = float(np.var(amplitude))
    
    # 相位特征
    phase = np.angle(complex_sig)
    phase_kurtosis_val = float(kurtosis(phase))
    phase_skewness_val = float(skew(phase))
    
    # 瞬时频率特征
    # 解包裹避免跳变
    d_phase_unwrapped = np.unwrap(phase)
    d_phase_unwrapped = np.diff(d_phase_unwrapped)
    instantaneous_freq = np.abs(d_phase_unwrapped * fs / (2 * np.pi))
    
    frequency_mean = float(np.mean(instantaneous_freq))
    frequency_var = float(np.var(instantaneous_freq))
    frequency_median = float(np.median(instantaneous_freq))
    frequency_std = float(np.std(instantaneous_freq))
    frequency_min = float(np.min(instantaneous_freq))
    frequency_max = float(np.max(instantaneous_freq))
    
    return {
        "amplitude_mean": amplitude_mean,
        "amplitude_var": amplitude_var,
        "phase_kurtosis": phase_kurtosis_val,
        "phase_skewness": phase_skewness_val,
        "frequency_mean": frequency_mean,
        "frequency_var": frequency_var,
        "frequency_median": frequency_median,
        "frequency_std": frequency_std,
        "frequency_min": frequency_min,
        "frequency_max": frequency_max,
    }


def extract_frequency_domain_features(I: np.ndarray, Q: np.ndarray, fs: float) -> Dict:
    """提取频域特征
    
    返回:
        psd_peak_frequency: PSD峰值频率
        psd_peak_value: PSD峰值
        spectral_centroid: 频谱质心
        spectral_bandwidth: 频谱带宽
        spectral_symmetry: 频谱对称性
    """
    complex_sig = I + 1j * Q
    
    # 计算PSD
    frequencies, psd = signal.welch(np.abs(complex_sig), fs=fs, nperseg=min(256, len(complex_sig)))
    
    # PSD峰值
    psd_peak_idx = np.argmax(psd)
    psd_peak_frequency = float(frequencies[psd_peak_idx])
    psd_peak_value = float(psd[psd_peak_idx])
    
    # 频谱质心
    spectral_centroid = float(np.sum(frequencies * psd) / np.sum(psd))
    
    # 频谱带宽
    spectral_bandwidth = float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd)))
    
    # 频谱对称性（以质心为界，比较左右两边的能量）
    mid_idx = np.argmin(np.abs(frequencies - spectral_centroid))
    left_energy = float(np.sum(psd[:mid_idx]))
    right_energy = float(np.sum(psd[mid_idx:]))
    total_energy = left_energy + right_energy
    spectral_symmetry = abs(left_energy - right_energy) / total_energy if total_energy > 0 else 0.0
    
    return {
        "psd_peak_frequency": psd_peak_frequency,
        "psd_peak_value": psd_peak_value,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_symmetry": spectral_symmetry,
    }


def extract_cyclic_stationary_features(I: np.ndarray, Q: np.ndarray, fs: float, symbol_rate: float) -> Dict:
    """提取循环平稳特征
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        symbol_rate: 符号率
    返回:
        cyclic_alpha_energy: 循环谱特征（alphas 和 energies）
        max_alpha_energy: 最大循环频率能量
        dominant_alpha: 主导循环频率
        symbol_rate_section_peak: 符号率区间的峰值能量
    """
    complex_sig = I + 1j * Q
    N = len(complex_sig)
    
    # 归一化信号
    sig_power = np.mean(np.abs(complex_sig) ** 2)
    if sig_power > 0:
        complex_sig_norm = complex_sig / np.sqrt(sig_power)
    else:
        complex_sig_norm = complex_sig
    
    # 计算循环谱特征（简化版谱相关密度）
    # 扫描循环频率 alpha 范围 [-symbol_rate, symbol_rate]
    max_alpha = symbol_rate
    num_alphas = 51
    alphas = np.linspace(-max_alpha, max_alpha, num_alphas)
    
    energies = []
    carry_length = max(16, int(fs / symbol_rate) * 4)
    carry_length = min(carry_length, N // 2)
    
    for alpha in alphas:
        f_offset = alpha / 2
        x1 = complex_sig_norm * np.exp(-1j * 2 * np.pi * f_offset * np.arange(N) / fs)
        x2 = complex_sig_norm * np.exp(1j * 2 * np.pi * f_offset * np.arange(N) / fs)
        
        Rxx1 = np.fft.fft(x1, n=carry_length)
        Rxx2 = np.fft.fft(x2, n=carry_length)
        
        # 取零频分量附近的谱相关
        Salpha = np.abs(Rxx1 * Rxx2.conj()) / N
        energy = np.max(Salpha)
        energies.append(float(energy))
    
    energies_arr = np.array(energies)
    max_idx = np.argmax(energies_arr)
    
    # 符号率区间的峰值能量
    symbol_rate_mask = (np.abs(alphas) >= symbol_rate * 0.8) & (np.abs(alphas) <= symbol_rate * 1.2)
    if np.any(symbol_rate_mask):
        symbol_rate_section_peak = float(np.max(energies_arr[symbol_rate_mask]))
    else:
        symbol_rate_section_peak = 0.0
    
    return {
        "cyclic_alpha_energy": {
            "alphas": alphas.tolist(),
            "energies": energies_arr.tolist(),
        },
        "max_alpha_energy": float(energies_arr[max_idx]),
        "dominant_alpha": float(alphas[max_idx]),
        "symbol_rate_section_peak": symbol_rate_section_peak,
    }


def extract_higher_order_cumulant_features(I: np.ndarray, Q: np.ndarray, fs: float, symbol_rate: float) -> Dict:
    """提取高阶累积量特征
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        symbol_rate: 符号率
    返回:
        C20: 二阶累积量 C20
        C21: 二阶累积量 C21
        C40: 四阶累积量 C40
        C42: 四阶累积量 C42
        C60: 六阶累积量 C60
        C63: 六阶累积量 C63
    """
    complex_sig = I + 1j * Q

    # 去直流
    complex_sig = complex_sig - np.mean(complex_sig)
    
    # 计算各阶矩
    E_x2 = np.mean(complex_sig ** 2)
    E_x_sq_mag = np.mean(np.abs(complex_sig) ** 2)
    E_x4 = np.mean(complex_sig ** 4)
    E_x_sq_mag2 = np.mean(np.abs(complex_sig) ** 4)
    E_x6 = np.mean(complex_sig ** 6)
    E_x3_mag3 = np.mean(np.abs(complex_sig) ** 6)
    E_x3_sq_mag = np.mean(complex_sig ** 3 * np.abs(complex_sig) ** 2)
    
    # 二阶累积量
    C20 = E_x2  # E[x^2]
    C21 = E_x_sq_mag  # E[|x|^2]
    
    # 四阶累积量
    C40 = E_x4 - E_x2 ** 2  # E[x^4] - E[x^2]^2
    C42 = E_x_sq_mag2 - 2 * E_x_sq_mag ** 2 + 2 * E_x_sq_mag ** 2 - E_x_sq_mag ** 2
    # 简化: C42 = E[|x|^4] - E[|x|^2]^2
    C42 = E_x_sq_mag2 - E_x_sq_mag ** 2
    
    # 六阶累积量
    C60 = E_x6 - 6 * E_x4 * E_x2 + 3 * E_x2 ** 3  # E[x^6] - 6E[x^4]E[x^2] + 3E[x^2]^3
    C63 = E_x3_mag3 - 3 * E_x3_sq_mag * E_x_sq_mag + 2 * E_x3_sq_mag * E_x_sq_mag
    # 简化: C63 = E[|x|^6] - 3E[x^3|x|^2]E[|x|^2] + 2E[|x|^2]^3
    C63 = E_x3_mag3 - 3 * E_x3_sq_mag * E_x_sq_mag + 2 * E_x_sq_mag ** 3
    
    return {
        "C20": float(np.abs(C20)),
        "C21": float(np.abs(C21)),
        "C40": float(np.abs(C40)),
        "C42": float(np.abs(C42)),
        "C60": float(np.abs(C60)),
        "C63": float(np.abs(C63)),
    }


def extract_all_features(I: np.ndarray, Q: np.ndarray, fs: float, symbol_rate: float) -> Dict:
    """提取所有信号特征，返回统一JSON格式
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        symbol_rate: 符号率
    返回:
        {
            "time_domain": { ... },
            "frequency_domain": { ... },
            "cyclic_stationary": { ... },
            "higher_order_cumulant": { ... }
        }
    """
    return {
        "time_domain": extract_time_domain_features(I, Q, fs),
        "frequency_domain": extract_frequency_domain_features(I, Q, fs),
        "cyclic_stationary": extract_cyclic_stationary_features(I, Q, fs, symbol_rate),
        "higher_order_cumulant": extract_higher_order_cumulant_features(I, Q, fs, symbol_rate),
    }


# ==================== 绘图数据生成函数 ====================

def compute_envelope(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """包络图数据
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        time: 时间轴
        envelope: 包络值 sqrt(I^2 + Q^2)
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    time = np.arange(len(I_d)) / fs
    envelope = np.sqrt(I_d ** 2 + Q_d ** 2)
    return {
        "time": time.tolist(),
        "envelope": envelope.tolist(),
    }


def compute_phase(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """相位图数据
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        time: 时间轴
        phase: 相位值（弧度），经解包裹处理
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    time = np.arange(len(I_d)) / fs
    phase = np.angle(I_d + 1j * Q_d)
    phase_unwrapped = np.unwrap(phase)
    return {
        "time": time.tolist(),
        "phase": phase_unwrapped.tolist(),
    }


def compute_instantaneous_frequency(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """瞬时频率图数据
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        time: 时间轴（比原信号少一个点）
        instantaneous_frequency: 瞬时频率（Hz）
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    phase = np.angle(I_d + 1j * Q_d)
    phase_unwrapped = np.unwrap(phase)
    d_phase = np.diff(phase_unwrapped)
    instantaneous_freq = d_phase * fs / (2 * np.pi)
    time = np.arange(len(instantaneous_freq)) / fs
    return {
        "time": time.tolist(),
        "instantaneous_frequency": instantaneous_freq.tolist(),
    }


def compute_psd(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """功率谱密度图数据（PSD）
    
    使用Welch方法计算功率谱密度
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        freqs: 频率轴（Hz）
        psd: 功率谱密度值
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    nperseg = min(256, len(complex_sig))
    freqs, psd = signal.welch(complex_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return {
        "freqs": freqs.tolist(),
        "psd": psd.tolist(),
    }


def compute_cepstrum(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """Cepstrum分析数据
    
    对FFT频谱取对数后再进行IFFT，用于检测信号的周期性结构
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        quefrency: Quefrency轴（秒）
        cepstrum: Cepstrum值
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    N = len(complex_sig)
    fft_vals = np.fft.fft(complex_sig, n=N)
    # 取对数幅度谱
    log_spectrum = np.log(np.abs(fft_vals) + 1e-12)
    # 实cepsstrum：对实数log幅度做IFFT
    real_cepstrum = np.fft.ifft(log_spectrum).real
    # 移位使cepsstrum对称
    real_cepstrum = np.fft.fftshift(real_cepstrum)
    # quefrency轴（时间延迟）
    quefrency = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    return {
        "quefrency": quefrency.tolist(),
        "cepstrum": real_cepstrum.tolist(),
    }


def compute_stft(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """STFT短时傅里叶变换时频图数据
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        freqs: 频率轴（Hz）
        times: 时间轴（秒）
        magnitude: STFT幅度（dB）
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    nperseg = min(256, len(complex_sig))
    f, t, Zxx = signal.stft(complex_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return {
        "freqs": f.tolist(),
        "times": t.tolist(),
        "magnitude": magnitude_db.tolist(),
    }


# ==================== 数据清洗功能 ====================

def compute_snr(I: np.ndarray, Q: np.ndarray, segment_len: int = 128) -> np.ndarray:
    """计算每个片段的 SNR（信噪比）
    
    方法：将信号分段，对每段计算信号功率与噪声功率之比
    - 信号功率：信号的总功率
    - 噪声功率：使用相邻片段中位数估计噪声 floor
    
    参数:
        I: I路信号
        Q: Q路信号
        segment_len: 片段长度（默认128，与embed接口一致）
    返回:
        snr_db: 每个片段的 SNR 值（dB）
    """
    complex_sig = I + 1j * Q
    N = len(complex_sig)
    num_segments = N // segment_len
    
    if num_segments == 0:
        return np.array([0.0])
    
    # 计算每个片段的功率
    segment_powers = np.zeros(num_segments)
    for idx in range(num_segments):
        start = idx * segment_len
        end = start + segment_len
        segment = complex_sig[start:end]
        segment_powers[idx] = np.mean(np.abs(segment) ** 2)
    
    # 使用中位数绝对偏差 (MAD) 估计噪声 floor
    # 噪声通常对应于功率较低的片段，中位数是一个稳健的噪声估计
    median_power = np.median(segment_powers)
    
    # 避免除零
    if median_power < 1e-12:
        return np.zeros(num_segments)
    
    # SNR = 10 * log10(信号功率 / 噪声功率)
    # 这里用中位数作为噪声基准
    snr_db = 10 * np.log10(segment_powers / median_power)
    
    return snr_db


def clean_by_snr(I: np.ndarray, Q: np.ndarray, segment_len: int = 128, 
                 snr_threshold_db: float = -10.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """基于 SNR 进行数据清洗，剔除低质量片段
    
    参数:
        I: I路信号（已归一化）
        Q: Q路信号（已归一化）
        segment_len: 片段长度（默认128）
        snr_threshold_db: SNR 阈值（dB），低于此值的片段被视为无效
    返回:
        I_cleaned: 清洗后的 I 路信号（只保留有效片段）
        Q_cleaned: 清洗后的 Q 路信号（只保留有效片段）
        cleaning_info: 清洗统计信息
            - total_segments: 总片段数
            - valid_segments: 有效片段数
            - removed_segments: 被剔除的片段数
            - snr_values: 所有片段的 SNR 值
            - valid_indices: 有效片段索引
            - mean_snr: 有效片段的平均 SNR
            - min_snr: 有效片段的最小 SNR
    """
    snr_values = compute_snr(I, Q, segment_len)
    num_segments = len(snr_values)
    
    # 找出有效片段（SNR >= 阈值）
    valid_mask = snr_values >= snr_threshold_db
    valid_indices = np.where(valid_mask)[0]
    
    # 如果所有片段都有效，直接返回原信号
    if np.all(valid_mask):
        return I, Q, {
            "total_segments": int(num_segments),
            "valid_segments": int(num_segments),
            "removed_segments": 0,
            "snr_values": snr_values.tolist(),
            "valid_indices": valid_indices.tolist(),
            "mean_snr": float(np.mean(snr_values)),
            "min_snr": float(np.min(snr_values)),
            "snr_threshold_db": snr_threshold_db,
        }
    
    # 拼接有效片段
    segments_to_keep = []
    for idx in valid_indices:
        start = idx * segment_len
        end = start + segment_len
        segments_to_keep.append(np.concatenate([I[start:end], Q[start:end]]))
    
    if not segments_to_keep:
        # 极端情况：所有片段都被剔除，返回原信号并警告
        return I, Q, {
            "total_segments": int(num_segments),
            "valid_segments": 0,
            "removed_segments": int(num_segments),
            "snr_values": snr_values.tolist(),
            "valid_indices": [],
            "mean_snr": 0.0,
            "min_snr": 0.0,
            "snr_threshold_db": snr_threshold_db,
            "warning": "所有片段SNR均低于阈值，返回原始信号",
        }
    
    combined = np.concatenate(segments_to_keep)
    I_cleaned = combined[0::2]
    Q_cleaned = combined[1::2]
    
    valid_snr = snr_values[valid_mask]
    
    return I_cleaned, Q_cleaned, {
        "total_segments": int(num_segments),
        "valid_segments": int(len(valid_indices)),
        "removed_segments": int(num_segments - len(valid_indices)),
        "snr_values": snr_values.tolist(),
        "valid_indices": valid_indices.tolist(),
        "mean_snr": float(np.mean(valid_snr)),
        "min_snr": float(np.min(valid_snr)),
        "snr_threshold_db": snr_threshold_db,
    }


def compute_wigner_ville(I: np.ndarray, Q: np.ndarray, fs: float, limit: int = 2048) -> Dict:
    """Wigner-Ville分布时频图数据
    
    使用Wigner-Ville分布计算时频表示，具有更高的时频分辨率
    
    参数:
        I: I路信号
        Q: Q路信号
        fs: 采样率
        limit: 最大数据点数
    返回:
        freqs: 频率轴（Hz）
        times: 时间轴（秒）
        wvd: Wigner-Ville分布值
    """
    I_d, Q_d = downsample_to_limit(I, Q, limit)
    complex_sig = I_d + 1j * Q_d
    N = len(complex_sig)
    
    # Wigner-Ville分布计算
    # W(t, f) = sum_tau x(t+tau) * xconj(t-tau) * exp(-j2*pi*f*tau)
    # 使用高效实现
    
    # 时滞范围（tau 只用于上面的数学注释，实际走高效 FFT 实现）
    max_tau = N // 2

    # 计算WVD
    T = N  # 时间点数量
    F = N  # 频率点数量

    WVD = np.zeros((F, T), dtype=complex)
    
    for t in range(T):
        # 对于每个时间点，计算其WVD切片
        contribution = np.zeros(max_tau, dtype=complex)
        for tau_val in range(1, max_tau):
            t_plus = t + tau_val
            t_minus = t - tau_val
            # 边界处理
            if 0 <= t_plus < N and 0 <= t_minus < N:
                contribution[tau_val - 1] += complex_sig[t_plus] * np.conj(complex_sig[t_minus])
        
        # 对时滞做FFT得到频率分布
        WVD[:, t] = np.fft.fftshift(np.fft.fft(contribution, n=F))
    
    # 取实部
    WVD_real = WVD.real
    # 归一化
    WVD_max = np.max(np.abs(WVD_real))
    if WVD_max > 0:
        WVD_real = WVD_real / WVD_max
    
    # 频率轴
    freqs = np.fft.fftshift(np.fft.fftfreq(F, 1 / fs))
    # 时间轴
    times = np.arange(T) / fs
    
    return {
        "freqs": freqs.tolist(),
        "times": times.tolist(),
        "wvd": WVD_real.tolist(),
    }


# ==================== 脉冲参数估计 ====================

def estimate_pulse_params(I: np.ndarray, Q: np.ndarray, fs: float,
                          threshold_method: str = "mad",
                          threshold_k: float = 3.0,
                          threshold_pct: float = 80.0,
                          min_pulse_width_us: float = 1.0,
                          min_pri_us: float = 10.0,
                          min_pulses: int = 3,
                          max_duty: float = 0.5,
                          max_pw_cv: float = 0.5) -> dict:
    """从 IQ 包络估计脉冲重复频率(PRF)与脉宽。

    三步:包络过门限 → 检脉冲 → 量脉宽/算 PRI。
    三道防误报:占空比过高(连续信号)、脉冲数不足、脉宽离散度太大。

    参数(用户数据有偏差时调这些,不改代码):
      threshold_method: mad(中位数+k*MAD) | percentile(百分位) | otsu(大津法)
      threshold_k:      mad 法灵敏度,越小越敏感
      threshold_pct:    percentile 法百分位(0-100)
      min_pulse_width_us: 最短脉宽(μs),短于此视为噪声尖峰
      min_pri_us:        最短 PRI(μs),短于此视为重复检测
      min_pulses:        少于此发数不出 PRI
      max_duty:          占空比超此值判非脉冲串
      max_pw_cv:         脉宽变异系数超此值判非整齐脉冲串
    """
    env = np.sqrt(np.asarray(I, dtype=np.float64) ** 2 +
                  np.asarray(Q, dtype=np.float64) ** 2)
    n = len(env)
    if n < 100:
        return {"pri_hz": None, "pulse_width_us": None, "pulse_count": 0,
                "duty_cycle": None, "threshold": None, "method": threshold_method}

    # ---- 门限 ----
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) * 1.4826   # MAD → 标准差近似
    if method_mad(threshold_method):
        thr = med + threshold_k * (mad or float(np.std(env)) or 1.0)
    elif threshold_method == "percentile":
        thr = float(np.percentile(env, threshold_pct))
    elif threshold_method == "otsu":
        thr = _otsu_threshold(env)
    else:
        thr = med + threshold_k * (mad or float(np.std(env)) or 1.0)

    above = env > thr
    duty = float(above.mean())
    if duty > max_duty:
        return {"pri_hz": None, "pulse_width_us": None, "pulse_count": 0,
                "duty_cycle": round(duty, 4), "threshold": round(thr, 6),
                "method": threshold_method}

    # ---- 找上升沿 / 下降沿 ----
    d = np.diff(above.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if len(starts) == 0 or len(ends) == 0:
        return {"pri_hz": None, "pulse_width_us": None, "pulse_count": 0,
                "duty_cycle": round(duty, 4), "threshold": round(thr, 6),
                "method": threshold_method}
    if ends[0] < starts[0]:          # 首个是截断脉冲,丢掉
        ends = ends[1:]
    npair = min(len(starts), len(ends))
    starts, ends = starts[:npair], ends[:npair]

    # ---- 脉宽 ----
    widths_s = (ends - starts) / fs
    min_w_s = min_pulse_width_us * 1e-6
    valid = widths_s >= min_w_s
    starts_v = starts[valid]
    widths_v = widths_s[valid]

    if len(starts_v) < min_pulses:
        return {"pri_hz": None,
                "pulse_width_us": round(float(np.median(widths_v)) * 1e6, 2) if len(widths_v) else None,
                "pulse_count": int(valid.sum()),
                "duty_cycle": round(duty, 4), "threshold": round(thr, 6),
                "method": threshold_method}

    # ---- PRI ----
    pri_s = np.diff(starts_v) / fs
    min_p_s = min_pri_us * 1e-6
    pri_valid = pri_s[pri_s >= min_p_s]
    if len(pri_valid) < min_pulses - 1:
        return {"pri_hz": None,
                "pulse_width_us": round(float(np.median(widths_v)) * 1e6, 2),
                "pulse_count": int(valid.sum()),
                "duty_cycle": round(duty, 4), "threshold": round(thr, 6),
                "method": threshold_method}

    # ---- 脉宽离散度 ----
    pw_mean = float(np.mean(widths_v))
    pw_cv = float(np.std(widths_v) / pw_mean) if pw_mean > 0 else 999.0
    if pw_cv > max_pw_cv:
        return {"pri_hz": None,
                "pulse_width_us": round(float(np.median(widths_v)) * 1e6, 2),
                "pulse_count": int(valid.sum()),
                "duty_cycle": round(duty, 4), "threshold": round(thr, 6),
                "method": threshold_method}

    pri_hz = float(1.0 / np.median(pri_valid))
    pw_us = float(np.median(widths_v)) * 1e6
    return {
        "pri_hz": round(pri_hz, 2),
        "pulse_width_us": round(pw_us, 2),
        "pulse_count": int(valid.sum()),
        "duty_cycle": round(duty, 4),
        "threshold": round(thr, 6),
        "method": threshold_method,
    }


def method_mad(name: str) -> bool:
    return name in ("mad", "median", "med", "")


def _otsu_threshold(env: np.ndarray, bins: int = 256) -> float:
    """大津法:最大化类间方差,自动分噪声/脉冲两级。"""
    hist, edges = np.histogram(env, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(np.mean(env))
    w0 = np.cumsum(hist)
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.cumsum(hist * centers) / np.where(w0 > 0, w0, 1)
        mu1 = (np.cumsum((hist * centers)[::-1])[::-1]) / np.where(w1 > 0, w1, 1)
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = 0
    idx = int(np.argmax(between))
    return float(centers[idx])
