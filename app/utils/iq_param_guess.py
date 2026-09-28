import numpy as np
from scipy.signal import find_peaks


# ============================================================
# 全部参数估计：一个函数搞定
# ============================================================
def analyze_iq(iq, mod_id, fs):
    """
    传入IQ和调制类别，返回全部参数估计

    参数:
        iq      : 基带IQ数据 (复数数组)
        mod_id  : 调制类别 (0~14)
        fs      : 采样率 (Hz)

    返回:
        dict: {
            'modulation'  : 调制名称,
            'carrier'     : 载波频率 (Hz),
            'bandwidth'   : 信号带宽 (Hz),
            'freq_offset' : 频偏 (Hz),
            'phase_offset': 相偏 (rad),
            'symbol_rate' : 符号速率 (Hz),
            'EVM'         : EVM (%),
            'SNR'         : SNR (dB),
            'eye_height'  : 垂直开口度 (0~1),
            'eye_width'   : 水平开口度 (0~1),
            'eye_opening' : 综合开口度 (0~1),
            'q_factor'    : Q因子,
        }
    """
    # ---------- 类别定义 ----------
    MOD_NAMES = {
        0: "BPSK", 1: "QPSK", 2: "8PSK", 3: "16QAM", 4: "64QAM",
        5: "256QAM", 6: "16APSK", 7: "32APSK", 8: "2FSK", 9: "4FSK",
        10: "MSK", 11: "GMSK", 12: "OQPSK", 13: "pi/4-DQPSK", 14: "Unknown"
    }
    EVM_APPLICABLE = {
        0: True, 1: True, 2: True, 3: True, 4: True, 5: True,
        6: True, 7: True, 8: False, 9: False, 10: False, 11: False,
        12: True, 13: True, 14: False
    }
    EYE_DIMENSION = {
        0: 'I', 1: 'IQ', 2: 'IQ', 3: 'IQ', 4: 'IQ', 5: 'IQ',
        6: 'IQ', 7: 'IQ', 8: 'freq', 9: 'freq', 10: 'freq',
        11: 'freq', 12: 'IQ', 13: 'IQ', 14: 'IQ'
    }
    FREQ_OFFSET_METHOD = {
        0: 'nonlinear2', 1: 'nonlinear4', 2: 'nonlinear8',
        3: 'decision', 4: 'decision', 5: 'decision',
        6: 'decision', 7: 'decision',
        8: 'inst_freq', 9: 'inst_freq', 10: 'inst_freq', 11: 'inst_freq',
        12: 'nonlinear4', 13: 'nonlinear4', 14: 'spectrum'
    }
    PHASE_OFFSET_METHOD = {
        0: 'nonlinear2', 1: 'nonlinear4', 2: 'nonlinear8',
        3: 'decision', 4: 'decision', 5: 'decision',
        6: 'decision', 7: 'decision',
        8: 'inst_freq', 9: 'inst_freq', 10: 'inst_freq', 11: 'inst_freq',
        12: 'nonlinear4', 13: 'none', 14: 'spectrum'
    }
    CYCLIC_SECTION = {
        0: 'fc', 1: 'fc', 2: 'fc', 3: 'fc', 4: 'fc', 5: 'fc',
        6: 'fc', 7: 'fc', 8: 'zero', 9: 'zero', 10: 'zero',
        11: 'zero', 12: 'fc', 13: 'fc', 14: 'fc'
    }
    BANDWIDTH_METHOD = {
        0: 'main_lobe', 1: 'main_lobe', 2: 'main_lobe',
        3: 'main_lobe', 4: 'main_lobe', 5: 'main_lobe',
        6: 'main_lobe', 7: 'main_lobe',
        8: 'inst_freq', 9: 'inst_freq', 10: 'inst_freq', 11: 'inst_freq',
        12: 'main_lobe', 13: 'main_lobe', 14: 'occupied'
    }

    # ---------- 内部工具 ----------
    def _parabolic_interp(P, idx, freqs):
        if idx <= 0 or idx >= len(P) - 1:
            return freqs[idx]
        y0, y1, y2 = P[idx - 1], P[idx], P[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom == 0:
            return freqs[idx]
        delta = 0.5 * (y0 - y2) / denom
        df = freqs[1] - freqs[0]
        return freqs[idx] + delta * df

    def _parabolic_interp_alpha(spec, idx, alpha_pos):
        if idx <= 0 or idx >= len(spec) - 1:
            return alpha_pos[idx]
        y0, y1, y2 = spec[idx - 1], spec[idx], spec[idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom == 0:
            return alpha_pos[idx]
        delta = 0.5 * (y0 - y2) / denom
        d_alpha = alpha_pos[1] - alpha_pos[0]
        return alpha_pos[idx] + delta * d_alpha

    # ---------- 判决函数 ----------
    def _decide_qam(symbols, side):
        levels = np.arange(-(side - 1), side, 2).astype(float)
        scale = np.sqrt(np.mean(levels ** 2) * 2)
        levels = levels / scale
        ref = np.array([complex(i, j) for i in levels for j in levels])
        idx = np.argmin(np.abs(symbols[:, None] - ref[None, :]), axis=1)
        return ref[idx]

    def _decide_apsk(symbols, rings):
        r = np.abs(symbols)
        radii = np.array([rr for _, rr in rings])
        ring_idx = np.argmin(np.abs(r[:, None] - radii[None, :]), axis=1)
        ref = np.zeros(len(symbols), dtype=complex)
        for i, (n_pts, radius) in enumerate(rings):
            mask = ring_idx == i
            if np.any(mask):
                angles = np.angle(symbols[mask])
                step = 2 * np.pi / n_pts
                idx = np.round(angles / step).astype(int) % n_pts
                ref[mask] = radius * np.exp(1j * idx * step)
        return ref

    def decide(symbols, mid):
        symbols = np.asarray(symbols)
        if mid == 0:
            return np.sign(symbols.real).astype(complex)
        elif mid in (1, 12):
            ref = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            idx = np.argmin(np.abs(symbols[:, None] - ref[None, :]), axis=1)
            return ref[idx]
        elif mid == 2:
            angles = np.angle(symbols)
            idx = np.round(angles / (np.pi / 4)).astype(int) % 8
            return np.exp(1j * idx * np.pi / 4)
        elif mid == 3:
            return _decide_qam(symbols, 4)
        elif mid == 4:
            return _decide_qam(symbols, 8)
        elif mid == 5:
            return _decide_qam(symbols, 16)
        elif mid == 6:
            return _decide_apsk(symbols, [(4, 1.0), (12, 2.7)])
        elif mid == 7:
            return _decide_apsk(symbols, [(4, 1.0), (12, 2.7), (16, 4.5)])
        elif mid == 13:
            diff = symbols[1:] * np.conj(symbols[:-1])
            angles = np.angle(diff)
            idx = np.round(angles / (np.pi / 4)).astype(int) % 8
            return np.exp(1j * idx * np.pi / 4)
        else:
            return symbols

    # ---------- 载波频率 ----------
    def estimate_carrier(x, fs_):
        x = x - np.mean(x)
        N = len(x)
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(x * win))
        P = np.abs(X) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs_))
        return float(_parabolic_interp(P, np.argmax(P), freqs))

    # ---------- 带宽 ----------
    def _bw_main_lobe(x, fs_):
        x = x - np.mean(x)
        N = len(x)
        n_fft = max(N, 4096)
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(x * win, n_fft))
        P = np.abs(X) ** 2
        P = P / np.max(P)
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / fs_))
        peak_idx = np.argmax(P)
        left = peak_idx
        while left > 0 and P[left] > 0.5:
            left -= 1
        right = peak_idx
        while right < len(P) - 1 and P[right] > 0.5:
            right += 1
        return float(freqs[right] - freqs[left])

    def _bw_occupied(x, fs_, ratio=0.99):
        x = x - np.mean(x)
        N = len(x)
        n_fft = max(N, 4096)
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(x * win, n_fft))
        P = np.abs(X) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / fs_))
        cumP = np.cumsum(P) / np.sum(P)
        lo = np.searchsorted(cumP, (1 - ratio) / 2)
        hi = np.searchsorted(cumP, 1 - (1 - ratio) / 2)
        return float(freqs[hi] - freqs[lo])

    def _bw_inst_freq(x, fs_):
        x = x - np.mean(x)
        phase = np.unwrap(np.angle(x))
        inst_freq = np.diff(phase) * fs_ / (2 * np.pi)
        inst_freq = inst_freq - np.mean(inst_freq)
        f_low = np.percentile(inst_freq, 1)
        f_high = np.percentile(inst_freq, 99)
        return float(f_high - f_low)

    def estimate_bandwidth(x, mid, fs_):
        method = BANDWIDTH_METHOD.get(mid, 'occupied')
        if method == 'main_lobe':
            return _bw_main_lobe(x, fs_)
        elif method == 'inst_freq':
            return _bw_inst_freq(x, fs_)
        else:
            return _bw_occupied(x, fs_)

    # ---------- 频偏 ----------
    def _freq_offset_nonlinear(x, fs_, order):
        x = x - np.mean(x)
        N = len(x)
        x_pow = x ** order
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(x_pow * win))
        P = np.abs(X) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs_))
        return float(_parabolic_interp(P, np.argmax(P), freqs) / order)

    def _freq_offset_spectrum(x, fs_):
        x = x - np.mean(x)
        N = len(x)
        win = np.hanning(N)
        X = np.fft.fftshift(np.fft.fft(x * win))
        P = np.abs(X) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs_))
        return float(_parabolic_interp(P, np.argmax(P), freqs))

    def _freq_offset_decision(x, mid, fs_, n_iter=5):
        x = x - np.mean(x)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        f_off = _freq_offset_spectrum(x, fs_)
        t = np.arange(len(x)) / fs_
        for _ in range(n_iter):
            x_corr = x * np.exp(-1j * 2 * np.pi * f_off * t)
            step = max(1, len(x_corr) // 2000)
            sym = x_corr[::step]
            ref = decide(sym, mid)
            err = sym * np.conj(ref)
            f_off += np.angle(np.mean(err)) * fs_ / (2 * np.pi * len(x_corr))
        return float(f_off)

    def _freq_offset_inst_freq(x, fs_):
        x = x - np.mean(x)
        phase = np.unwrap(np.angle(x))
        return float(np.mean(np.diff(phase) * fs_ / (2 * np.pi)))

    def estimate_freq_offset(x, mid, fs_):
        method = FREQ_OFFSET_METHOD.get(mid, 'spectrum')
        if method == 'nonlinear2':
            return _freq_offset_nonlinear(x, fs_, 2)
        elif method == 'nonlinear4':
            return _freq_offset_nonlinear(x, fs_, 4)
        elif method == 'nonlinear8':
            return _freq_offset_nonlinear(x, fs_, 8)
        elif method == 'decision':
            return _freq_offset_decision(x, mid, fs_)
        elif method == 'inst_freq':
            return _freq_offset_inst_freq(x, fs_)
        else:
            return _freq_offset_spectrum(x, fs_)

    # ---------- 相偏 ----------
    def _phase_offset_nonlinear(x, order):
        x = x - np.mean(x)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        mean_phase = np.angle(np.mean(x ** order))
        phase_off = mean_phase / order
        return float(np.angle(np.exp(1j * phase_off * order)) / order)

    def _phase_offset_decision(x, mid):
        x = x - np.mean(x)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        step = max(1, len(x) // 2000)
        sym = x[::step]
        ref = decide(sym, mid)
        return float(np.angle(np.mean(sym * np.conj(ref))))

    def _phase_offset_inst_freq(x):
        x = x - np.mean(x)
        return float(np.mean(np.unwrap(np.angle(x))))

    def _phase_offset_spectrum(x):
        x = x - np.mean(x)
        X = np.fft.fftshift(np.fft.fft(x))
        P = np.abs(X) ** 2
        return float(np.angle(X[np.argmax(P)]))

    def estimate_phase_offset(x, mid):
        method = PHASE_OFFSET_METHOD.get(mid, 'spectrum')
        if method == 'nonlinear2':
            return _phase_offset_nonlinear(x, 2)
        elif method == 'nonlinear4':
            return _phase_offset_nonlinear(x, 4)
        elif method == 'nonlinear8':
            return _phase_offset_nonlinear(x, 8)
        elif method == 'decision':
            return _phase_offset_decision(x, mid)
        elif method == 'inst_freq':
            return _phase_offset_inst_freq(x)
        elif method == 'none':
            return 0.0
        else:
            return _phase_offset_spectrum(x)

    # ---------- 符号速率 ----------
    def estimate_symbol_rate(x, mid, fs_, N_fft=1024, n_avg=64,
                             skip_ratio=0.002, peak_ratio=0.3):
        x = x - np.mean(x)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        L = N_fft
        P = N_fft // 4
        n_seg = min((len(x) - L) // (L - P) + 1, n_avg)
        if n_seg < 1:
            return None
        win = np.hanning(L)
        S = np.zeros((2 * N_fft + 1, N_fft), dtype=complex)
        for i in range(n_seg):
            seg = x[i * (L - P):i * (L - P) + L] * win
            X = np.fft.fftshift(np.fft.fft(seg, N_fft))
            for a_idx in range(2 * N_fft + 1):
                shift = a_idx - N_fft
                S[a_idx] += X * np.conj(np.roll(X, shift))
        S /= n_seg

        alpha = np.fft.fftshift(np.fft.fftfreq(2 * N_fft + 1, 1 / fs_))
        f_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / fs_))
        pos_idx = np.where(alpha > 0)[0]
        alpha_pos = alpha[pos_idx]
        skip = np.searchsorted(alpha_pos, fs_ * skip_ratio)

        section = CYCLIC_SECTION.get(mid, 'fc')
        f_sections = (-0.25, -0.125, 0, 0.125, 0.25) if section == 'fc' \
                     else (-0.02, -0.01, 0, 0.01, 0.02)

        candidates = []
        for f_rel in f_sections:
            f_idx = np.argmin(np.abs(f_axis - f_rel * fs_))
            spec = np.abs(S[pos_idx, f_idx])
            if len(spec[skip:]) == 0:
                continue
            peaks, _ = find_peaks(
                spec[skip:],
                height=np.max(spec[skip:]) * peak_ratio,
                distance=max(1, len(spec[skip:]) // 50)
            )
            if len(peaks) == 0:
                continue
            first_peak = skip + peaks[0]
            Rs_cand = _parabolic_interp_alpha(spec, first_peak, alpha_pos)
            candidates.append((Rs_cand, spec[first_peak]))

        if not candidates:
            return None
        vals = np.array([c[1] for c in candidates])
        Rs_arr = np.array([c[0] for c in candidates])
        return float(np.sum(Rs_arr * vals) / np.sum(vals))

    # ---------- 符号定时同步 ----------
    def symbol_sync(x, fs_, Rs):
        Ts = fs_ / Rs
        n_sym = int(len(x) / Ts)
        samples = []
        for i in range(n_sym):
            start = int(i * Ts)
            end = int((i + 1) * Ts)
            if end <= len(x):
                seg = x[start:end]
                samples.append(seg[np.argmax(np.abs(seg))])
        return np.array(samples)

    # ---------- 眼图开口度 ----------
    def _eye_metrics_single(signal):
        signal = np.asarray(signal).flatten()
        if len(signal) < 10:
            return {'eye_height': 0, 'eye_width': 0,
                    'eye_opening': 0, 'q_factor': 0}

        levels = np.sort(np.unique(np.round(signal, 1)))
        if len(levels) < 2:
            eye_height = 0.0
        else:
            hist, edges = np.histogram(signal, bins=50)
            peaks, _ = find_peaks(hist, height=np.max(hist) * 0.2)
            if len(peaks) >= 2:
                peak_vals = edges[peaks]
                high = np.max(peak_vals)
                low = np.min(peak_vals)
                eye_height = (high - low) / (np.max(signal) - np.min(signal) + 1e-12)
            else:
                eye_height = 0.0

        sign_changes = np.where(np.diff(np.sign(signal)))[0]
        if len(sign_changes) > 2:
            intervals = np.diff(sign_changes)
            eye_width = np.min(intervals) / (np.mean(intervals) + 1e-12)
            eye_width = min(eye_width, 1.0)
        else:
            eye_width = 0.0

        if len(levels) >= 2:
            mu = np.mean(signal)
            high_mask = signal > mu
            low_mask = signal <= mu
            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                mu1 = np.mean(signal[high_mask])
                mu0 = np.mean(signal[low_mask])
                sig1 = np.std(signal[high_mask])
                sig0 = np.std(signal[low_mask])
                q_factor = (mu1 - mu0) / (sig1 + sig0) if sig1 + sig0 > 0 else 0.0
            else:
                q_factor = 0.0
        else:
            q_factor = 0.0

        eye_opening = 0.5 * eye_height + 0.5 * eye_width
        return {'eye_height': float(eye_height),
                'eye_width': float(eye_width),
                'eye_opening': float(eye_opening),
                'q_factor': float(q_factor)}

    def eye_opening(symbols, mid):
        symbols = np.asarray(symbols)
        if len(symbols) < 10:
            return None
        dim = EYE_DIMENSION.get(mid, 'IQ')
        if dim == 'freq':
            phase = np.unwrap(np.angle(symbols))
            signal = np.diff(phase)
            signal = signal - np.mean(signal)
            return _eye_metrics_single(signal)
        elif dim == 'I':
            return _eye_metrics_single(symbols.real)
        else:
            ri = _eye_metrics_single(symbols.real)
            rq = _eye_metrics_single(symbols.imag)
            return {
                'eye_height': min(ri['eye_height'], rq['eye_height']),
                'eye_width': min(ri['eye_width'], rq['eye_width']),
                'eye_opening': min(ri['eye_opening'], rq['eye_opening']),
                'q_factor': min(ri['q_factor'], rq['q_factor']),
            }

    # ============================================================
    # 主流程
    # ============================================================
    iq = np.asarray(iq, dtype=np.complex128)
    results = {'modulation': MOD_NAMES.get(mod_id, 'Unknown')}

    # 0. 载波频率 + 带宽（所有调制都适用）
    results['carrier'] = estimate_carrier(iq, fs)
    results['bandwidth'] = estimate_bandwidth(iq, mod_id, fs)

    # 不适用EVM的调制，直接返回
    if not EVM_APPLICABLE.get(mod_id, False):
        results.update({
            'freq_offset': None, 'phase_offset': None, 'symbol_rate': None,
            'EVM': None, 'SNR': None,
            'eye_height': None, 'eye_width': None,
            'eye_opening': None, 'q_factor': None,
            'note': '该调制不适用EVM/眼图'
        })
        return results

    # 1. 频偏
    f_off = estimate_freq_offset(iq, mod_id, fs)
    results['freq_offset'] = f_off

    # 2. 校正频偏
    t = np.arange(len(iq)) / fs
    iq_corr = iq * np.exp(-1j * 2 * np.pi * f_off * t)

    # 3. 相偏
    phase = estimate_phase_offset(iq_corr, mod_id)
    results['phase_offset'] = phase

    # 4. 校正相偏
    iq_corr = iq_corr * np.exp(-1j * phase)

    # 5. 符号速率
    Rs = estimate_symbol_rate(iq_corr, mod_id, fs)
    results['symbol_rate'] = Rs
    if Rs is None or Rs <= 0:
        results.update({'EVM': None, 'SNR': None,
                        'eye_height': None, 'eye_width': None,
                        'eye_opening': None, 'q_factor': None})
        return results

    # 6. 符号定时同步
    symbols = symbol_sync(iq_corr, fs, Rs)
    if len(symbols) < 10:
        results.update({'EVM': None, 'SNR': None,
                        'eye_height': None, 'eye_width': None,
                        'eye_opening': None, 'q_factor': None})
        return results

    # 7. 功率归一化
    symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    # 8. 判决
    ref = decide(symbols, mod_id)
    symbols_for_evm = symbols
    if mod_id == 13:
        symbols_for_evm = symbols[1:]
        n = min(len(symbols_for_evm), len(ref))
        symbols_for_evm = symbols_for_evm[:n]
        ref = ref[:n]
    ref = ref / np.sqrt(np.mean(np.abs(ref) ** 2))

    # 9. EVM
    err = symbols_for_evm - ref
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(ref) ** 2)) * 100
    results['EVM'] = float(evm)

    # 10. SNR
    results['SNR'] = float(-20 * np.log10(evm / 100)) if evm > 0 else float('inf')

    # 11. 眼图
    eye = eye_opening(symbols, mod_id)
    if eye is not None:
        results.update(eye)

    return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    fs = 1e6
    n = 200000
    t = np.arange(n) / fs

    # QPSK 仿真
    Rs_true = 50e3
    n_sym = int(n / (fs / Rs_true))
    bits = np.random.randint(0, 4, n_sym)
    sym = np.exp(1j * (np.pi / 4 + bits * np.pi / 2))
    sym_up = np.repeat(sym, int(fs / Rs_true))[:n]
    iq = sym_up * np.exp(1j * (2 * np.pi * 5e3 * t + 0.3))
    iq += (np.random.randn(n) + 1j * np.random.randn(n)) * 0.05

    results = analyze_iq(iq, mod_id=1, fs=fs)
    print("=" * 55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")