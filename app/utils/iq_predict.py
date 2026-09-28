"""调制分类预测(XGBoost + CNN 加权融合),从 dc 项目移植,模型路径改 qingxi settings。"""

import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import xgboost as xgb

from ..config import settings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IQCNN(nn.Module):
    """CNN 调制分类模型(与 dc 训练的 cnn_15class_full.pth 结构一致)。"""

    def __init__(self, num_class):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


_label_map = None
_xgb_model = None
_cnn_model = None


def _load_models():
    """加载模型(全局缓存,只加载一次)。"""
    global _label_map, _xgb_model, _cnn_model
    if _xgb_model is not None and _cnn_model is not None:
        return

    with open(settings.iq_label_map_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    _label_map = {int(k): v for k, v in raw.items()}

    _xgb_model = xgb.XGBClassifier()
    _xgb_model.load_model(settings.iq_xgb_model_path)

    # cnn_15class_full.pth 里 pickle 引用 __main__.IQCNN,加载前把类注入 __main__
    _main = sys.modules.get("__main__")
    if _main is not None and not hasattr(_main, "IQCNN"):
        setattr(_main, "IQCNN", IQCNN)

    _cnn_model = torch.load(settings.iq_cnn_model_path, map_location=DEVICE, weights_only=False)
    _cnn_model.eval()


def _extract_xgb_features(iq):
    """提取 XGBoost 特征(24维)。"""
    I = iq[:, 0]
    Q = iq[:, 1]
    s = I + 1j * Q

    amp = np.abs(s)
    phase = np.unwrap(np.angle(s))
    inst_freq = np.diff(phase)

    amp_mean = np.mean(amp)
    amp_var = np.var(amp)
    phase_kurt = kurtosis(phase)
    phase_skew = skew(phase)
    freq_mean = np.mean(inst_freq)
    freq_var = np.var(inst_freq)
    freq_median = np.median(inst_freq)
    freq_std = np.std(inst_freq)
    freq_min = np.min(inst_freq)
    freq_max = np.max(inst_freq)

    f, Pxx = signal.welch(s, fs=1.0, nperseg=min(128, len(s)))
    psd_peak_freq = f[np.argmax(Pxx)]
    psd_peak_val = np.max(Pxx)
    spec_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    spec_bandwidth = np.sqrt(np.sum(((f - spec_centroid) ** 2) * Pxx) / np.sum(Pxx))
    spec_symmetry = np.sum(Pxx * np.abs(f - spec_centroid)) / np.sum(Pxx)

    S = fft(s)
    S_power = np.abs(S) ** 2
    cyc_energy = np.max(S_power)
    cyc_dominant = np.argmax(S_power) / len(S_power)
    sym_rate_peak = np.max(S_power[:len(S_power)//4])

    s_norm = s / np.sqrt(np.mean(np.abs(s) ** 2))
    C20 = np.mean(s_norm ** 2)
    C21 = np.mean(np.abs(s_norm) ** 2)
    C40 = np.mean(s_norm ** 4) - 3 * (np.mean(s_norm ** 2)) ** 2
    C42 = np.mean(np.abs(s_norm) ** 4) - np.abs(np.mean(s_norm ** 2)) ** 2 - 2 * (np.mean(np.abs(s_norm) ** 2)) ** 2
    C60 = np.mean(s_norm ** 6) - 15 * np.mean(s_norm ** 2) * np.mean(s_norm ** 4) + 30 * (np.mean(s_norm ** 2)) ** 3
    C63 = np.mean(np.abs(s_norm) ** 6) - 9 * np.mean(np.abs(s_norm) ** 4) * np.mean(np.abs(s_norm) ** 2) + 12 * (np.mean(np.abs(s_norm) ** 2)) ** 3

    return {
        'amp_mean': amp_mean, 'amp_var': amp_var,
        'phase_kurt': phase_kurt, 'phase_skew': phase_skew,
        'freq_mean': freq_mean, 'freq_var': freq_var, 'freq_median': freq_median,
        'freq_std': freq_std, 'freq_min': freq_min, 'freq_max': freq_max,
        'psd_peak_freq': psd_peak_freq, 'psd_peak_val': psd_peak_val,
        'spec_centroid': spec_centroid, 'spec_bandwidth': spec_bandwidth,
        'spec_symmetry': spec_symmetry,
        'cyc_energy': cyc_energy, 'cyc_dominant': cyc_dominant,
        'sym_rate_peak': sym_rate_peak,
        'C20': np.real(C20), 'C21': np.real(C21),
        'C40': np.real(C40), 'C42': np.real(C42),
        'C60': np.real(C60), 'C63': np.real(C63),
    }


def predict_xgb(iq: np.ndarray) -> dict:
    """XGBoost 预测,输入 (128, 2)。"""
    _load_models()

    iq = np.asarray(iq, dtype=np.float32)
    if iq.shape != (128, 2):
        raise ValueError(f"IQ 形状应为 (128, 2)，当前为 {iq.shape}")

    feats = _extract_xgb_features(iq)
    x = pd.DataFrame([feats])
    prob = _xgb_model.predict_proba(x)[0]

    pred_idx = int(np.argmax(prob))
    pred_name = _label_map.get(pred_idx, f'class_{pred_idx}')
    confidence = float(prob[pred_idx])

    prob_dict = {
        _label_map.get(i, f'class_{i}'): float(prob[i])
        for i in range(len(_label_map))
    }

    return {
        'class': pred_name,
        'confidence': confidence,
        'probabilities': prob_dict
    }


def predict_cnn(iq: np.ndarray) -> dict:
    """CNN 预测,输入 (128, 2)。"""
    _load_models()

    iq = np.asarray(iq, dtype=np.float32)
    if iq.shape != (128, 2):
        raise ValueError(f"IQ 形状应为 (128, 2)，当前为 {iq.shape}")

    x = torch.tensor(iq, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _cnn_model(x)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(prob))
    pred_name = _label_map.get(pred_idx, f'class_{pred_idx}')
    confidence = float(prob[pred_idx])

    prob_dict = {
        _label_map.get(i, f'class_{i}'): float(prob[i])
        for i in range(len(_label_map))
    }

    return {
        'class': pred_name,
        'confidence': confidence,
        'probabilities': prob_dict
    }


XGB_WEIGHT = 0.4
CNN_WEIGHT = 0.6


def predict_weighted(iq: np.ndarray) -> dict:
    """XGBoost + CNN 加权融合预测,输入 (128, 2)。"""
    _load_models()

    xgb_result = predict_xgb(iq)
    cnn_result = predict_cnn(iq)

    all_classes = set(list(xgb_result['probabilities'].keys()) + list(cnn_result['probabilities'].keys()))

    weighted_probs = {}
    for cls in all_classes:
        xgb_prob = xgb_result['probabilities'].get(cls, 0.0)
        cnn_prob = cnn_result['probabilities'].get(cls, 0.0)
        weighted_probs[cls] = XGB_WEIGHT * xgb_prob + CNN_WEIGHT * cnn_prob

    weighted_class = max(weighted_probs, key=weighted_probs.get)
    weighted_confidence = weighted_probs[weighted_class]

    return {
        'xgb': xgb_result,
        'cnn': cnn_result,
        'weighted': {
            'class': weighted_class,
            'confidence': weighted_confidence,
            'probabilities': weighted_probs,
            'weights': {
                'xgb': XGB_WEIGHT,
                'cnn': CNN_WEIGHT,
            }
        }
    }
