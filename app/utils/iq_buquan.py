"""IQ 补全(ISTA_Net),从 dc 项目移植,去掉对 dc config 的依赖,模型路径改 qingxi settings。

对外接口:
- buquan_iq(iq)        -> 补全后的一维复数 IQ
- buquan_iq_both(iq)   -> (补全后, 补全前) 两个一维复数 IQ
- IQCompleter.complete -> (recovered, masked, full_stft, masked_stft, mask)
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal

from ..config import settings

N_FFT = 32
HOP_LENGTH = 8
MASK_RATIO = 0.4
N_LAYERS = 9
N_CHANNELS = 32
KERNEL_SIZE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def iq_to_stft(iq, n_fft, hop_length):
    _, _, Z = signal.stft(iq, nperseg=n_fft, noverlap=n_fft - hop_length)
    return np.stack([Z.real, Z.imag], axis=0).astype(np.float32)


def random_mask(stft_img, mask_ratio):
    _, F, T = stft_img.shape
    mask = (np.random.rand(F, T) > mask_ratio).astype(np.float32)
    return stft_img * mask[None, :, :], mask


def stft_to_iq(stft_img, n_fft, hop_length):
    Z = stft_img[0] + 1j * stft_img[1]
    _, iq = signal.istft(Z, nperseg=n_fft, noverlap=n_fft - hop_length,
                         input_onesided=False)
    return iq


class BasicBlock(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        pad = k // 2
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, k, padding=pad),
        )

    def forward(self, x):
        return x + self.body(x)


class ISTA_Net(nn.Module):
    def __init__(self, n_layers=9, n_channels=32, k=3, in_ch=2):
        super().__init__()
        self.n_layers = n_layers
        self.in_ch = in_ch
        self.n_channels = n_channels

        self.embed = nn.Conv2d(in_ch + 1, n_channels, k, padding=k // 2)
        self.to_img = nn.ModuleList([nn.Conv2d(n_channels, in_ch, k, padding=k // 2) for _ in range(n_layers)])
        self.to_feat = nn.ModuleList([nn.Conv2d(in_ch, n_channels, k, padding=k // 2) for _ in range(n_layers)])
        self.prox = nn.ModuleList([BasicBlock(n_channels, k) for _ in range(n_layers)])

        self.steps = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)])
        self.thresholds = nn.ParameterList([nn.Parameter(torch.tensor(0.01)) for _ in range(n_layers)])

        self.final = nn.Conv2d(n_channels, in_ch, k, padding=k // 2)

    def soft_threshold(self, x, thr):
        return torch.sign(x) * torch.clamp(torch.abs(x) - thr, min=0.0)

    def forward(self, masked, mask):
        feat = self.embed(torch.cat([masked, mask], dim=1))
        for i in range(self.n_layers):
            img = self.to_img[i](feat)
            residual = (masked - img) * mask
            img = img + self.steps[i] * residual
            feat = self.to_feat[i](img)
            feat = self.prox[i](feat)
            feat = self.soft_threshold(feat, torch.abs(self.thresholds[i]))
        return self.final(feat)


class IQCompleter:
    """封装补全器,外部只调 complete(iq)。"""

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or DEVICE)
        self.model = ISTA_Net(N_LAYERS, N_CHANNELS, KERNEL_SIZE, in_ch=2).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
        self.model.eval()

    def complete(self, iq):
        """输入一维复数 IQ,返回 (补全后, 补全前, full_stft, masked_stft, mask)。"""
        iq = iq.astype(np.complex64)
        full_stft = iq_to_stft(iq, N_FFT, HOP_LENGTH)
        masked_stft, mask = random_mask(full_stft, MASK_RATIO)
        iq_masked = stft_to_iq(masked_stft, N_FFT, HOP_LENGTH)

        masked_t = torch.from_numpy(masked_stft)[None].to(self.device)
        mask_t = torch.from_numpy(mask)[None, None].to(self.device)
        with torch.no_grad():
            out = self.model(masked_t, mask_t)[0].cpu().numpy()

        iq_rec = stft_to_iq(out, N_FFT, HOP_LENGTH)
        return iq_rec, iq_masked, full_stft, masked_stft, mask


_completer = None


def get_completer():
    """懒加载全局单例,避免重复加载模型。"""
    global _completer
    if _completer is None:
        _completer = IQCompleter(settings.iq_buquan_model_path)
    return _completer


def buquan_iq_both(iq):
    """返回 (补全后, 补全前) 两个一维复数 IQ。"""
    comp = get_completer()
    iq_rec, iq_masked, _, _, _ = comp.complete(iq)
    return iq_rec, iq_masked


def buquan_iq(iq):
    """返回补全后的一维复数 IQ。"""
    return buquan_iq_both(iq)[0]
