"""Spectral and time-domain metrics for audio comparison.

Every function accepts tensors of shape (B, 1, T) or (B, T) and returns a Python
float (aggregated over the batch). GPU-friendly — all ops are torch.
"""
import torch
import torch.nn.functional as F


def _as_bt(x):
    return x.squeeze(1) if x.dim() == 3 else x


def _stft_mag(x, n_fft: int, hop: int):
    x = _as_bt(x)
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return spec.abs()


def stft_l1(pred, target, n_fft: int = 2048, hop: int = 512) -> float:
    """L1 between linear magnitude spectrograms."""
    return F.l1_loss(_stft_mag(pred, n_fft, hop), _stft_mag(target, n_fft, hop)).item()


def log_stft_l1(pred, target, n_fft: int = 2048, hop: int = 512, eps: float = 1e-7) -> float:
    """L1 between log-magnitude spectrograms. Perceptually closer to auditory distance."""
    p = torch.log(_stft_mag(pred, n_fft, hop) + eps)
    t = torch.log(_stft_mag(target, n_fft, hop) + eps)
    return F.l1_loss(p, t).item()


def mrstft_l1(pred, target, n_ffts=(256, 512, 1024, 2048)) -> float:
    """Multi-resolution STFT L1, averaged across resolutions."""
    total = 0.0
    for n_fft in n_ffts:
        total += stft_l1(pred, target, n_fft=n_fft, hop=n_fft // 4)
    return total / len(n_ffts)


def rms_envelope_l1(pred, target, window: int = 1024, hop: int = 256) -> float:
    """L1 between RMS envelopes — directly measures compression fidelity."""
    def rms(x):
        x = _as_bt(x)
        return x.unfold(-1, window, hop).pow(2).mean(-1).sqrt()
    return F.l1_loss(rms(pred), rms(target)).item()


def si_sdr_db(pred, target, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio in dB. Higher is better."""
    pred = _as_bt(pred)
    target = _as_bt(target)
    pred = pred - pred.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    alpha = (pred * target).sum(-1, keepdim=True) / (target.pow(2).sum(-1, keepdim=True) + eps)
    s_target = alpha * target
    e_noise = pred - s_target
    ratio = (s_target.pow(2).sum(-1) + eps) / (e_noise.pow(2).sum(-1) + eps)
    return (10.0 * torch.log10(ratio)).mean().item()


def _spectral_centroid_hz(x, sample_rate: int, n_fft: int = 2048, hop: int = 512) -> float:
    mag = _stft_mag(x, n_fft, hop)
    freqs = torch.linspace(0, sample_rate / 2, mag.size(-2), device=mag.device, dtype=mag.dtype)
    centroid = (mag * freqs.view(-1, 1)).sum(-2) / (mag.sum(-2) + 1e-8)
    return centroid.mean().item()


def spectral_centroid_error(pred, target, sample_rate: int = 44100) -> float:
    """|centroid(pred) - centroid(target)| in Hz — measures brightness shift."""
    return abs(
        _spectral_centroid_hz(pred, sample_rate)
        - _spectral_centroid_hz(target, sample_rate)
    )


def all_metrics(pred, target, sample_rate: int = 44100) -> dict[str, float]:
    """Compute every metric in one pass. Convenience for per-batch evaluation."""
    return {
        "stft_l1": stft_l1(pred, target),
        "log_stft_l1": log_stft_l1(pred, target),
        "mrstft_l1": mrstft_l1(pred, target),
        "rms_env_l1": rms_envelope_l1(pred, target),
        "si_sdr_db": si_sdr_db(pred, target),
        "centroid_err_hz": spectral_centroid_error(pred, target, sample_rate),
    }
