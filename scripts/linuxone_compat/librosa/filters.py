from __future__ import annotations

import numpy as np


def get_window(window, Nx, fftbins=True):
    Nx = int(Nx)
    if isinstance(window, str) and window.lower() in {"hann", "hanning"}:
        return np.hanning(Nx + 1)[:-1] if fftbins else np.hanning(Nx)
    if isinstance(window, str) and window.lower() == "ones":
        return np.ones(Nx)
    raise ValueError(f"unsupported window: {window}")


def _hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, **kwargs):
    del kwargs
    fmax = float(sr) / 2.0 if fmax is None else float(fmax)
    edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), int(n_mels) + 2)
    bins = np.floor((int(n_fft) + 1) * _mel_to_hz(edges) / float(sr)).astype(int)
    matrix = np.zeros((int(n_mels), int(n_fft) // 2 + 1), dtype=np.float32)
    for index in range(int(n_mels)):
        left, center, right = bins[index:index + 3]
        if center > left:
            matrix[index, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            matrix[index, center:right] = (right - np.arange(center, right)) / float(right - center)
    return matrix


def window_sumsquare(window, n_frames, win_length, n_fft, hop_length, **kwargs):
    del kwargs
    win = get_window(window, win_length, fftbins=True)
    left = (int(n_fft) - int(win_length)) // 2
    padded = np.pad(win ** 2, (left, int(n_fft) - int(win_length) - left))
    result = np.zeros(int(n_fft) + int(hop_length) * (int(n_frames) - 1), dtype=np.float32)
    for frame in range(int(n_frames)):
        start = frame * int(hop_length)
        result[start:start + len(padded)] += padded
    return result
