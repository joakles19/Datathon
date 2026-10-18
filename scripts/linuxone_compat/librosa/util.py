from __future__ import annotations

import numpy as np


def pad_center(data, *, size, axis=-1):
    data = np.asarray(data)
    total = int(size) - data.shape[axis]
    if total < 0:
        raise ValueError("target size is smaller than input")
    padding = [(0, 0)] * data.ndim
    padding[axis] = (total // 2, total - total // 2)
    return np.pad(data, padding)


def normalize(data, norm=None, axis=0, threshold=None, fill=None):
    del threshold, fill
    data = np.asarray(data, dtype=np.float32)
    if norm is None:
        return data
    scale = np.max(np.abs(data), axis=axis, keepdims=True)
    return data / np.where(scale == 0, 1.0, scale)
