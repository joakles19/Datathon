from __future__ import annotations

import numpy as np
import soundfile as sf

from . import filters, util

__version__ = "linuxone-compat-1"


def load(path, *, sr=None, mono=True, dtype=np.float32):
    samples, source_sr = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=dtype)
    if mono and samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sr is not None and int(sr) != int(source_sr):
        samples = resample(samples, orig_sr=source_sr, target_sr=sr)
        source_sr = int(sr)
    return np.ascontiguousarray(samples, dtype=dtype), int(source_sr)


def resample(y, *, orig_sr, target_sr, res_type="soxr_hq"):
    del res_type
    source = np.asarray(y, dtype=np.float32)
    output_length = int(round(len(source) * int(target_sr) / int(orig_sr)))
    if output_length <= 1 or len(source) <= 1:
        return np.resize(source, output_length).astype(np.float32)
    source_positions = np.arange(len(source), dtype=np.float64)
    target_positions = np.linspace(0.0, len(source) - 1, output_length)
    return np.interp(target_positions, source_positions, source).astype(np.float32)


class _Exceptions:
    class ParameterError(ValueError):
        pass


util.exceptions = _Exceptions
