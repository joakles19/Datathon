"""Canonical WAV preprocessing and 40-dimensional MFCC extraction."""

from pathlib import Path

import librosa
import numpy as np

from .config import (
    FEATURE_ORDER,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    RESAMPLE_TYPE,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)


class AudioProcessingError(ValueError):
    """Raised when an audio input cannot meet the feature contract."""


def prepare_audio(samples: np.ndarray, source_sample_rate: int) -> np.ndarray:
    """Convert audio to the fixed 16 kHz, 5-second analysis window."""
    if not isinstance(source_sample_rate, (int, np.integer)) or source_sample_rate <= 0:
        raise AudioProcessingError("Sample rate must be a positive integer")

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[1] not in (1, 2):
            raise AudioProcessingError("Audio must have one or two channels")
        waveform = np.mean(waveform, axis=1)
    elif waveform.ndim != 1:
        raise AudioProcessingError("Audio must be mono or have shape (samples, channels)")
    if waveform.size == 0:
        raise AudioProcessingError("Audio file is empty")
    if not np.isfinite(waveform).all():
        raise AudioProcessingError("Audio contains non-finite sample values")

    duration_seconds = waveform.size / source_sample_rate
    if duration_seconds > WINDOW_SAMPLES / SAMPLE_RATE:
        raise AudioProcessingError(
            f"Audio is longer than {WINDOW_SAMPLES / SAMPLE_RATE:g} seconds "
            f"({duration_seconds:.3f} seconds)"
        )

    if source_sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=source_sample_rate,
            target_sr=SAMPLE_RATE,
            res_type=RESAMPLE_TYPE,
        )

    if waveform.size > WINDOW_SAMPLES:
        # Resampling may round the output length up by a sample at the 5-second boundary.
        waveform = waveform[:WINDOW_SAMPLES]
    elif waveform.size < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.size))

    if not np.isfinite(waveform).all():
        raise AudioProcessingError("Audio preprocessing produced non-finite values")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def extract_features(samples: np.ndarray, source_sample_rate: int) -> np.ndarray:
    """Return 20 MFCC means followed by 20 MFCC standard deviations."""
    waveform = prepare_audio(samples, source_sample_rate)
    coefficients = librosa.feature.mfcc(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        dct_type=2,
        norm="ortho",
        lifter=0,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2,
        power=2.0,
        htk=False,
        mel_norm="slaney",
        center=True,
        pad_mode="constant",
    )
    means = np.mean(coefficients, axis=1)
    standard_deviations = np.std(coefficients, axis=1, ddof=0)
    features = np.concatenate((means, standard_deviations)).astype(np.float32)

    if features.shape != (len(FEATURE_ORDER),) or not np.isfinite(features).all():
        raise AudioProcessingError("MFCC extraction produced invalid features")
    return features


def extract_features_from_file(audio_path: str | Path) -> np.ndarray:
    """Load a WAV file and extract its features using the shared pipeline."""
    try:
        samples, source_sample_rate = librosa.load(
            str(audio_path),
            sr=None,
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:
        raise AudioProcessingError(f"Could not decode audio file: {audio_path}") from exc

    return extract_features(samples, source_sample_rate)
