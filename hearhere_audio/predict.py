"""Validated inference using A1's saved model and canonical MFCC extractor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import TypedDict
import warnings

import joblib
from librosa.util.exceptions import ParameterError
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.utils.validation import check_is_fitted
import soundfile as sf

from . import config
from .features import AudioProcessingError, extract_features_from_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/hearhere_rf.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"
DEFAULT_MAX_AUDIO_BYTES = 10 * 1024 * 1024
WAV_FORMATS = frozenset({"WAV", "WAVEX", "RF64"})


class ModelLoadError(RuntimeError):
    """The supplied model, metadata or runtime cannot meet the contract."""


class PredictionError(RuntimeError):
    """Inference failed or the estimator returned invalid scores."""


class PredictionResult(TypedDict):
    predicted_class: str
    model_score: float
    scores: dict[str, float]
    model_version: str
    processing_ms: float


def _read_metadata(path: Path) -> dict:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"Could not read model metadata: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ModelLoadError("Model metadata must be a JSON object")
    if metadata.get("schema_version") != "1.0":
        raise ModelLoadError("Unsupported metadata schema; expected schema_version '1.0'")
    version = metadata.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ModelLoadError("Metadata must contain a non-empty model_version")
    if metadata.get("model_type") != "sklearn.ensemble.RandomForestClassifier":
        raise ModelLoadError("Metadata must declare sklearn.ensemble.RandomForestClassifier")
    digest = metadata.get("model_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ModelLoadError("Metadata must contain a 64-character model_sha256")
    classes = metadata.get("class_order")
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ModelLoadError("Metadata class_order must be a list of class labels")
    if len(classes) != len(config.CLASSES) or set(classes) != set(config.CLASSES):
        raise ModelLoadError("Metadata class_order must contain each of the eight HearHere labels once")
    feature = metadata.get("feature_extractor")
    if not isinstance(feature, dict):
        raise ModelLoadError("Metadata must contain feature_extractor settings")
    expected = {
        "entrypoint": "hearhere_audio.features.extract_features_from_file",
        "channels": 1,
        "sample_rate_hz": config.SAMPLE_RATE,
        "window_seconds": config.WINDOW_SECONDS,
        "feature_count": len(config.FEATURE_ORDER),
        "feature_order": ",".join(config.FEATURE_ORDER),
        "n_mfcc": config.N_MFCC,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "n_mels": config.N_MELS,
        "resample_type": config.RESAMPLE_TYPE,
        "short_audio_policy": "zero_pad",
        "long_audio_policy": "reject",
        "std_ddof": 0,
    }
    for key, value in expected.items():
        if feature.get(key) != value:
            raise ModelLoadError(f"Metadata feature_extractor.{key} does not match the shared feature code")
    return metadata


class AudioPredictor:
    """Load artifacts once and reuse this object for consecutive WAV requests.

    Only load trusted team artifacts: joblib uses pickle internally. The
    checksum checks consistency with metadata, not the publisher's identity.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        *,
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    ) -> None:
        if isinstance(max_audio_bytes, bool) or not isinstance(max_audio_bytes, int) or max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be a positive integer")
        self.max_audio_bytes = max_audio_bytes
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = _read_metadata(self.metadata_path)
        try:
            with self.model_path.open("rb") as artifact:
                digest = hashlib.file_digest(artifact, "sha256").hexdigest()
                if digest != self.metadata["model_sha256"].lower():
                    raise ModelLoadError("Model checksum does not match metadata; use the matching A1 artifact pair")
                artifact.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    self.model = joblib.load(artifact)
        except ModelLoadError:
            raise
        except InconsistentVersionWarning as exc:
            raise ModelLoadError("scikit-learn version differs from the saved model; install A1's recorded version") from exc
        except Exception as exc:
            raise ModelLoadError(f"Could not load model: {self.model_path}: {exc}") from exc
        if not isinstance(self.model, RandomForestClassifier):
            raise ModelLoadError("Loaded artifact is not a RandomForestClassifier")
        try:
            check_is_fitted(self.model)
        except Exception as exc:
            raise ModelLoadError("Loaded Random Forest is not fitted") from exc
        if getattr(self.model, "n_features_in_", None) != len(config.FEATURE_ORDER):
            raise ModelLoadError("Loaded model must expect 40 MFCC features")
        if getattr(self.model, "n_outputs_", None) != 1:
            raise ModelLoadError("Loaded model must be a single-output classifier")
        self.classes = tuple(str(c) for c in self.model.classes_)
        if list(self.classes) != self.metadata["class_order"]:
            raise ModelLoadError("Model class order differs from metadata class_order")
        self.model_version = self.metadata["model_version"]

    def _validate_wav(self, audio_path: str | Path) -> Path:
        path = Path(audio_path)
        try:
            if not path.is_file():
                raise AudioProcessingError(f"Audio file does not exist or is not a regular file: {path}")
            size = path.stat().st_size
            if size == 0:
                raise AudioProcessingError("Audio file is empty")
            if size > self.max_audio_bytes:
                raise AudioProcessingError(f"Audio file exceeds the {self.max_audio_bytes}-byte limit")
            info = sf.info(str(path))
        except AudioProcessingError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise AudioProcessingError(f"Could not read WAV header: {path}") from exc
        if info.format not in WAV_FORMATS:
            raise AudioProcessingError(f"Expected WAV audio; decoded container is {info.format}")
        if info.channels not in (1, 2):
            raise AudioProcessingError("WAV input must have one or two channels")
        if info.samplerate <= 0:
            raise AudioProcessingError("WAV sample rate must be positive")
        if info.frames <= 0:
            raise AudioProcessingError("WAV contains no audio samples")
        if info.frames > info.samplerate * config.WINDOW_SECONDS:
            raise AudioProcessingError("Audio is longer than 5 seconds; split it into windows before prediction")
        return path

    def predict_file(self, audio_path: str | Path) -> PredictionResult:
        """Score one WAV; processing_ms includes validation, features and scoring.

        Capture, transport and initial model loading are outside this timer.
        model_score is uncalibrated. Alert policy belongs to the backend.
        """
        start = time.perf_counter()
        path = self._validate_wav(audio_path)
        try:
            features = extract_features_from_file(path)
        except AudioProcessingError as exc:
            # librosa can reject NaN/Inf before A1's prepare_audio checks them.
            # Keep A1's extractor unchanged and expose the specific input error.
            if isinstance(exc.__cause__, ParameterError) and "not finite" in str(exc.__cause__).lower():
                raise AudioProcessingError("Audio contains non-finite sample values") from exc
            raise
        if features.shape != (40,) or not np.isfinite(features).all():
            raise AudioProcessingError("Shared extractor must return 40 finite features")
        try:
            probabilities = np.asarray(self.model.predict_proba(features.reshape(1, -1)), dtype=np.float64)
        except Exception as exc:
            raise PredictionError(f"Model inference failed: {exc}") from exc
        if (
            probabilities.shape != (1, len(self.classes))
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-6)
        ):
            raise PredictionError("Model returned invalid class scores")
        scores = probabilities[0]
        best = int(np.argmax(scores))
        return {
            "predicted_class": self.classes[best],
            "model_score": float(scores[best]),
            "scores": {label: float(score) for label, score in zip(self.classes, scores)},
            "model_version": self.model_version,
            "processing_ms": float((time.perf_counter() - start) * 1000),
        }


def predict_file(
    audio_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> PredictionResult:
    """One-shot convenience function; services should reuse AudioPredictor instead."""
    return AudioPredictor(model_path, metadata_path).predict_file(audio_path)
