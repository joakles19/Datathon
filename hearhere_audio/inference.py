"""Load the trained classifier and produce scores using the shared features."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ALERT_CLASSES, CLASSES, FEATURE_ORDER
from .features import AudioProcessingError, extract_features_from_bytes
from .panns import (
    EMBEDDING_DIM,
    DEFAULT_CACHE_PATH,
    PannsEmbeddingExtractor,
    PannsError,
    extract_embedding_from_bytes,
)
from .portable_model import PortableModelError, load_portable_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hearhere_rf.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "thresholds.json"


class ModelArtifactError(RuntimeError):
    """Raised when model metadata and inference artifacts do not match."""


class ModelPredictionError(RuntimeError):
    """Raised when the loaded estimator returns an invalid score vector."""


class HearHereClassifier:
    """A loaded model with its matching validation-selected thresholds."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.thresholds_path = Path(thresholds_path)
        self.metadata = self._read_json(self.metadata_path)
        self.thresholds = self._read_json(self.thresholds_path)
        metadata_model_path = self.metadata.get("model_file")
        if self.model_path == DEFAULT_MODEL_PATH and isinstance(metadata_model_path, str):
            self.model_path = self.metadata_path.parent / metadata_model_path
        if not self.model_path.is_file():
            raise ModelArtifactError(f"Trained model not found: {self.model_path}")
        model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if self.metadata.get("model_sha256") != model_hash:
            raise ModelArtifactError("Model file hash does not match model_metadata.json")
        self.model = self._load_model()
        self._validate_artifacts()
        self.panns_extractor = None
        if self.metadata.get("feature_extractor", {}).get("type", "mfcc") == "panns_cnn14":
            checkpoint_path = self.metadata.get("panns_checkpoint_path") or DEFAULT_CACHE_PATH
            try:
                self.panns_extractor = PannsEmbeddingExtractor(checkpoint_path)
            except PannsError as exc:
                raise ModelArtifactError(str(exc)) from exc

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Could not read model artifact: {path}") from exc
        if not isinstance(value, dict):
            raise ModelArtifactError(f"Model artifact must contain a JSON object: {path}")
        return value

    def _load_model(self):
        portable_name = self.metadata.get("portable_model_file")
        portable_path = self.metadata_path.parent / portable_name if isinstance(portable_name, str) else None
        if platform.machine() == "s390x" and portable_path is not None and portable_path.is_file():
            portable_hash = hashlib.sha256(portable_path.read_bytes()).hexdigest()
            if portable_hash != self.metadata.get("portable_model_sha256"):
                raise ModelArtifactError("Portable model hash does not match model_metadata.json")
            try:
                return load_portable_model(portable_path)
            except PortableModelError as exc:
                raise ModelArtifactError(str(exc)) from exc
        try:
            return joblib.load(self.model_path)
        except Exception as exc:
            if portable_path is None or not portable_path.is_file():
                raise ModelArtifactError(f"Could not load trained model: {self.model_path}") from exc
            try:
                return load_portable_model(portable_path)
            except PortableModelError as portable_exc:
                raise ModelArtifactError(f"Could not load trained model: {self.model_path}") from portable_exc

    def _validate_artifacts(self) -> None:
        if self.metadata.get("artifact_status") != "trained_fold4_validated":
            raise ModelArtifactError("Model metadata does not describe a validated model")
        if self.thresholds.get("artifact_status") != "fold4_thresholds_selected":
            raise ModelArtifactError("Fold 4 alert thresholds are not available")
        if self.thresholds.get("model_version") != self.metadata.get("model_version"):
            raise ModelArtifactError("Thresholds do not match the loaded model version")
        feature_extractor = self.metadata.get("feature_extractor", {})
        feature_type = feature_extractor.get("type", "mfcc")
        expected_features = len(FEATURE_ORDER) if feature_type == "mfcc" else (
            EMBEDDING_DIM if feature_type == "panns_cnn14" else None
        )
        if expected_features is None:
            raise ModelArtifactError("Model metadata has an unsupported feature extractor")
        if feature_extractor.get("feature_count") != expected_features:
            raise ModelArtifactError("Model metadata feature count does not match the extractor")
        if feature_type == "mfcc" and feature_extractor.get("feature_order") != ",".join(FEATURE_ORDER):
            raise ModelArtifactError("Model feature order does not match the extractor")
        if feature_type == "panns_cnn14" and feature_extractor.get("checkpoint_md5") != "362fc5ff18f1d6ad2f6d464b45893f2c":
            raise ModelArtifactError("PANNs checkpoint does not match the approved upstream release")
        if tuple(self.model.classes_) != tuple(self.metadata.get("class_order", ())):
            raise ModelArtifactError("Model classes do not match model_metadata.json")
        if set(self.model.classes_) != set(CLASSES):
            raise ModelArtifactError("Model must contain all configured sound classes")
        if getattr(self.model, "n_features_in_", None) != expected_features:
            raise ModelArtifactError("Model feature count does not match the extractor")
        configured_scores = self.thresholds.get("minimum_scores", {})
        recommended_classes = tuple(self.thresholds.get("recommended_alert_classes", ()))
        if not set(recommended_classes).issubset(ALERT_CLASSES):
            raise ModelArtifactError("Threshold artifact lists an unknown alert class")
        if set(configured_scores) != set(recommended_classes):
            raise ModelArtifactError("Threshold scores must match qualifying alert classes")
        if set(self.thresholds.get("disabled_alert_classes", ())) != set(ALERT_CLASSES) - set(recommended_classes):
            raise ModelArtifactError("Disabled alert classes do not match the threshold recommendations")
        if not all(
            isinstance(score, (int, float)) and np.isfinite(score) and 0 <= score <= 1
            for score in configured_scores.values()
        ):
            raise ModelArtifactError("Alert thresholds must be numeric scores between 0 and 1")
        if self.thresholds.get("minimum_precision") != 0.85:
            raise ModelArtifactError("Threshold artifact has an unexpected minimum precision")
        validation_metrics = self.thresholds.get("validation_metrics", {})
        for label, threshold in configured_scores.items():
            result = validation_metrics.get(label, {})
            if (
                result.get("threshold") != threshold
                or result.get("precision", 0) < 0.85
                or result.get("recall", 0) < 0
                or result.get("support", 0) <= 0
            ):
                raise ModelArtifactError(f"Threshold metrics are inconsistent for {label}")

    def predict_bytes(self, audio_bytes: bytes) -> dict[str, Any]:
        """Return the top class and scores for all configured model classes."""
        feature_type = self.metadata.get("feature_extractor", {}).get("type", "mfcc")
        if feature_type == "panns_cnn14":
            try:
                features = extract_embedding_from_bytes(
                    audio_bytes, self.panns_extractor
                ).reshape(1, -1)
            except PannsError as exc:
                raise AudioProcessingError(str(exc)) from exc
        else:
            features = extract_features_from_bytes(audio_bytes).reshape(1, -1)
        try:
            probabilities = np.asarray(self.model.predict_proba(features), dtype=np.float64)
        except Exception as exc:
            raise ModelPredictionError(f"Model inference failed: {exc}") from exc
        if (
            probabilities.shape != (1, len(self.model.classes_))
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-6)
        ):
            raise ModelPredictionError("Model returned invalid class scores")
        scores = probabilities[0]
        score_by_class = {
            label: float(score)
            for label, score in zip(self.model.classes_, scores)
        }
        label = max(score_by_class, key=score_by_class.get)
        return {
            "label": label,
            "score": score_by_class[label],
            "scores": score_by_class,
            "model_version": self.metadata["model_version"],
        }
