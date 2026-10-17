"""Load the trained classifier and produce scores using the shared features."""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ALERT_CLASSES, CLASSES, FEATURE_ORDER
from .features import extract_features_from_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hearhere_rf.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "thresholds.json"


class ModelArtifactError(RuntimeError):
    """Raised when model metadata and inference artifacts do not match."""


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
        if not self.model_path.is_file():
            raise ModelArtifactError(f"Trained model not found: {self.model_path}")
        model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if self.metadata.get("model_sha256") != model_hash:
            raise ModelArtifactError("Model file hash does not match model_metadata.json")
        self.model = self._load_model()
        self._validate_artifacts()

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
        try:
            return joblib.load(self.model_path)
        except Exception as exc:
            raise ModelArtifactError(f"Could not load trained model: {self.model_path}") from exc

    def _validate_artifacts(self) -> None:
        if self.metadata.get("artifact_status") != "trained_fold4_validated":
            raise ModelArtifactError("Model metadata does not describe a validated model")
        if self.thresholds.get("artifact_status") != "fold4_thresholds_selected":
            raise ModelArtifactError("Fold 4 alert thresholds are not available")
        if self.thresholds.get("model_version") != self.metadata.get("model_version"):
            raise ModelArtifactError("Thresholds do not match the loaded model version")
        feature_extractor = self.metadata.get("feature_extractor", {})
        if feature_extractor.get("feature_count") != len(FEATURE_ORDER):
            raise ModelArtifactError("Model metadata feature count does not match the extractor")
        if feature_extractor.get("feature_order") != ",".join(FEATURE_ORDER):
            raise ModelArtifactError("Model feature order does not match the extractor")
        if tuple(self.model.classes_) != tuple(self.metadata.get("class_order", ())):
            raise ModelArtifactError("Model classes do not match model_metadata.json")
        if set(self.model.classes_) != set(CLASSES):
            raise ModelArtifactError("Model must contain all configured sound classes")
        if getattr(self.model, "n_features_in_", None) != len(FEATURE_ORDER):
            raise ModelArtifactError("Model feature count does not match the extractor")
        if tuple(self.thresholds.get("recommended_alert_classes", ())) != ALERT_CLASSES:
            raise ModelArtifactError("Threshold artifact does not list the expected alert classes")
        configured_scores = self.thresholds.get("minimum_scores", {})
        if set(configured_scores) != set(ALERT_CLASSES):
            raise ModelArtifactError("Threshold artifact must define every important sound")
        if not all(
            isinstance(score, (int, float)) and 0 <= score <= 1
            for score in configured_scores.values()
        ):
            raise ModelArtifactError("Alert thresholds must be numeric scores between 0 and 1")
        if self.thresholds.get("minimum_recall") != 0.75:
            raise ModelArtifactError("Threshold artifact has an unexpected minimum recall")
        validation_metrics = self.thresholds.get("validation_metrics", {})
        for label, threshold in configured_scores.items():
            result = validation_metrics.get(label, {})
            if result.get("threshold") != threshold or result.get("recall", 0) < 0.75:
                raise ModelArtifactError(f"Threshold metrics are inconsistent for {label}")

    def predict_bytes(self, audio_bytes: bytes) -> dict[str, Any]:
        """Return the top class and scores for all configured model classes."""
        features = extract_features_from_bytes(audio_bytes).reshape(1, -1)
        scores = self.model.predict_proba(features)[0]
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
