"""Adapt the full A2 result to B's existing (label, score) interface."""

from pathlib import Path

from .predict import AudioPredictor


def predict_label_score(audio_path: str | Path, predictor: AudioPredictor) -> tuple[str, float]:
    """Pass a predictor created during backend startup; input errors propagate."""
    result = predictor.predict_file(audio_path)
    return result["predicted_class"], result["model_score"]
