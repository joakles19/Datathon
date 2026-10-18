import json
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from hearhere_audio.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLDS_PATH,
    HearHereClassifier,
    ModelArtifactError,
    ModelPredictionError,
)


def wav_bytes() -> bytes:
    sample_rate = 16_000
    samples = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)
    audio = BytesIO()
    sf.write(audio, samples.astype(np.float32), sample_rate, format="WAV")
    return audio.getvalue()


def test_classifier_returns_all_class_scores_and_model_version():
    classifier = HearHereClassifier()
    prediction = classifier.predict_bytes(wav_bytes())

    assert set(prediction["scores"]) == set(classifier.model.classes_)
    assert prediction["label"] == max(prediction["scores"], key=prediction["scores"].get)
    assert prediction["score"] == prediction["scores"][prediction["label"]]
    assert prediction["model_version"] == classifier.metadata["model_version"]
    assert "keyboard_typing" not in prediction["scores"]
    assert len(prediction["scores"]) == 10
    assert sum(prediction["scores"].values()) == pytest.approx(1.0)


def test_classifier_resolves_metadata_model_file_for_string_default_path():
    classifier = HearHereClassifier(model_path=str(DEFAULT_MODEL_PATH))

    assert classifier.model_path.name == "hearhere_panns_lr.joblib"
    assert classifier.metadata["model_file"] == classifier.model_path.name


def test_classifier_rejects_invalid_model_scores(monkeypatch):
    classifier = HearHereClassifier()
    monkeypatch.setattr(classifier.model, "predict_proba", lambda _: np.full((1, 10), np.nan))

    with pytest.raises(ModelPredictionError, match="invalid class scores"):
        classifier.predict_bytes(wav_bytes())


def test_classifier_rejects_unreadable_audio():
    with pytest.raises(ValueError, match="Could not decode uploaded audio bytes"):
        HearHereClassifier().predict_bytes(b"not a WAV")


def test_classifier_checks_hash_before_loading_model(tmp_path):
    metadata = json.loads(
        (DEFAULT_MODEL_PATH.parent / "model_metadata.json").read_text(encoding="utf-8")
    )
    metadata["model_sha256"] = "0" * 64
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"invalid serialized model")

    with pytest.raises(ModelArtifactError, match="hash does not match"):
        HearHereClassifier(
            model_path=model_path,
            metadata_path=metadata_path,
            thresholds_path=DEFAULT_THRESHOLDS_PATH,
        )


def test_classifier_rejects_thresholds_from_another_model(tmp_path):
    thresholds = json.loads(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    thresholds["model_version"] = "different-model"
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="do not match"):
        HearHereClassifier(thresholds_path=thresholds_path)


def test_classifier_rejects_missing_disabled_alert_class(tmp_path):
    thresholds = json.loads(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    thresholds["disabled_alert_classes"] = []
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="Disabled alert classes"):
        HearHereClassifier(thresholds_path=thresholds_path)


def test_classifier_rejects_threshold_without_qualifying_precision(tmp_path):
    thresholds = json.loads(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    label = thresholds["recommended_alert_classes"][0]
    thresholds["validation_metrics"][label]["precision"] = 0.84
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="inconsistent"):
        HearHereClassifier(thresholds_path=thresholds_path)
