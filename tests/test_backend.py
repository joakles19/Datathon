from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backends import main as backend
from hearhere_audio.features import AudioProcessingError


def wav_bytes() -> bytes:
    sample_rate = 16_000
    samples = np.zeros(sample_rate, dtype=np.float32)
    audio = BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV")
    return audio.getvalue()


class FakeDatabase:
    def __init__(self):
        self.calls = []

    def execute(self, query, values=()):
        self.calls.append((query, values))
        return SimpleNamespace(lastrowid=17)

    def commit(self):
        pass


def test_classify_uses_important_class_score_even_if_top_class_is_ordinary(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(backend, "db", database)
    monkeypatch.setattr(backend, "last_alert", {})
    monkeypatch.setattr(backend, "snoozed_until", {})
    scores = {
        "door_wood_knock": 0.08,
        "engine": 0.04,
        "glass_breaking": 0.06,
        "keyboard_typing": 0.05,
        "rain": 0.40,
        "siren": 0.31,
        "vacuum_cleaner": 0.03,
        "wind": 0.03,
    }
    monkeypatch.setattr(
        backend,
        "predict",
        lambda _: {
            "label": "rain",
            "score": 0.40,
            "scores": scores,
            "model_version": "test-model",
        },
    )

    response = TestClient(backend.app).post(
        "/classify", files={"file": ("sample.wav", wav_bytes(), "audio/wav")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "siren"
    assert payload["predicted_label"] == "rain"
    assert payload["confidence"] == 0.31
    assert payload["action"] == "alert"
    assert payload["scores"] == scores
    assert payload["model_version"] == "test-model"
    assert payload["event_id"] == 17
    assert database.calls[0][1][1:4] == ("siren", 0.31, "alert")


def test_classify_returns_422_for_invalid_audio(monkeypatch):
    monkeypatch.setattr(
        backend,
        "predict",
        lambda _: (_ for _ in ()).throw(AudioProcessingError("Could not decode uploaded audio bytes")),
    )

    response = TestClient(backend.app).post(
        "/classify", files={"file": ("broken.wav", b"broken", "audio/wav")}
    )

    assert response.status_code == 422
    assert "Could not decode" in response.json()["detail"]


def test_ordinary_top_class_does_not_trigger_an_alert(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(backend, "db", database)
    scores = {
        "door_wood_knock": 0.02,
        "engine": 0.03,
        "glass_breaking": 0.02,
        "keyboard_typing": 0.03,
        "rain": 0.80,
        "siren": 0.04,
        "vacuum_cleaner": 0.03,
        "wind": 0.03,
    }
    monkeypatch.setattr(
        backend,
        "predict",
        lambda _: {
            "label": "rain",
            "score": 0.80,
            "scores": scores,
            "model_version": "test-model",
        },
    )

    response = TestClient(backend.app).post(
        "/classify", files={"file": ("sample.wav", wav_bytes(), "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json()["action"] == "ignore"
    assert response.json()["label"] == "rain"
    assert database.calls == []


def test_classify_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(
        backend,
        "predict",
        lambda _: (_ for _ in ()).throw(AssertionError("predict must not run")),
    )

    response = TestClient(backend.app).post(
        "/classify",
        files={"file": ("large.wav", b"x" * (backend.MAX_UPLOAD_BYTES + 1), "audio/wav")},
    )

    assert response.status_code == 413


def test_classify_runs_loaded_model_on_uploaded_wav(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(backend, "db", database)
    monkeypatch.setattr(backend, "last_alert", {})

    response = TestClient(backend.app).post(
        "/classify", files={"file": ("sample.wav", wav_bytes(), "audio/wav")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "hearhere-rf-v1"
    assert set(payload["scores"]) == set(backend.classifier.model.classes_)
    assert sum(payload["scores"].values()) == pytest.approx(1.0)
