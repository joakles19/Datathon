from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backends import main as backend
from hearhere_audio.features import AudioProcessingError
from hearhere_audio.config import ALERT_CLASSES


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


def test_health_reports_loaded_model():
    response = TestClient(backend.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": backend.classifier.metadata["model_version"],
    }


def test_classify_uses_important_class_score_even_if_top_class_is_ordinary(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(backend, "db", database)
    monkeypatch.setattr(backend, "last_alert", {})
    monkeypatch.setattr(backend, "snoozed_until", {})
    monkeypatch.setitem(backend.PREFS["siren"], "enabled", True)
    monkeypatch.setitem(backend.PREFS["siren"], "threshold", 0.3)
    scores = {
        "door_wood_knock": 0.08,
        "car_horn": 0.01,
        "clock_alarm": 0.01,
        "crying_baby": 0.01,
        "engine": 0.04,
        "glass_breaking": 0.06,
        "rain": 0.42,
        "siren": 0.31,
        "vacuum_cleaner": 0.03,
        "wind": 0.03,
    }
    monkeypatch.setattr(
        backend.classifier,
        "predict_bytes",
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
        backend.classifier,
        "predict_bytes",
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
        "car_horn": 0.01,
        "clock_alarm": 0.01,
        "crying_baby": 0.01,
        "engine": 0.03,
        "glass_breaking": 0.02,
        "rain": 0.80,
        "siren": 0.04,
        "vacuum_cleaner": 0.03,
        "wind": 0.03,
    }
    monkeypatch.setattr(
        backend.classifier,
        "predict_bytes",
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


@pytest.mark.parametrize("alert_class", ALERT_CLASSES)
def test_each_configured_alert_class_can_trigger_alert(monkeypatch, alert_class):
    database = FakeDatabase()
    monkeypatch.setattr(backend, "db", database)
    monkeypatch.setattr(backend, "last_alert", {})
    monkeypatch.setitem(backend.PREFS[alert_class], "enabled", True)
    monkeypatch.setitem(backend.PREFS[alert_class], "threshold", 0.7)
    threshold = backend.PREFS[alert_class]["threshold"]
    scores = {label: 0.01 for label in backend.classifier.model.classes_}
    scores[alert_class] = min(0.99, threshold + 0.05)
    monkeypatch.setattr(
        backend.classifier,
        "predict_bytes",
        lambda _: {
            "label": alert_class,
            "score": scores[alert_class],
            "scores": scores,
            "model_version": "test-model",
        },
    )

    response = TestClient(backend.app).post(
        "/classify", files={"file": ("sample.wav", wav_bytes(), "audio/wav")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == alert_class
    assert payload["action"] == "alert"
    assert payload["confidence"] == scores[alert_class]
    assert database.calls[0][1][1] == alert_class


def test_fold4_nonqualifying_alert_is_disabled_but_remains_a_model_output(monkeypatch):
    disabled = backend.classifier.thresholds["disabled_alert_classes"]
    if not disabled:
        pytest.skip("selected model has no disabled alert classes")
    label = disabled[0]
    assert label in backend.classifier.model.classes_
    assert backend.PREFS[label]["enabled"] is False


def test_classify_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(
        backend.classifier,
        "predict_bytes",
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
    assert payload["model_version"] == backend.classifier.metadata["model_version"]
    assert set(payload["scores"]) == set(backend.classifier.model.classes_)
    assert sum(payload["scores"].values()) == pytest.approx(1.0)


def test_live_monitor_confirms_same_alert_class_in_two_of_three_windows(monkeypatch):
    database = FakeDatabase()
    score_state = {"class": "door_wood_knock"}
    monkeypatch.setattr(backend, "db", database)
    monkeypatch.setattr(backend, "last_alert", {})
    monkeypatch.setattr(backend, "monitor_windows", backend.OrderedDict())

    def predict(_):
        scores = {label: 0.01 for label in backend.classifier.model.classes_}
        scores[score_state["class"]] = 0.8
        return {"label": score_state["class"], "score": 0.8, "scores": scores, "model_version": "test-model"}

    monkeypatch.setattr(backend.classifier, "predict_bytes", predict)
    client = TestClient(backend.app)

    first = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-a", "window_index": "0"},
    )
    other_stream = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-b", "window_index": "0"},
    )
    confirmed = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-a", "window_index": "1"},
    )

    assert first.status_code == 200
    assert first.json()["confirmation"] == "pending"
    assert first.json()["action"] == "ignore"
    assert first.json()["event_id"] is None
    assert other_stream.json()["confirmation"] == "pending"
    assert confirmed.status_code == 200
    assert confirmed.json()["confirmation"] == "confirmed"
    assert confirmed.json()["action"] == "alert"
    assert len(database.calls) == 1


def test_live_monitor_requires_matching_class_and_increasing_window_indices(monkeypatch):
    score_state = {"class": "door_wood_knock"}
    monkeypatch.setattr(backend, "monitor_windows", backend.OrderedDict())
    for label in ("door_wood_knock", "siren"):
        monkeypatch.setitem(backend.PREFS[label], "enabled", True)
        monkeypatch.setitem(backend.PREFS[label], "threshold", 0.7)
    monkeypatch.setattr(
        backend.classifier,
        "predict_bytes",
        lambda _: {
            "label": score_state["class"],
            "score": 0.8,
            "scores": {
                **{label: 0.01 for label in backend.classifier.model.classes_},
                score_state["class"]: 0.8,
            },
            "model_version": "test-model",
        },
    )
    client = TestClient(backend.app)

    first = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-c", "window_index": "0"},
    )
    score_state["class"] = "siren"
    different_class = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-c", "window_index": "1"},
    )
    repeated_index = client.post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-c", "window_index": "1"},
    )

    assert first.json()["confirmation"] == "pending"
    assert different_class.json()["confirmation"] == "pending"
    assert repeated_index.status_code == 409


def test_monitor_metadata_must_be_provided_as_a_pair():
    response = TestClient(backend.app).post(
        "/classify",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
        data={"stream_id": "stream-d"},
    )

    assert response.status_code == 400
