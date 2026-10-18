# A2 handoff to B, D and A1

## Current status — 18 October 2026

The v2 fold-1 sample reference is recorded in `tests/sample_reference.json`, including the model/WAV hashes, 2048-value PANNs embedding and ten class scores. It is an integration reference, not a held-out accuracy measurement.

The `dev` branch includes the real-inference FastAPI service, React client and Fold 4-selected PANNs model. **The API contract in `classify_api.md` supersedes the earlier placeholder-replacement instructions below for team integration.** Use the browser client, `ClassifyClient`, or `scripts/classify_api.py` to consume `POST /classify`. Preserve `predicted_label`, policy-selected `label`, `confidence = scores[label]`, all scores and server actions. The backend owns alert policy and event handling; the browser and API client do not retrain the model or select thresholds.

## Files and ownership

- A1 owns `config.py`, `features.py`, the fitted model and its metadata. The v2 model is a ten-class model and excludes `keyboard_typing`.
- A2 adds `predict.py`, `backend_adapter.py`, `cli.py`, the CLI entry points, tests, requirements updates and documentation.
- B owns the HTTP handler and alert policy in `backends/main.py`; A2's API client remains a separate consumer for service-to-service use.
- D owns deployment and environment validation on LinuxONE.

## B: startup and requests

1. Load one `AudioPredictor` during backend startup. Missing or mismatched artifacts should stop startup, not silently use a fixed prediction.
2. Make one warm-up prediction with an agreed WAV; mark the service ready after it succeeds. Warm-up output should not generate an alert/event.
3. For each request, save the received WAV to a distinct temporary file whose lifetime includes the synchronous prediction call. Enforce upload limits before buffering a full request.
4. Call `predict_label_score(path, predictor)` to preserve `(label, score)`, or use the full result if the endpoint evolves.
5. Catch `AudioProcessingError` for invalid input and return a client error without a success score. `PredictionError` is a service inference failure. Decide HTTP status codes in B's application.
6. Do not share or overwrite one temporary filename across concurrent requests. Keep cleanup in a finally block after prediction completes.
7. The classifier call is blocking CPU work. In an async server, use its supported worker/thread facility rather than blocking the event loop. Limit concurrent inference to a level D has measured; the saved model's `n_jobs=-1` setting can otherwise increase CPU use.
8. Keep model/configuration immutable during serving. Recreate the predictor during a controlled model update.

```python
from hearhere_audio.features import AudioProcessingError
from hearhere_audio.predict import AudioPredictor, ModelLoadError, PredictionError
from hearhere_audio.backend_adapter import predict_label_score

# Create in the backend's startup lifecycle, with explicit artifact paths.
predictor = AudioPredictor(metadata_path="models/model_metadata.json")

def predict(audio_path):
    return predict_label_score(audio_path, predictor)
```

Integrate this function body with B's actual signature and existing request handler. This snippet is not a complete server. Audio capture, notifications and alert rules remain outside the predictor.

## A1: compare the same WAV

Use `examples/1-31482-A-42.wav`. SHA256:

```text
e4c640a001c2abe808389c90b90ba7084fb1922d58235e3ec95c18c4afb4ef5c
```

The v2 reference is `tests/sample_reference.json`: all 2048 PANNs embedding values, all ten scores, model version, model checksum and WAV checksum. It is not held-out evaluation.

Run the same shared preprocessing, PANNs embedding and model on A1's environment and compare embeddings with `rtol=1e-5, atol=1e-4`. The model-score comparison target for the recorded sample is absolute tolerance `1e-12`. If platform differences move embeddings or scores outside tolerance, investigate dependencies and preprocessing before changing the reference or tolerance.

Expected selected-model sample prediction: `siren`, uncalibrated model score `0.9980500936508179`.

The `dev` branch includes the training manifest, Fold 4 comparison, threshold artifact and selected-model metadata. Fold 5 was evaluated once after the model and thresholds were locked; see `reports/test_metrics.json`. Do not rerun Fold 5 or tune from it. LinuxONE runtime and inference remain unverified deployment gates.

## Timing and evidence

- Model loading happens once outside `processing_ms`.
- Header validation, feature extraction and scoring are inside `processing_ms`.
- Measure microphone collection and transport separately for sound-onset-to-visible-alert latency.
- Record cold startup and warm inference separately. One Windows sample's timing is not a LinuxONE benchmark.
- Save/reload and invalid-input checks are in the test suite. Test fixtures do not establish real-world sound-detection accuracy.
- Record the actual LinuxONE host/runtime and commands D runs before making deployment claims.

## Acceptance checklist

- [x] The selected PANNs embedding and model scores match the committed local sample reference in the Python test suite.
- [x] B/A1 source now has real inference at `POST /classify` on `ai_david` (source inspected).
- [ ] Obtain B/D's running service URL and test an upload against that deployment.
- [ ] D installs supported packages and loads the saved artifact on LinuxONE.
- [ ] D runs the same WAV and records any differences from the reference.
- [ ] E measures realistic noise conditions, false alerts and capture-to-alert latency.
- [x] Fold 5 final evaluation was run once after model freeze and recorded in `reports/test_metrics.json`; do not rerun or tune from it.
