# A2 handoff to B, D and A1

## Current status — 17 October 2026

The v2 fold-1 sample reference is recorded in `tests/sample_reference.json`, including the model/WAV hashes, 40-value feature vector and ten class scores. It is an integration reference, not a held-out accuracy measurement.

The `main` branch now includes the real-inference FastAPI service and React client. **The API contract in `classify_api.md` supersedes the earlier placeholder-replacement instructions below for team integration.** Use the browser client, `ClassifyClient`, or `scripts/classify_api.py` to consume `POST /classify`. Preserve `predicted_label`, policy-selected `label`, `confidence = scores[label]`, all scores and server actions. The backend owns alert policy and event handling; the browser and API client do not retrain the model or select thresholds.

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
predictor = AudioPredictor("models/hearhere_rf.joblib", "models/model_metadata.json")

def predict(audio_path):
    return predict_label_score(audio_path, predictor)
```

Integrate this function body with B's actual signature and existing request handler. This snippet is not a complete server. Audio capture, notifications and alert rules remain outside the predictor.

## A1: compare the same WAV

Use `examples/1-31482-A-42.wav`. SHA256:

```text
e4c640a001c2abe808389c90b90ba7084fb1922d58235e3ec95c18c4afb4ef5c
```

The v2 reference is `tests/sample_reference.json`: all 40 feature values, all ten scores, model version, model checksum and WAV checksum. It is not held-out evaluation.

Run the same shared extractor and model on A1's environment and compare features with `rtol=1e-5, atol=1e-4`. The model-score comparison target for the recorded sample is absolute tolerance `1e-12`. If feature/platform differences lead to a tree decision changing, investigate the dependencies and preprocessing before changing the reference or tolerance.

Expected A2 sample prediction: `siren`, model score `0.8566666666666667`.

The latest `ai_david` branch now includes the training manifest, validation metrics, threshold artifact and updated metadata. Its predictor retains the same model checksum. The published A2 branch inherits these A1 files unchanged. The path for final-test metrics in metadata does not prove Fold 5 has been run.

## Timing and evidence

- Model loading happens once outside `processing_ms`.
- Header validation, feature extraction and scoring are inside `processing_ms`.
- Measure microphone collection and transport separately for sound-onset-to-visible-alert latency.
- Record cold startup and warm inference separately. One Windows sample's timing is not a LinuxONE benchmark.
- Save/reload and invalid-input checks are in the test suite. Test fixtures do not establish real-world sound-detection accuracy.
- Record the actual LinuxONE host/runtime and commands D runs before making deployment claims.

## Acceptance checklist

- [x] A1 confirms sample features and scores against the supplied reference (user-relayed report, 17 October 2026).
- [x] B/A1 source now has real inference at `POST /classify` on `ai_david` (source inspected).
- [ ] Obtain B/D's running service URL and test an upload against that deployment.
- [ ] D installs supported packages and loads the saved artifact on LinuxONE.
- [ ] D runs the same WAV and records any differences from the reference.
- [ ] E measures realistic noise conditions, false alerts and capture-to-alert latency.
- [ ] A1/E perform final Fold 5 evaluation at the agreed model-freeze point.
