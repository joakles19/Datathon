# HearHere — A2 inference delivery

Run the selected ten-class HearHere model on WAV recordings and return class scores. The active candidate uses frozen PANNs CNN14 embeddings with a standardized Logistic Regression head; the existing Random Forest remains available as the comparison baseline. A2 supplies a persistent predictor, a one-shot function, a JSON CLI and a tuple adapter for B.

**Current integration status:** The `dev` branch includes the frontend, backend and Fold 4-selected PANNs model. The browser sends WAV windows to `POST /classify`; the Python API consumer below remains available for command-line integration and diagnostics. See `docs/classify_api.md` for the current response contract.

## Consume the shared backend

With a reachable backend base URL:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

The URL above is an example. Upload field: `file`. The client preserves the server response, including `predicted_label` (highest-scoring class), `label` (selected alert candidate), `confidence` (`scores[label]`), `action`, `priority`, `message`, `event_id`, all ten v2 `scores` and `model_version`. It does not load a local model or choose thresholds. The server owns action policy and event handling. Source being pushed to GitHub does not itself start a shared backend service.

## Setup on Windows

Use Python **3.12** for this handoff. A1 recorded 3.12.3; A2 was tested with 3.12.14. Confirm your interpreter before creating the environment. Do not use an existing global Python 3.14 environment for this documented setup.

```powershell
# Run inside the repository. Replace the path with your Python 3.12 interpreter.
& 'C:\path\to\Python312\python.exe' scripts/setup_environment.py
```

`requirements.txt` records the pinned CPU PyTorch/PANNs runtime and the shared audio stack. `docs/a2-tested-environment.txt` records the complete Windows review environment; it is evidence, not a guarantee that all packages have LinuxONE wheels.

Keep these A1 artifacts together:

```text
hearhere_audio/config.py
hearhere_audio/features.py
hearhere_audio/panns.py
models/hearhere_rf.joblib
models/hearhere_panns_lr.joblib
models/model_metadata.json
```

The predictor checks the selected model's SHA256 against its metadata, classes/order, estimator type, fitted state and feature settings. PANNs inference also requires the verified checkpoint in the local cache created by setup. Only load trusted team model artifacts. A checksum verifies a matching pair; it is not a signature. A different scikit-learn artifact version raises a clear loading error.

## Classify a WAV

```powershell
.\.venv\Scripts\python.exe -m hearhere_audio 'examples/1-31482-A-42.wav' --pretty
```

The supplied fold-1 sample is included under `examples/` for integration checks with dataset attribution. It is not a final-test performance example.

Alternative entry point:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py 'example.wav' --metadata models/model_metadata.json --pretty
```

Successful stdout is JSON with exactly:

```json
{
  "predicted_class": "siren",
  "model_score": 0.9980500936508179,
  "scores": {
    "car_horn": 0.0005899363313801587,
    "clock_alarm": 0.00022119721688795835,
    "crying_baby": 0.0005849687149748206,
    "door_wood_knock": 0.0000757737725507468,
    "engine": 0.0001899117196444422,
    "glass_breaking": 0.00004869161421083845,
    "rain": 0.00011065942817367613,
    "siren": 0.9980500936508179,
    "vacuum_cleaner": 0.00009659271745476872,
    "wind": 0.000032235071557806805
  },
  "model_version": "hearhere-panns-v2",
  "processing_ms": 123.4
}
```

The label/scores above came from the supplied WAV in the A2 review. `processing_ms: 123.4` is an illustrative placeholder, not a benchmark. Actual CLI output computes the time. `model_score` is uncalibrated; there is no `should_alert` in A2's output.

CLI exit codes: 0 success; 2 audio input/argument error; 3 model/metadata error; 4 inference error. Prediction errors go to stderr as JSON, leaving stdout empty. `--help` uses the normal argparse help output.

## Local predictor and legacy tuple usage

```python
from hearhere_audio.predict import AudioPredictor, predict_file

# For B: create once during service startup and reuse.
predictor = AudioPredictor(
    metadata_path="models/model_metadata.json",
)
result = predictor.predict_file("example.wav")

# One-shot scripts can use the agreed function signature.
result = predict_file("example.wav", metadata_path="models/model_metadata.json")
```

The one-shot function loads a model on each call. Use the persistent object in a server. Defaults resolve relative to this package's project root; explicitly supplied relative paths resolve relative to the process's working directory.

For B's existing `(label, confidence)` function:

```python
from hearhere_audio.backend_adapter import predict_label_score

# predictor is the object created during service startup above.
def predict(audio_path):
    return predict_label_score(audio_path, predictor)
```

The tuple's second item is the uncalibrated `model_score`, even if a legacy caller calls it confidence. This tuple helper is not the current HTTP response: the current server already has real inference and can select a different alert candidate from the top class. API consumers must preserve that distinction. See `docs/A2_handoff.md` and `docs/classify_api.md`. The `dev` branch carries the selected model, preprocessing and threshold artifacts together. B/D supplies the running backend; LinuxONE compatibility remains unverified.

## Audio and timing contract

- Actual WAV containers (`WAV`, `WAVEX`, `RF64`), mono or stereo input; file suffix alone is not trusted.
- Positive sample rate, with output resampled by A1's shared code to mono 16 kHz.
- At most five seconds by source frame count. Shorter audio is zero-padded to 80,000 samples. Longer audio is rejected; the caller supplies windows.
- Empty, corrupt and non-finite inputs raise `AudioProcessingError`.
- Files larger than 10 MiB are rejected before decoding. Set `max_audio_bytes` on the predictor to adjust the local file guard; B must also limit HTTP uploads and decoded request size.
- Selected features: one finite 2048-value PANNs CNN14 embedding from each shared mono 16 kHz, five-second window. The Random Forest baseline uses the 40-value MFCC extractor.
- Unknown/background sounds can still receive a confident known-class score. Silence is accepted as valid audio. No unknown-sound detector or silence-rejection policy is claimed here.
- `processing_ms` includes local header validation, feature extraction and model scoring. It excludes model loading, microphone-window collection and network transport.
- The first feature call can be much slower due to startup/JIT work. Run one startup prediction with a known WAV before marking the service ready, then measure warm requests separately.

## Tests

The integration sample is included at `examples/1-31482-A-42.wav`. Run the A2 suites:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_predict.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_api_client.py -v
```

Local predictor tests use the actual saved model, check all ten reference scores, shared features, load-once behavior, save/reload consistency, tuple compatibility, metadata integrity, CLI behavior and invalid inputs. API consumer tests check multipart upload and response handling, including a case where selected and predicted labels differ. They do not train a model, use Fold 5 or establish classifier accuracy.

## LinuxONE

See `docs/linuxone.md`. Local Windows success does not demonstrate LinuxONE deployment or hardware acceleration. D should verify Python, s390x package availability, installation, model loading and the same WAV output on the actual host.
