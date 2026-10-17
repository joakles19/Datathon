# HearHere — A2 inference delivery

Run A1's saved eight-class Random Forest on WAV recordings and return class scores. A2 supplies a persistent predictor, a one-shot function, a JSON CLI and a tuple adapter for B. Training and MFCC extraction remain A1's responsibility.

**Current integration status:** A1 has confirmed the sample features and all eight scores match across environments. The shared backend now exposes `POST /classify` on branch `ai_david`. Use the API consumer below for team integration; the local predictor remains available for diagnostics. See `docs/classify_api.md` for the current response contract.

## Consume the shared backend

With B/D's running service base URL:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

The URL above is an example. Upload field: `file`. The client preserves the server response, including `predicted_label` (highest-scoring class), `label` (selected alert candidate), `confidence` (`scores[label]`), `action`, `priority`, `message`, `event_id`, all eight `scores` and `model_version`. It does not load a local model or choose thresholds. Actions and event policy remain with B. Source being pushed to GitHub does not by itself establish a running server address.

## Setup on Windows

Use Python **3.12** for this handoff. A1 recorded 3.12.3; A2 was tested with 3.12.14. Confirm your interpreter before creating the environment. Do not use an existing global Python 3.14 environment for this documented setup.

```powershell
# Run inside the repository. Replace the path with your Python 3.12 interpreter.
& 'C:\path\to\Python312\python.exe' scripts/setup_environment.py
```

`requirements.txt` records A1's six named library versions plus the soxr/numba versions used by A2. `docs/a2-tested-environment.txt` records the complete Windows review environment; it is evidence, not a guarantee that all packages have LinuxONE wheels.

Keep these A1 artifacts together:

```text
hearhere_audio/config.py
hearhere_audio/features.py
models/hearhere_rf.joblib
models/model_metadata.json
```

The predictor checks the model's SHA256 against its metadata, classes/order, estimator type, fitted state and feature settings. Only load trusted team model artifacts. A checksum verifies a matching pair; it is not a signature. A different scikit-learn artifact version raises a clear loading error.

## Classify a WAV

```powershell
.\.venv\Scripts\python.exe -m hearhere_audio 'examples/1-31482-A-42.wav' --pretty
```

The supplied fold-1 sample is included under `examples/` for integration checks with dataset attribution. It is not a final-test performance example.

Alternative entry point:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py 'example.wav' --model models/hearhere_rf.joblib --metadata models/model_metadata.json --pretty
```

Successful stdout is JSON with exactly:

```json
{
  "predicted_class": "siren",
  "model_score": 0.8566666666666667,
  "scores": {
    "door_wood_knock": 0.016666666666666666,
    "engine": 0.01,
    "glass_breaking": 0.013333333333333334,
    "keyboard_typing": 0.05333333333333334,
    "rain": 0.0033333333333333335,
    "siren": 0.8566666666666667,
    "vacuum_cleaner": 0.01,
    "wind": 0.03666666666666667
  },
  "model_version": "hearhere-rf-v1",
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
    model_path="models/hearhere_rf.joblib",
    metadata_path="models/model_metadata.json",
)
result = predictor.predict_file("example.wav")

# One-shot scripts can use the agreed function signature.
result = predict_file("example.wav", "models/hearhere_rf.joblib", "models/model_metadata.json")
```

The one-shot function loads a model on each call. Use the persistent object in a server. Defaults resolve relative to this package's project root; explicitly supplied relative paths resolve relative to the process's working directory.

For B's existing `(label, confidence)` function:

```python
from hearhere_audio.backend_adapter import predict_label_score

# predictor is the object created during service startup above.
def predict(audio_path):
    return predict_label_score(audio_path, predictor)
```

The tuple's second item is the uncalibrated `model_score`, even if a legacy caller calls it confidence. This tuple helper is not the current HTTP response: the current server already has real inference and can select a different alert candidate from the top class. API consumers must preserve that distinction. See `docs/A2_handoff.md` and `docs/classify_api.md`. The published A2 branch is based on ai_david and preserves A1's canonical model/preprocessing/threshold files. B/D supplies the running backend.

## Audio and timing contract

- Actual WAV containers (`WAV`, `WAVEX`, `RF64`), mono or stereo input; file suffix alone is not trusted.
- Positive sample rate, with output resampled by A1's shared code to mono 16 kHz.
- At most five seconds by source frame count. Shorter audio is zero-padded to 80,000 samples. Longer audio is rejected; the caller supplies windows.
- Empty, corrupt and non-finite inputs raise `AudioProcessingError`.
- Files larger than 10 MiB are rejected before decoding. Set `max_audio_bytes` on the predictor to adjust the local file guard; B must also limit HTTP uploads and decoded request size.
- Features: MFCC 0–19 means followed by MFCC 0–19 standard deviations, exactly 40 finite values.
- Unknown/background sounds can still receive a confident known-class score. Silence is accepted as valid audio. No unknown-sound detector or silence-rejection policy is claimed here.
- `processing_ms` includes local header validation, feature extraction and model scoring. It excludes model loading, microphone-window collection and network transport.
- The first feature call can be much slower due to startup/JIT work. Run one startup prediction with a known WAV before marking the service ready, then measure warm requests separately.

## Tests

The integration sample is included at `examples/1-31482-A-42.wav`. Run the A2 suites:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_predict.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_api_client.py -v
```

Local predictor tests use the actual saved model, check all eight reference scores, shared features, load-once behavior, save/reload consistency, tuple compatibility, metadata integrity, CLI behavior and invalid inputs. API consumer tests check multipart upload and response handling, including a case where selected and predicted labels differ. They do not train a model, use Fold 5 or establish classifier accuracy. A1 confirmed the reference on 17 October 2026; see `docs/A1_parity_confirmation.json` for the reported environment and independently compared feature vector.

## LinuxONE

See `docs/linuxone.md`. Local Windows success does not demonstrate LinuxONE deployment or hardware acceleration. D should verify Python, s390x package availability, installation, model loading and the same WAV output on the actual host.
