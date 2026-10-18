# HearHere

HearHere is an audio safety monitor. It listens to a short recording or a live microphone stream, classifies the sound, and turns selected high-signal sounds into an alert that a person can acknowledge, snooze, or review later. The current implementation is a local development and demonstration system; it is not a certified emergency service and must not be the only protection for a person or property.

The active artifact is `hearhere-panns-v2`: a frozen PANNs CNN14 AudioSet embedding extractor followed by a standardized multinomial Logistic Regression head. The FastAPI backend owns alert thresholds, live-window confirmation, cooldowns, snooze state, acknowledgement state, and the local SQLite event log. The React client presents the returned result and does not reinterpret the model score as a probability.

## Implemented product flow

The classifier accepts mono 16 kHz audio in a five-second window. Short clips are zero-padded; clips longer than five seconds are rejected by the shared preprocessing contract. The v2 model produces scores for these ten ESC-50 classes:

`car_horn`, `clock_alarm`, `crying_baby`, `door_wood_knock`, `engine`, `glass_breaking`, `rain`, `siren`, `vacuum_cleaner`, `wind`.

The browser supports three workflows:

- Upload an audio file. The browser decodes it, converts it to mono 16 kHz PCM WAV, and splits it into windows of at most five seconds.
- Record one microphone clip and analyse it when recording stops.
- Start live monitoring. The browser sends overlapping five-second windows once per second. The backend requires the same threshold-clearing important sound in at least two of the latest three windows before it creates a live alert.

Demo mode can simulate responses for a presentation with `VITE_DEMO_MODE=true`. It is separate from real inference and never proves model performance.

## Architecture

```text
Browser microphone or upload
          |
          v
React/Vite client  -- mono 16 kHz PCM WAV -->  FastAPI /classify
                                                      |
                                   PANNs CNN14 2048-D embedding
                                                      |
                                      StandardScaler + LogisticRegression
                                                      |
                              threshold policy, confirmation, cooldown
                                                      |
                                      SQLite event log + JSON response
```

The frontend runs on port `5173` and the API runs on port `8000` by default. `events.db` is a local runtime database and is not a production event store. The full diagram is in [docs/solution-architecture.md](docs/solution-architecture.md).

## Quick start

The reproducible local environment uses Python **3.12**. The setup script creates `.venv`, installs the pinned CPU dependencies, checks the environment, and downloads the 358,668,570-byte CNN14 checkpoint to `~/.cache/hearhere/panns/`. The checkpoint is verified against the Zenodo MD5 before inference and is not committed to the repository.

Windows PowerShell:

```powershell
git clone --branch main --single-branch https://github.com/joakles19/Datathon.git
cd Datathon
py -3.12 scripts/setup_environment.py
.\.venv\Scripts\python.exe -m hearhere_audio examples/1-31482-A-42.wav --pretty
```

macOS/Linux:

```sh
git clone --branch main --single-branch https://github.com/joakles19/Datathon.git
cd Datathon
python3.12 scripts/setup_environment.py
.venv/bin/python -m hearhere_audio examples/1-31482-A-42.wav --pretty
```

The included WAV is an integration smoke sample, not a benchmark sample. It should classify as `siren`; its score is an uncalibrated model score. See [sample attribution](examples/README.md) and the checkpoint attribution in [`hearhere_audio/panns.py`](hearhere_audio/panns.py).

## Run the application

Start the backend from the repository root:

```sh
.venv/bin/python -m uvicorn backends.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the frontend:

```sh
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The default API is `http://localhost:8000`. Set `VITE_API_URL` when the API is hosted elsewhere, or set `VITE_DEMO_MODE=true` for simulated presentation results. The Settings panel's **Test connection** button checks `/health` and the expected model version.

For a real API smoke test while the backend is running:

```sh
.venv/bin/python scripts/classify_api.py examples/1-31482-A-42.wav \
  --base-url http://127.0.0.1:8000 --pretty
```

`127.0.0.1` and `localhost` refer to the requesting computer. A teammate on another machine needs a backend URL reachable from both machines; a loopback URL is not shared access.

## API behavior

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports API readiness and the loaded model version. |
| `POST /classify` | Accepts multipart field `file`; optional `stream_id` and `window_index` enable live-window confirmation. |
| `GET /events?limit=N` | Returns the newest local events; `N` is 1 to 200 and defaults to 50. |
| `POST /events/{event_id}/ack` | Acknowledges one stored event. |
| `GET /prefs` | Returns per-class enabled state, threshold, cooldown, priority, and message. |
| `PUT /prefs/{label}` | Updates enabled state, threshold, cooldown, or priority for one configured class. |
| `POST /snooze` | Temporarily suppresses alerts for a class. |

The `/classify` response has three fields that must be kept distinct:

- `predicted_label` is the model's highest-scoring class.
- `label` is the backend-selected important-sound candidate. It may differ from `predicted_label` when an important class clears its own alert threshold.
- `confidence` is exactly `scores[label]`. It is an uncalibrated classifier score, not the probability that a real-world event occurred.

The response also includes all ten `scores`, `model_version`, `action` (`alert`, `ignore`, or `suppressed`), optional `priority` and `message`, `event_id`, `processing_ms`, and `confirmation`. Live requests return `confirmation: pending` until a second matching window arrives; no event is created during that pending response.

## Model and evaluation

Training uses the selected ESC-50 classes with folds 1-3 (240 clips) for training and fold 4 (80 clips, eight examples per class) for validation and threshold selection. The model input is one 2,048-dimensional frozen PANNs CNN14 16 kHz AudioSet embedding. The Logistic Regression head uses `StandardScaler`, `C=1.0`, `lbfgs`, `max_iter=2000`, and random seed 42. The trained model and metadata are checked by SHA-256 before the API starts.

### Top-class classification metrics

These are ordinary ten-class predictions from the locked v2 classifier. They are separate from the alert-threshold results below.

| Split | Samples | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| ESC-50 Fold 4 validation | 80 | 93.75% | 93.78% | 93.78% |
| ESC-50 Fold 5 final test | 80 | 97.50% | 97.49% | 97.49% |

On Fold 4, the important classes had these top-class metrics (precision / recall / F1):

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `clock_alarm` | 100.0% | 100.0% | 100.0% | 8 |
| `crying_baby` | 100.0% | 100.0% | 100.0% | 8 |
| `door_wood_knock` | 100.0% | 100.0% | 100.0% | 8 |
| `glass_breaking` | 100.0% | 100.0% | 100.0% | 8 |
| `car_horn` | 80.0% | 100.0% | 88.9% | 8 |
| `siren` | 80.0% | 100.0% | 88.9% | 8 |

Fold 5 was evaluated once after the model and Fold 4 rules were locked. It is a small held-out benchmark, not evidence of real-world microphone accuracy. Do not rerun Fold 5 or use its results to tune features, parameters, or thresholds.

## Alert thresholds and policy

The backend applies a separate one-vs-rest policy to the six important sounds: `door_wood_knock`, `siren`, `glass_breaking`, `clock_alarm`, `crying_baby`, and `car_horn`. For each class, Fold 4 chooses the observed score threshold with the highest recall among thresholds reaching at least **85% precision**. A class with no qualifying threshold is disabled by default.

The current defaults are:

| Class | Default threshold | Default alert state |
| --- | ---: | --- |
| `clock_alarm` | 0.9809 | enabled |
| `crying_baby` | 0.7742 | enabled |
| `door_wood_knock` | 0.5444 | enabled |
| `glass_breaking` | 0.6603 | enabled |
| `siren` | 0.9974 | enabled |
| `car_horn` | no qualifying Fold 4 threshold | disabled |

The threshold metrics must not be confused with the top-class metrics. For example, `siren` has Fold 4 top-class precision 80% and recall 100%, while its selected alert threshold has precision 100% and recall 87.5%. Thresholds are policy controls over uncalibrated scores; they do not establish real-world precision. Cooldowns, snooze, preferences, event merging, and acknowledgement remain backend responsibilities.

## Local performance

The recorded macOS arm64 benchmark used 25 warm predictions on the included WAV and included WAV validation, preprocessing, PANNs embedding extraction, and classifier scoring. It excluded model startup and HTTP transport.

| Measurement | Result |
| --- | ---: |
| Model startup | 1,688.65 ms |
| Warm median | 26.40 ms |
| Warm p95 | 27.64 ms |
| Warm maximum | 29.26 ms |
| Monitoring request cadence | 1,000 ms |

The warm path fits within the one-second monitoring cadence on this machine. This is not a LinuxONE latency measurement and does not include browser capture or network time.

## IBM LinuxONE: completed deployment validation

LinuxONE CPU deployment and API smoke validation are complete. The evidence is recorded in [docs/linuxone_deployment_evidence.json](docs/linuxone_deployment_evidence.json), verified on 2026-10-18 against an Ubuntu 24.04.5 LTS `s390x` host.

The verified runtime used:

- IBM's `torch-1.10.0-cp39-cp39-linux_s390x.whl` on Python 3.9.18.
- NumPy 1.23.5, SciPy 1.9.3, torchlibrosa 0.1.0, FastAPI 0.128.8, and Uvicorn 0.39.0.
- The same 358,668,570-byte PANNs CNN14 checkpoint, verified with MD5 `362fc5ff18f1d6ad2f6d464b45893f2c`.
- `models/hearhere_panns_lr_portable.npz`, a NumPy `StandardScaler` plus multinomial Logistic Regression representation used where a compatible scikit-learn wheel is unavailable.

The completed smoke gates show that the example WAV produces a finite, repeatable 2,048-value embedding; the deployed `/health` response reports `hearhere-panns-v2`; `/classify` returns the expected contract and ten class scores; and the portable head differs from the local joblib pipeline by at most `6.1e-08` in probability on the integration sample. The LinuxONE API returned `siren` for the supplied example and measured five sample processing times between 359.4 ms and 450.9 ms on the two-CPU test VM.

LinuxONE uses a small deployment-specific librosa compatibility layer because no usable s390x Numba wheel was available. This is recorded and reproducible in [`scripts/linuxone_compat/README.md`](scripts/linuxone_compat/README.md). The deployment is complete for CPU smoke validation and the demonstrated API contract. Public ingress, HTTPS, a system service, production capacity planning, and a specialised AI accelerator were not part of this validation.

See the [LinuxONE loading guide](docs/linuxone.md) for the exact setup and operational checks.

## Training, evaluation, and benchmark commands

The dataset is intentionally not included in the repository. With ESC-50 available at `data/raw/ESC-50` and the verified checkpoint in the user cache:

```sh
# Train the Random Forest baseline and PANNs candidate; select using Fold 4.
.venv/bin/python scripts/train.py --dataset data/raw/ESC-50

# Run the one-time final Fold 5 evaluation on the locked artifacts.
.venv/bin/python scripts/evaluate_fold5.py --dataset data/raw/ESC-50

# Measure warm local inference against the one-second cadence.
.venv/bin/python scripts/benchmark_inference.py --iterations 25
```

`evaluate_fold5.py` refuses to overwrite an existing `reports/test_metrics.json`; this protects the independence of the final test. Do not delete the report to enable tuning.

## Verification

Run the Python tests from the repository root:

```sh
.venv/bin/python -m pytest -q
```

Build the frontend:

```sh
cd frontend
npm ci
npm run build
```

The regression suite covers audio preparation, model artifact validation, inference, backend alert behavior, live-window confirmation, HTTP contracts, and the API client. Tests and ESC-50 reports do not establish performance on unseen microphones, rooms, noise conditions, or emergency scenarios.

## Repository map

```text
hearhere_audio/    Audio preprocessing, PANNs loading, inference, CLI, and API client
backends/          FastAPI service, alert policy, preferences, and SQLite event storage
frontend/          React/Vite recorder, uploader, live monitor, results, and settings UI
scripts/           Environment setup, training, Fold 5 evaluation, API smoke test, benchmark
tests/             Python regression and contract tests
models/            Trained artifacts, metadata, thresholds, and portable LinuxONE head
reports/           Fold 4/Fold 5 metrics, confusion matrices, and local benchmark evidence
examples/          Attributed integration WAV and usage notes
docs/              API, model, project-structure, handoff, and LinuxONE documentation
```

See also:

- [Current API contract](docs/classify_api.md)
- [Model artifact contract](docs/model-artifacts.md)
- [LinuxONE loading guide](docs/linuxone.md)
- [Solution architecture](docs/solution-architecture.md)
- [Project structure](docs/project-structure.md)
- [Example attribution](examples/README.md)

## Limitations and responsible claims

- ESC-50 Fold 4 and Fold 5 contain only eight examples per class; their metrics are controlled dataset results, not field detection rates.
- Model scores are not calibrated event probabilities. A high score does not prove that a real incident occurred.
- Microphone placement, room acoustics, competing sounds, browser permissions, browser codec support, CPU load, and network delay can change behavior.
- A live alert requires two matching threshold-clearing windows, so a short or intermittent sound may remain pending or be missed.
- The local SQLite database, wildcard development CORS policy, loopback defaults, and demo mode are suitable for development and demonstration, not an internet-facing deployment.
- The system should support human awareness and review. It must not replace smoke alarms, security systems, emergency services, safeguarding procedures, or other required safety controls.
