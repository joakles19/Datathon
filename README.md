# HearHere — A2 inference and API tools

This branch adds ZiYi's A2 predictor, JSON CLI, input validation, backend adapter and `/classify` API client on top of the `ai_david` model workflow. The trained model, preprocessing and Fold 4 thresholds are inherited from A1. A2 does not retrain the model or choose thresholds.

## Clone and run a real prediction

Python **3.12** is required for the documented model environment. On Windows, with Python 3.12 installed:

```powershell
git clone --branch ai_ZiYi --single-branch https://github.com/joakles19/Datathon.git
cd Datathon
py -3.12 scripts/setup_environment.py
.\.venv\Scripts\python.exe -m hearhere_audio examples/1-31482-A-42.wav --pretty
```

If `py -3.12` is unavailable, run `scripts/setup_environment.py` using the full path to your Python 3.12 interpreter. On macOS/Linux, use `python3.12 scripts/setup_environment.py`, then `.venv/bin/python` for subsequent commands.

The model and an attributed test WAV are included. No full dataset download is needed for this smoke test. The supplied example returns `siren` with uncalibrated score `0.8566666666666667`. It is a fold-1 integration example, not an accuracy benchmark. See [sample attribution](examples/README.md).

## Test the shared API

Obtain the running backend's reachable base URL from B/D, then:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

The address above works only when a backend is running on your own computer. A1's `127.0.0.1` address is not reachable from another teammate's machine. This branch contains A2 tools and A1 model artifacts; the current A1 baseline does not include `backends/main.py`. B/D supplies the server or shared deployment.

Upload field: `file`. Preserve `predicted_label` (model top class), `label` (backend-selected candidate), `confidence = scores[label]`, all eight scores and the server's `action`. B owns thresholds, cooldown, snooze, acknowledgement and event handling. Scores are not calibrated real-world probabilities.

## Run A2's 31 tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_predict.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_api_client.py -v
```

These are the 20 predictor/integration tests and 11 API-client tests. A1's other test modules are preserved; backend tests require B's backend source and the development dependencies. A2's tests do not use Fold 5 or establish real-world detection accuracy.

## Handoff documents

- [A2 predictor usage](docs/A2_README.md)
- [Current API contract](docs/classify_api.md)
- [Ownership and integration handoff](docs/A2_handoff.md)
- [LinuxONE loading guide](docs/linuxone.md)
- [A1 parity confirmation](docs/A1_parity_confirmation.json)
- [Local HTTP integration evidence](docs/classify_integration_verification.json)
- [A1 model artifact contract](docs/model-artifacts.md)
- [A1 workflow README retained for reference](docs/A1_model_workflow.md)

Local virtual environments, credentials, downloaded datasets, recordings, runtime databases and logs are excluded from Git. LinuxONE execution and a shared reachable API remain deployment checks for D.
