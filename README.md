# HearHere

HearHere connects the selected audio classifier, FastAPI alert policy, and React interface. The active model uses frozen PANNs CNN14 embeddings with a standardized Logistic Regression head, selected against the existing Random Forest on ESC-50 Fold 4.

## Clone and run a real prediction

Python **3.12** is required for the documented model environment. Setup installs the CPU PyTorch/PANNs dependencies and downloads the 359 MB CNN14 checkpoint to `~/.cache/hearhere/panns/`, verifying it against the published Zenodo checksum. The checkpoint is not committed. On Windows, with Python 3.12 installed:

```powershell
git clone --branch dev --single-branch https://github.com/joakles19/Datathon.git
cd Datathon
py -3.12 scripts/setup_environment.py
.\.venv\Scripts\python.exe -m hearhere_audio examples/1-31482-A-42.wav --pretty
```

If `py -3.12` is unavailable, run `scripts/setup_environment.py` using the full path to your Python 3.12 interpreter. On macOS/Linux, use `python3.12 scripts/setup_environment.py`, then `.venv/bin/python` for subsequent commands.

The selected classifier and an attributed test WAV are included. Setup verifies the PANNs checkpoint before use; no full dataset download is needed for this smoke test. The supplied example returns `siren` with uncalibrated score about `0.998`. It is a fold-1 integration example, not an accuracy benchmark. PANNs checkpoint attribution and license are recorded in `hearhere_audio/panns.py`; see [sample attribution](examples/README.md).

## Run the connected frontend and backend

Python **3.12** is required. Install Python dependencies from the repository root, then start the API:

```sh
python3.12 scripts/setup_environment.py
.venv/bin/python -m uvicorn backends.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the React app:

```sh
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Demo mode is off by default, and the frontend sends converted WAV windows to `http://localhost:8000/classify`. Set `VITE_API_URL` for another reachable backend or `VITE_DEMO_MODE=true` only when simulated results are wanted. The Settings panel's **Test connection** checks `/health`.

The browser converts uploaded or recorded audio to mono 16 kHz PCM WAV and splits it into windows of at most five seconds. Live monitoring requests browser noise suppression and automatic gain control when supported, then sends overlapping five-second windows once per second. The backend confirms an important sound in two of the latest three windows before creating a live alert event. The UI displays pending checks separately; single recordings and uploads retain their one-clip behavior. Browser audio processing is device-dependent and must be checked with real target microphones.

## Test the shared API

With the local backend running, or using a reachable shared backend URL:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

The address above works only when a backend is running on your own computer. A `127.0.0.1` address is not reachable from another teammate's machine; for team testing, use a backend URL reachable by that teammate.

Upload field: `file`. Preserve `predicted_label` (model top class), `label` (backend-selected candidate), `confidence = scores[label]`, all ten v2 class scores and the server's `action`. B owns thresholds, cooldown, snooze, acknowledgement and event handling. Scores are not calibrated real-world probabilities.

## Run the tests

```sh
.venv/bin/python -m pytest -q
```

This covers audio features, model loading and inference, the HTTP backend, and the API client. Tests do not use Fold 5 or establish real-world detection accuracy.

## Handoff documents

- [A2 predictor usage](docs/A2_README.md)
- [Current API contract](docs/classify_api.md)
- [Ownership and integration handoff](docs/A2_handoff.md)
- [LinuxONE loading guide](docs/linuxone.md)
- [A1 parity confirmation](docs/A1_parity_confirmation.json)
- [Local HTTP integration evidence](docs/classify_integration_verification.json)
- [Model comparison and artifact contract](docs/model-artifacts.md)
- [Project structure and local runtime boundaries](docs/project-structure.md)
- [A1 workflow README retained for reference](docs/A1_model_workflow.md)

Local virtual environments, credentials, downloaded datasets, recordings, runtime databases and logs are excluded from Git. LinuxONE documentation is retained for deployment history; local macOS execution is the active development path.
