# Project structure

The repository keeps runtime code, model artifacts, evaluation evidence, and the browser client in separate top-level areas:

```text
hearhere_audio/    Shared audio preprocessing, model loading, inference, and CLI/API clients
backends/          FastAPI service, alert policy, event storage, and preference endpoints
frontend/          Vite/React browser application
scripts/           Dataset training, Fold 5 evaluation, setup, API, and benchmark commands
tests/             Python regression and contract tests
models/            Tracked model artifacts, metadata, and threshold policy
reports/           Tracked validation/final evaluation evidence and local benchmark records
examples/          Small attributed WAV and its usage notes
docs/              Handoffs, API/model contracts, deployment notes, and project documentation
```

The local development flow is:

1. `scripts/setup_environment.py` creates `.venv` and verifies the PANNs checkpoint in the user cache.
2. `backends/main.py` loads the validated model and serves `/health` and `/classify` on port 8000.
3. `frontend/` runs the Vite client on port 5173 and sends five-second mono WAV windows to the backend.

The following are intentionally local and ignored by Git: `.venv/`, `frontend/node_modules/`, `frontend/dist/`, `.pytest_cache/`, Python bytecode caches, `.codegraph/`, `.playwright-cli/`, `.test_tmp/`, recordings, datasets, logs, and runtime SQLite databases such as `events.db`. Tracked files under `models/` and `reports/` are reproducibility evidence; they must not be replaced with placeholder values.

LinuxONE notes remain in [`linuxone.md`](linuxone.md) for deployment history, but local macOS execution is the active development path.
