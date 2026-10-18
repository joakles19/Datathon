# HearHere

HearHere is a sound-awareness prototype that classifies selected environmental sounds and presents visual prompts. It is an assistive prototype, not a validated emergency-warning device.

## Current status

- Shared WAV preprocessing, 40-dimensional MFCC extraction and frozen PANNs CNN14 embeddings are implemented in `hearhere_audio/`.
- `scripts/train.py` compares the existing Random Forest baseline with standardized Logistic Regression over PANNs embeddings.
- Fold 4 alert thresholds require at least 85% precision per enabled class; nonqualifying classes are disabled as alerts but remain classifier outputs.
- `POST /classify` loads the validated model, returns all ten class scores, and applies B's per-sound alert policy for six alert classes.
- Fold 5 was evaluated once after the model and alert rules were fixed; do not rerun it or tune from its results.
- Do not treat files ending in `.template.json` as trained-model metadata or measured results.

## Audio feature contract

The shared feature function converts audio to mono at 16 kHz, pads clips shorter than five seconds, and rejects clips longer than five seconds. It extracts 20 MFCCs using a 2,048-sample FFT and a 512-sample hop, then returns the 20 coefficient means followed by their 20 population standard deviations.

Run the current feature tests from the repository root:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Compare both candidates and create the selected model plus Fold 4 validation artifacts (requires the verified local PANNs checkpoint):

```sh
python -m scripts.train --dataset data/raw/ESC-50
```

Run the API locally after installing runtime and development requirements:

```sh
uvicorn backends.main:app --reload
```

## Dataset and evaluation plan

The selected v2 ESC-50 classes are `car_horn`, `clock_alarm`, `crying_baby`, `door_wood_knock`, `engine`, `glass_breaking`, `rain`, `siren`, `vacuum_cleaner`, and `wind`. `keyboard_typing` is excluded because it was too easily confused with door knocks. The intended split is official folds 1-3 for training, fold 4 for validation and threshold selection, and fold 5 for final evaluation after the model and rules are fixed. Dataset files are local inputs and are not included in Git.

The current selected candidate is PANNs CNN14 with a Logistic Regression head. Fold 4 selected five alert classes; `car_horn` is disabled because it had no threshold at or above the required precision. Fold 5 was run once after model and thresholds were locked. See `docs/model-artifacts.md` for measured results and limitations. LinuxONE remains a hard deployment gate and has not been verified.

## Model artifacts

Artifact field meanings and the rules for recording measured results are documented in [`docs/model-artifacts.md`](docs/model-artifacts.md). Local environment versions are recorded separately in [`reports/environment.local.json`](reports/environment.local.json); that snapshot does not establish LinuxONE compatibility.
