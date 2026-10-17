# HearHere

HearHere is a sound-awareness prototype that classifies selected environmental sounds and presents visual prompts. It is an assistive prototype, not a validated emergency-warning device.

## Current status

- Shared WAV preprocessing and 40-dimensional MFCC extraction are implemented in `hearhere_audio/`.
- The Random Forest training and Fold 4 validation flow is implemented in `scripts/train.py`.
- Fold 5 remains reserved for final evaluation after the model and alert rules are fixed.
- Do not treat files ending in `.template.json` as trained-model metadata or measured results.

## Audio feature contract

The shared feature function converts audio to mono at 16 kHz, pads clips shorter than five seconds, and rejects clips longer than five seconds. It extracts 20 MFCCs using a 2,048-sample FFT and a 512-sample hop, then returns the 20 coefficient means followed by their 20 population standard deviations.

Run the current feature tests from the repository root:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Train the baseline and create the model plus Fold 4 validation artifacts:

```sh
python -m scripts.train --dataset data/raw/ESC-50
```

## Dataset and evaluation plan

The selected ESC-50 classes are `door_wood_knock`, `siren`, `glass_breaking`, `rain`, `wind`, `keyboard_typing`, `vacuum_cleaner`, and `engine`. The intended split is official folds 1-3 for training, fold 4 for validation and threshold selection, and fold 5 for final evaluation after the model and rules are fixed. Dataset files are local inputs and are not included in Git.

## Model artifacts

Artifact field meanings and the rules for recording measured results are documented in [`docs/model-artifacts.md`](docs/model-artifacts.md). Local environment versions are recorded separately in [`reports/environment.local.json`](reports/environment.local.json); that snapshot does not establish LinuxONE compatibility.
