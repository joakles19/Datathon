# Model Artifact Contract

This document defines the files produced by the HearHere model workflow. Files ending in `.template.json` are scaffolds only. Training and evaluation code must write separate generated artifacts and must never report template values as results.

## Artifact lifecycle

| Artifact | Purpose | Ready when |
| --- | --- | --- |
| `models/model_metadata.template.json` | Contract for selected model identity, labels, feature settings, split, training parameters, and package versions | A real model has been trained and all fields describe that exact model |
| `reports/validation_metrics.template.json` | Contract for Fold 4 classification metrics and confusion matrix | The fixed training run has been evaluated on Fold 4 |
| `models/thresholds.template.json` | Contract for validation-selected alert recommendations | Threshold policy is evaluated on Fold 4 and nonqualifying alerts are disabled |
| `reports/test_metrics.template.json` | Contract for final Fold 5 classification metrics | Model and rules are fixed and the reserved Fold 5 evaluation has run |
| `reports/environment.local.json` | Snapshot of the current local development environment | Informational snapshot; regenerate for the environment used to train the model |
| `models/hearhere_rf.joblib` | Existing Random Forest comparison baseline trained on folds 1-3 | Included with its baseline SHA-256 in active metadata |
| `models/hearhere_panns_lr.joblib` | Standardized Logistic Regression head over frozen CNN14 embeddings | Created by the reproducible training command |
| `models/model_metadata.json` | Metadata for the serialized model | Must identify the same model bytes and training run |
| `reports/validation_metrics.json` | Measured Fold 4 report | Generated from real Fold 4 audio and the identified model |
| `models/thresholds.json` | Agreed threshold recommendations | Selected using Fold 4 only and tied to the same model version |
| `reports/test_metrics.json` | Final Fold 5 report | Generated only after model and rules are frozen; must not drive later tuning |

## Shared values

The model metadata `class_order` array is alphabetical and must match the model's score columns:

`car_horn`, `clock_alarm`, `crying_baby`, `door_wood_knock`, `engine`, `glass_breaking`, `rain`, `siren`, `vacuum_cleaner`, `wind`.

The validation report's `class_order` array defines the row and column order of `confusion_matrix`. It follows `hearhere_audio/config.py`, which differs from the estimator's alphabetical score-column order. Keep both arrays explicit; do not infer one order from the other. Per-class metrics are keyed by class name.

The current Fold 4 comparison reports 80 validation samples: the PANNs candidate reached mean alert recall `0.8125` across all six important classes and macro-F1 `0.9378`; the Random Forest reached `0.7292` and `0.7639`. Five PANNs alert classes met the precision rule and `car_horn` remains disabled by default. These Fold 4 values describe eight examples per class and do not establish real-world microphone performance. `reports/training_manifest.csv` records the training and validation files.

The model comparison uses the shared mono, 16 kHz, five-second audio preparation. The Random Forest receives the 40-value MFCC vector (means of MFCC 0-19 followed by population standard deviations); the PANNs candidate receives one finite 2048-value frozen CNN14 embedding. Training and inference use the same respective extraction functions. Metadata records the selected feature settings and upstream checkpoint checksum.

Use official ESC-50 folds: 1-3 for training, 4 for validation and threshold selection, and 5 for the final evaluation. Keep the source file manifest and actual counts with the generated run. Dataset labels, filenames, and fold numbers are bookkeeping fields; they are not model input features.

## Metric and threshold rules

The validation report records per-class precision, recall, F1, and support, plus accuracy, macro and weighted precision/recall/F1, and a confusion matrix whose rows are true labels and columns are predicted labels in its declared `class_order`. Report the six important sounds separately: `door_wood_knock`, `siren`, `glass_breaking`, `clock_alarm`, `crying_baby`, and `car_horn`.

Thresholds are recommendations for B's alert policy. For each candidate and important class, Fold 4 selects the threshold with maximum recall among observed scores whose one-vs-rest precision is at least 85%. Classes with no qualifying threshold are disabled as alerts but remain in model output. Candidates rank by mean recall over all six classes (disabled classes count as zero), then macro-F1, then Random Forest preference. The selected thresholds record precision, recall, F1, support, alert count, and false-positive count. Scores are not calibrated real-world event probabilities. The classifier does not own event IDs, cooldowns, user preferences, or acknowledgement state.

`hearhere_audio.inference.HearHereClassifier` verifies the model SHA-256 before deserialization, checks the model class and declared 40- or 2048-feature contract, and returns the top label plus a score for every v2 class. The backend enables only Fold 4 qualifying alert recommendations and retains ownership of alert actions. The `/classify` response's `predicted_label` is the classifier's top class, while `label` is the selected threshold-clearing alert candidate (or the top class when no alert qualifies). Its legacy `confidence` field contains the selected label's uncalibrated score; `scores` contains all class scores and `model_version` identifies the loaded artifact.

Do not populate metric or threshold values until they have been measured. Do not use Fold 5 to choose features, model parameters, or thresholds. If a change is made after examining Fold 5 results, those results are no longer an independent final evaluation.

The locked PANNs artifact was evaluated once on Fold 5 after Fold 4 selection: 80 samples, accuracy `0.9750`, macro-F1 `0.9749`. The per-class report and fixed-threshold metrics are in `reports/test_metrics.json`. Each class has eight examples; these results are a small held-out benchmark and do not establish microphone accuracy. Do not rerun Fold 5 or tune from it.

The local warm-inference benchmark in `reports/inference_benchmark.local.json` measured 25 requests on macOS arm64: median `26.85 ms`, p95 `27.71 ms`. It excludes model startup and HTTP transport and does not establish LinuxONE latency or deployability.

## Dependency records

`requirements.txt` is the install input for direct runtime dependencies. The generated `model_metadata.json` must also record the exact Python and relevant package versions used for that training run. `reports/environment.local.json` is only a snapshot of this machine; it is not a lockfile or evidence that the same environment works on LinuxONE.
