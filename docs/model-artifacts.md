# Model Artifact Contract

This document defines the files produced by the HearHere model workflow. Files ending in `.template.json` are scaffolds only. Training and evaluation code must write separate generated artifacts and must never report template values as results.

## Artifact lifecycle

| Artifact | Purpose | Ready when |
| --- | --- | --- |
| `models/model_metadata.template.json` | Contract for model identity, labels, feature settings, split, training parameters, and package versions | A real model has been trained and all fields describe that exact model |
| `reports/validation_metrics.template.json` | Contract for Fold 4 classification metrics and confusion matrix | The fixed training run has been evaluated on Fold 4 |
| `models/thresholds.template.json` | Contract for validation-selected alert recommendations | Threshold policy and selection method have been agreed with B and evaluated on Fold 4 |
| `reports/test_metrics.template.json` | Contract for final Fold 5 classification metrics | Model and rules are fixed and the reserved Fold 5 evaluation has run |
| `reports/environment.local.json` | Snapshot of the current local development environment | Informational snapshot; regenerate for the environment used to train the model |
| `models/hearhere_rf.joblib` | Serialized, actually trained Random Forest | Created by the reproducible training command |
| `models/model_metadata.json` | Metadata for the serialized model | Must identify the same model bytes and training run |
| `reports/validation_metrics.json` | Measured Fold 4 report | Generated from real Fold 4 audio and the identified model |
| `models/thresholds.json` | Agreed threshold recommendations | Selected using Fold 4 only and tied to the same model version |
| `reports/test_metrics.json` | Final Fold 5 report | Generated only after model and rules are frozen; must not drive later tuning |

## Shared values

The model metadata `class_order` array is alphabetical and must match the model's score columns:

`door_wood_knock`, `engine`, `glass_breaking`, `keyboard_typing`, `rain`, `siren`, `vacuum_cleaner`, `wind`.

The validation report's `class_order` array defines the row and column order of `confusion_matrix`. It follows `hearhere_audio/config.py`, which differs from the estimator's alphabetical score-column order. Keep both arrays explicit; do not infer one order from the other. Per-class metrics are keyed by class name.

The current Fold 4 run reports 64 validation samples, 0.8125 accuracy, 0.809334 macro F1, and 0.809334 weighted F1. These values are recorded in `reports/validation_metrics.json` and describe this baseline run only; they do not establish real-world microphone performance. Preserve the source-file manifest with the run and record the dataset revision when it becomes available.

The feature settings must match `hearhere_audio/config.py` and the shared extractor in `hearhere_audio/features.py`. The feature vector contains the means of MFCC 0-19 followed by the population standard deviations of MFCC 0-19. Metadata should record the complete extractor settings, not only the feature count.

Use official ESC-50 folds: 1-3 for training, 4 for validation and threshold selection, and 5 for the final evaluation. Keep the source file manifest and actual counts with the generated run. Dataset labels, filenames, and fold numbers are bookkeeping fields; they are not model input features.

## Metric and threshold rules

The validation report records per-class precision, recall, F1, and support, plus accuracy, macro and weighted precision/recall/F1, and a confusion matrix whose rows are true labels and columns are predicted labels in its declared `class_order`. Report the three important sounds separately: `door_wood_knock`, `siren`, and `glass_breaking`.

Thresholds are recommendations for B's alert policy. They must be chosen using Fold 4 and an agreed selection objective. Random Forest scores are not calibrated real-world event probabilities. Ordinary sounds should not produce important-sound alert recommendations. The classifier does not own event IDs, cooldowns, user preferences, or acknowledgement state.

Do not populate metric or threshold values until they have been measured. Do not use Fold 5 to choose features, model parameters, or thresholds. If a change is made after examining Fold 5 results, those results are no longer an independent final evaluation.

## Dependency records

`requirements.txt` is the install input for direct runtime dependencies. The generated `model_metadata.json` must also record the exact Python and relevant package versions used for that training run. `reports/environment.local.json` is only a snapshot of this machine; it is not a lockfile or evidence that the same environment works on LinuxONE.
