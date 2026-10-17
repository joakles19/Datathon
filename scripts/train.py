"""Train the Random Forest baseline and evaluate it on ESC-50 fold 4."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from hearhere_audio.config import (
    ALERT_CLASSES,
    CLASSES,
    FEATURE_ORDER,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    RESAMPLE_TYPE,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from hearhere_audio.features import extract_features_from_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "ESC-50"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
TRAIN_FOLDS = (1, 2, 3)
VALIDATION_FOLD = 4
RANDOM_SEED = 42
N_ESTIMATORS = 300
MINIMUM_ALERT_RECALL = 0.75


def read_rows(dataset_root: Path) -> list[dict[str, str]]:
    """Read and validate only folds used for training and validation."""
    metadata_path = dataset_root / "meta" / "esc50.csv"
    audio_dir = dataset_root / "audio"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ESC-50 metadata not found: {metadata_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        required_columns = {
            "filename", "target", "fold", "category", "esc10", "src_file", "take"
        }
        if not required_columns.issubset(reader.fieldnames or ()):
            raise ValueError(f"Metadata must contain columns: {sorted(required_columns)}")
        selected_rows = [row for row in reader if row["category"] in CLASSES]

    invalid_folds = [
        row["filename"]
        for row in selected_rows
        if not row["fold"].isdigit() or not 1 <= int(row["fold"]) <= 5
    ]
    if invalid_folds:
        raise ValueError(f"Invalid fold values in selected metadata rows: {invalid_folds[:5]}")
    rows = [
        row
        for row in selected_rows
        if int(row["fold"]) in (*TRAIN_FOLDS, VALIDATION_FOLD)
    ]

    rows.sort(key=lambda row: (int(row["fold"]), row["filename"]))
    duplicates = [name for name, count in Counter(row["filename"] for row in rows).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate filenames in selected folds: {duplicates[:5]}")

    missing = [row["filename"] for row in rows if not (audio_dir / row["filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audio files ({len(missing)}): {missing[:5]}")

    expected_classes_per_fold = 8
    counts = Counter((int(row["fold"]), row["category"]) for row in rows)
    incorrect = {
        (fold, category): counts[(fold, category)]
        for fold in (*TRAIN_FOLDS, VALIDATION_FOLD)
        for category in CLASSES
        if counts[(fold, category)] != expected_classes_per_fold
    }
    if incorrect:
        raise ValueError(f"Unexpected class counts per fold: {incorrect}")
    return rows


def extract_matrix(rows: list[dict[str, str]], audio_dir: Path) -> np.ndarray:
    features = []
    for index, row in enumerate(rows, start=1):
        vector = extract_features_from_file(audio_dir / row["filename"])
        if vector.shape != (len(FEATURE_ORDER),) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feature vector for {row['filename']}")
        features.append(vector)
        if index % 32 == 0 or index == len(rows):
            print(f"Extracted {index}/{len(rows)} feature vectors")
    return np.vstack(features)


def build_validation_metrics(
    true_labels: list[str], predicted_labels: list[str]
) -> tuple[dict, np.ndarray]:
    per_class = classification_report(
        true_labels,
        predicted_labels,
        labels=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(CLASSES))
    macro_precision = precision_score(
        true_labels, predicted_labels, labels=list(CLASSES), average="macro", zero_division=0
    )
    macro_recall = recall_score(
        true_labels, predicted_labels, labels=list(CLASSES), average="macro", zero_division=0
    )
    weighted_precision = precision_score(
        true_labels, predicted_labels, labels=list(CLASSES), average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        true_labels, predicted_labels, labels=list(CLASSES), average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=list(CLASSES), average="macro", zero_division=0)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(f1_score(true_labels, predicted_labels, labels=list(CLASSES), average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"]),
            }
            for label in CLASSES
        },
        "important_sound_classes": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"]),
            }
            for label in ALERT_CLASSES
        },
    }
    return metrics, matrix


def select_alert_thresholds(
    true_labels: np.ndarray,
    scores: np.ndarray,
    model_classes: np.ndarray,
    minimum_recall: float = MINIMUM_ALERT_RECALL,
) -> dict[str, dict[str, float | int]]:
    """Choose per-alert-class thresholds using Fold 4 scores only.

    The highest observed score that still meets the recall constraint minimizes
    binary alert positives, and therefore false alerts, for that class.
    """
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be greater than 0 and at most 1")
    if scores.shape != (len(true_labels), len(model_classes)):
        raise ValueError("scores must have one column per model class")

    class_indices = {label: index for index, label in enumerate(model_classes.tolist())}
    selected: dict[str, dict[str, float | int]] = {}
    for label in ALERT_CLASSES:
        if label not in class_indices:
            raise ValueError(f"Alert class is missing from model classes: {label}")
        class_scores = scores[:, class_indices[label]]
        is_positive = true_labels == label
        support = int(is_positive.sum())
        if support == 0:
            raise ValueError(f"No validation examples for alert class: {label}")

        # Thresholds only need to be checked at observed scores because the
        # predicted alert set is unchanged between adjacent score values.
        for threshold in sorted(np.unique(class_scores).tolist(), reverse=True):
            is_alert = class_scores >= threshold
            true_positive = int(np.logical_and(is_alert, is_positive).sum())
            false_positive = int(np.logical_and(is_alert, ~is_positive).sum())
            false_negative = support - true_positive
            recall = true_positive / support
            if recall < minimum_recall:
                continue
            alert_count = true_positive + false_positive
            precision = true_positive / alert_count if alert_count else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            selected[label] = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
                "alert_count": alert_count,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
            }
            break
        else:
            raise ValueError(
                f"No threshold for {label} meets minimum recall {minimum_recall}"
            )
    return selected


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["true\\predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix):
            writer.writerow([label, *map(int, row)])


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ["filename", "target", "category", "fold", "split", "esc10", "src_file", "take"]
        )
        for row in rows:
            fold = int(row["fold"])
            split = "train" if fold in TRAIN_FOLDS else "validation"
            writer.writerow(
                [
                    row["filename"], row["target"], row["category"], fold, split,
                    row["esc10"], row["src_file"], row["take"],
                ]
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_training(dataset_root: Path, model_dir: Path, report_dir: Path) -> None:
    rows = read_rows(dataset_root)
    audio_dir = dataset_root / "audio"
    train_rows = [row for row in rows if int(row["fold"]) in TRAIN_FOLDS]
    validation_rows = [row for row in rows if int(row["fold"]) == VALIDATION_FOLD]
    if len(train_rows) != 192 or len(validation_rows) != 64:
        raise ValueError(
            f"Expected 192 training and 64 validation rows; got "
            f"{len(train_rows)} and {len(validation_rows)}"
        )

    print("Extracting training features from folds 1-3")
    x_train = extract_matrix(train_rows, audio_dir)
    y_train = np.asarray([row["category"] for row in train_rows])
    print("Extracting validation features from fold 4")
    x_validation = extract_matrix(validation_rows, audio_dir)
    y_validation = np.asarray([row["category"] for row in validation_rows])

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_validation)
    probabilities = model.predict_proba(x_validation)
    metrics, matrix = build_validation_metrics(y_validation.tolist(), predicted.tolist())
    threshold_metrics = select_alert_thresholds(
        y_validation,
        probabilities,
        model.classes_,
    )
    threshold_values = {
        label: threshold_metrics[label]["threshold"]
        for label in ALERT_CLASSES
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hearhere_rf.joblib"
    joblib.dump(model, model_path)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path = report_dir / "training_manifest.csv"
    write_manifest(manifest_path, rows)

    metadata = {
        "schema_version": "1.0",
        "artifact_status": "trained_fold4_validated",
        "model_version": "hearhere-rf-v1",
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "created_at_utc": created_at,
        "model_sha256": sha256_file(model_path),
        "class_order": list(model.classes_),
        "important_classes": list(ALERT_CLASSES),
        "feature_extractor": {
            "entrypoint": "hearhere_audio.features.extract_features_from_file",
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "window_seconds": WINDOW_SECONDS,
            "short_audio_policy": "zero_pad",
            "long_audio_policy": "reject",
            "n_mfcc": N_MFCC,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "resample_type": RESAMPLE_TYPE,
            "feature_count": len(FEATURE_ORDER),
            "feature_order": ",".join(FEATURE_ORDER),
            "std_ddof": 0,
        },
        "dataset": {
            "name": "ESC-50",
            "source_url": "https://github.com/karolpiczak/ESC-50",
            "revision": None,
            "license": "CC BY-NC 3.0 (selected classes are outside the CC BY ESC-10 subset)",
            "source_metadata_sha256": sha256_file(dataset_root / "meta" / "esc50.csv"),
            "class_counts": {
                "train": {
                    category: sum(row["category"] == category for row in train_rows)
                    for category in CLASSES
                },
                "validation": {
                    category: sum(row["category"] == category for row in validation_rows)
                    for category in CLASSES
                },
            },
            "fold_counts": {
                str(fold): sum(int(row["fold"]) == fold for row in rows)
                for fold in (*TRAIN_FOLDS, VALIDATION_FOLD)
            },
            "sample_counts": {
                "train": len(train_rows),
                "validation": len(validation_rows),
            },
            "source_manifest_path": str(manifest_path.relative_to(PROJECT_ROOT))
            if manifest_path.is_relative_to(PROJECT_ROOT)
            else str(manifest_path),
            "split": {
                "train_folds": list(TRAIN_FOLDS),
                "validation_folds": [VALIDATION_FOLD],
                "final_test_folds": [5],
            },
        },
        "training": {
            "random_seed": RANDOM_SEED,
            "parameters": model.get_params(),
        },
        "dependency_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "soundfile": importlib.metadata.version("soundfile"),
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "evaluation_artifacts": {
            "validation_metrics": "reports/validation_metrics.json",
            "thresholds": "models/thresholds.json",
            "final_test_metrics": "reports/test_metrics.json",
        },
        "alert_policy": {
            "minimum_recall": MINIMUM_ALERT_RECALL,
            "selection_objective": "Minimize false alerts subject to the per-class recall constraint",
            "selection_method": "Highest observed Fold 4 score meeting the minimum recall for each alert class",
        },
    }
    threshold_report = {
        "schema_version": "1.0",
        "artifact_status": "fold4_thresholds_selected",
        "model_version": metadata["model_version"],
        "selection_split": {
            "dataset": "ESC-50",
            "folds": [VALIDATION_FOLD],
        },
        "score_interpretation": "Uncalibrated Random Forest class score; not a real-world event probability",
        "policy_owner": "backend",
        "minimum_recall": MINIMUM_ALERT_RECALL,
        "selection_objective": "Minimize false alerts subject to at least 75% recall for each important class",
        "selection_method": "For each important class, evaluate unique Fold 4 scores in descending order and select the highest score whose binary recall meets the minimum recall constraint",
        "recommended_alert_classes": list(ALERT_CLASSES),
        "minimum_scores": threshold_values,
        "validation_metrics": threshold_metrics,
        "notes": "Selected on Fold 4 only. B owns alert decisions, preferences, cooldowns, event merging, and acknowledgements. Fold 5 remains reserved for final evaluation.",
    }
    validation_report = {
        "schema_version": "1.0",
        "artifact_status": "fold4_measured_thresholds_selected",
        "model_version": metadata["model_version"],
        "evaluation": {
            "dataset": "ESC-50",
            "split": "validation",
            "folds": [VALIDATION_FOLD],
            "sample_count": len(validation_rows),
            "evaluated_at_utc": created_at,
        },
        "class_order": list(CLASSES),
        "summary": {
            key: metrics[key]
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            )
        },
        "per_class": metrics["per_class"],
        "important_sound_classes": metrics["important_sound_classes"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axis_order": "rows=true_class,columns=predicted_class; both follow class_order",
        "thresholds": threshold_values,
        "threshold_note": "Per-class thresholds selected on Fold 4 to minimize false alerts while retaining at least 75% recall for each important class.",
    }

    write_json(model_dir / "model_metadata.json", metadata)
    write_json(model_dir / "thresholds.json", threshold_report)
    write_json(report_dir / "validation_metrics.json", validation_report)
    write_confusion_matrix(report_dir / "confusion_matrix.csv", matrix)
    with (report_dir / "fold4_predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["filename", "true_class", "predicted_class", *model.classes_])
        for row, true_label, predicted_label, scores in zip(
            validation_rows, y_validation, predicted, probabilities
        ):
            writer.writerow(
                [row["filename"], true_label, predicted_label, *[float(score) for score in scores]]
            )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {model_dir / 'model_metadata.json'}")
    print(f"Saved threshold recommendations: {model_dir / 'thresholds.json'}")
    print(f"Saved validation report: {report_dir / 'validation_metrics.json'}")
    print(f"Saved used-data manifest: {manifest_path}")
    print(json.dumps(threshold_report, indent=2))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    run_training(args.dataset.resolve(), args.model_dir.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
