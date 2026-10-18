"""Compare the existing Random Forest with a frozen PANNs embedding head."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
from hearhere_audio.panns import (
    CHECKPOINT_MD5,
    CHECKPOINT_URL,
    DEFAULT_CACHE_PATH,
    EMBEDDING_DIM,
    PannsEmbeddingExtractor,
    extract_embedding_from_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "ESC-50"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
TRAIN_FOLDS = (1, 2, 3)
VALIDATION_FOLD = 4
RANDOM_SEED = 42
N_ESTIMATORS = 300
MINIMUM_ALERT_PRECISION = 0.85
SAMPLES_PER_CLASS_PER_FOLD = 8
LOGISTIC_REGRESSION_PARAMETERS = {
    "C": 1.0,
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}


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

    expected_classes_per_fold = SAMPLES_PER_CLASS_PER_FOLD
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


def extract_embedding_matrix(
    rows: list[dict[str, str]], audio_dir: Path, extractor: PannsEmbeddingExtractor
) -> np.ndarray:
    embeddings = []
    for index, row in enumerate(rows, start=1):
        vector = extract_embedding_from_file(audio_dir / row["filename"], extractor)
        if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid PANNs embedding for {row['filename']}")
        embeddings.append(vector)
        if index % 32 == 0 or index == len(rows):
            print(f"Extracted {index}/{len(rows)} PANNs embeddings")
    return np.vstack(embeddings)


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
    minimum_precision: float = MINIMUM_ALERT_PRECISION,
) -> dict[str, dict[str, float | int]]:
    """Choose maximum-recall Fold 4 thresholds that meet precision target."""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be greater than 0 and at most 1")
    if scores.shape != (len(true_labels), len(model_classes)):
        raise ValueError("scores must have one column per model class")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("scores must contain finite values between zero and one")

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

        qualifying = []
        for threshold in sorted(np.unique(class_scores).tolist(), reverse=True):
            is_alert = class_scores >= threshold
            true_positive = int(np.logical_and(is_alert, is_positive).sum())
            false_positive = int(np.logical_and(is_alert, ~is_positive).sum())
            false_negative = support - true_positive
            recall = true_positive / support
            alert_count = true_positive + false_positive
            precision = true_positive / alert_count if alert_count else 0.0
            if precision < minimum_precision:
                continue
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            qualifying.append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
                "alert_count": alert_count,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
            })
        if qualifying:
            selected[label] = max(
                qualifying,
                key=lambda result: (
                    result["recall"], result["precision"], result["threshold"]
                ),
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
    final_report = report_dir / "test_metrics.json"
    if final_report.exists():
        raise FileExistsError(
            f"Fold 5 has already been evaluated: {final_report}; do not retrain or tune from it"
        )
    rows = read_rows(dataset_root)
    audio_dir = dataset_root / "audio"
    train_rows = [row for row in rows if int(row["fold"]) in TRAIN_FOLDS]
    validation_rows = [row for row in rows if int(row["fold"]) == VALIDATION_FOLD]
    expected_train = len(CLASSES) * len(TRAIN_FOLDS) * SAMPLES_PER_CLASS_PER_FOLD
    expected_validation = len(CLASSES) * SAMPLES_PER_CLASS_PER_FOLD
    if len(train_rows) != expected_train or len(validation_rows) != expected_validation:
        raise ValueError(
            f"Expected {expected_train} training and {expected_validation} validation rows; "
            f"got {len(train_rows)} and {len(validation_rows)}"
        )

    y_train = np.asarray([row["category"] for row in train_rows])
    y_validation = np.asarray([row["category"] for row in validation_rows])

    print("Extracting Random Forest features from folds 1-4")
    rf_validation_features = extract_matrix(validation_rows, audio_dir)
    rf_model_path = model_dir / "hearhere_rf.joblib"
    rf_metadata_path = model_dir / "model_metadata.json"
    rf_metadata = json.loads(rf_metadata_path.read_text(encoding="utf-8"))
    rf_model_hash = sha256_file(rf_model_path)
    expected_rf_hash = rf_metadata.get("model_sha256")
    if rf_metadata.get("model_file") != "hearhere_rf.joblib":
        expected_rf_hash = rf_metadata.get("training", {}).get(
            "random_forest_baseline", {}
        ).get("model_sha256")
    if rf_model_hash != expected_rf_hash:
        raise ValueError("Existing Random Forest model hash does not match its metadata")
    rf_model = joblib.load(rf_model_path)
    if set(rf_model.classes_) != set(CLASSES):
        raise ValueError("Existing Random Forest does not contain all configured classes")
    rf_scores = rf_model.predict_proba(rf_validation_features)
    rf_predictions = rf_model.predict(rf_validation_features)
    rf_metrics, rf_matrix = build_validation_metrics(
        y_validation.tolist(), rf_predictions.tolist()
    )
    rf_thresholds = select_alert_thresholds(y_validation, rf_scores, rf_model.classes_)

    print("Loading frozen PANNs CNN14 on CPU")
    panns_extractor = PannsEmbeddingExtractor(DEFAULT_CACHE_PATH)
    print("Extracting PANNs embeddings from folds 1-4")
    panns_train_features = extract_embedding_matrix(train_rows, audio_dir, panns_extractor)
    panns_validation_features = extract_embedding_matrix(
        validation_rows, audio_dir, panns_extractor
    )
    panns_model = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(**LOGISTIC_REGRESSION_PARAMETERS)),
    ])
    panns_model.fit(panns_train_features, y_train)
    panns_scores = panns_model.predict_proba(panns_validation_features)
    panns_predictions = panns_model.predict(panns_validation_features)
    panns_metrics, panns_matrix = build_validation_metrics(
        y_validation.tolist(), panns_predictions.tolist()
    )
    panns_thresholds = select_alert_thresholds(
        y_validation, panns_scores, panns_model.classes_
    )

    def comparison_record(predictions, metrics, thresholds, model_name):
        per_alert = {}
        alert_recalls = []
        for label in ALERT_CLASSES:
            result = thresholds.get(label)
            alert_recalls.append(float(result["recall"]) if result else 0.0)
            per_alert[label] = result or {
                "enabled": False,
                "reason": "No Fold 4 score threshold reaches 85% precision",
                "support": int(np.sum(y_validation == label)),
                "recall": 0.0,
            }
            if result:
                per_alert[label]["enabled"] = True
        return {
            "model": model_name,
            "summary": {
                key: metrics[key]
                for key in (
                    "accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_precision", "weighted_recall", "weighted_f1",
                )
            },
            "macro_f1": metrics["macro_f1"],
            "mean_alert_recall": float(np.mean(alert_recalls)),
            "alert_classes": per_alert,
            "per_class": metrics["per_class"],
            "confusion_matrix": (
                rf_matrix if model_name == "random_forest" else panns_matrix
            ).tolist(),
            "top_class_predictions": predictions.tolist(),
        }

    candidate_results = {
        "random_forest": comparison_record(
            rf_predictions, rf_metrics, rf_thresholds, "random_forest"
        ),
        "panns_cnn14_logistic_regression": comparison_record(
            panns_predictions, panns_metrics, panns_thresholds,
            "panns_cnn14_logistic_regression",
        ),
    }
    winner = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["mean_alert_recall"],
            candidate_results[name]["macro_f1"],
            name == "random_forest",
        ),
    )
    selected_model = rf_model if winner == "random_forest" else panns_model
    selected_scores = rf_scores if winner == "random_forest" else panns_scores
    selected_predictions = rf_predictions if winner == "random_forest" else panns_predictions
    selected_metrics = rf_metrics if winner == "random_forest" else panns_metrics
    selected_matrix = rf_matrix if winner == "random_forest" else panns_matrix
    selected_thresholds = rf_thresholds if winner == "random_forest" else panns_thresholds
    selected_feature_type = "mfcc" if winner == "random_forest" else "panns_cnn14"
    selected_model_file = (
        "hearhere_rf.joblib" if winner == "random_forest" else "hearhere_panns_lr.joblib"
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / selected_model_file
    joblib.dump(panns_model, model_dir / "hearhere_panns_lr.joblib")
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path = report_dir / "training_manifest.csv"
    write_manifest(manifest_path, rows)

    metadata = {
        "schema_version": "1.0",
        "artifact_status": "trained_fold4_validated",
        "model_version": (
            "hearhere-rf-v2" if selected_feature_type == "mfcc" else "hearhere-panns-v2"
        ),
        "model_file": selected_model_file,
        "model_type": (
            "sklearn.ensemble.RandomForestClassifier"
            if selected_feature_type == "mfcc"
            else "sklearn.pipeline.Pipeline(StandardScaler,LogisticRegression)"
        ),
        "created_at_utc": created_at,
        "model_sha256": sha256_file(model_path),
        "class_order": list(selected_model.classes_),
        "important_classes": list(ALERT_CLASSES),
        "feature_extractor": {
            "type": selected_feature_type,
            "entrypoint": (
                "hearhere_audio.features.extract_features_from_file"
                if selected_feature_type == "mfcc"
                else "hearhere_audio.panns.extract_embedding_from_file"
            ),
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "window_seconds": WINDOW_SECONDS,
            "short_audio_policy": "zero_pad",
            "long_audio_policy": "reject",
            "n_mfcc": N_MFCC if selected_feature_type == "mfcc" else None,
            "n_fft": N_FFT if selected_feature_type == "mfcc" else None,
            "hop_length": HOP_LENGTH if selected_feature_type == "mfcc" else None,
            "n_mels": N_MELS if selected_feature_type == "mfcc" else None,
            "resample_type": RESAMPLE_TYPE,
            "feature_count": len(FEATURE_ORDER) if selected_feature_type == "mfcc" else EMBEDDING_DIM,
            "feature_order": ",".join(FEATURE_ORDER) if selected_feature_type == "mfcc" else None,
            "std_ddof": 0 if selected_feature_type == "mfcc" else None,
            "backbone": "PANNs CNN14 16 kHz AudioSet" if selected_feature_type == "panns_cnn14" else None,
            "checkpoint_md5": CHECKPOINT_MD5 if selected_feature_type == "panns_cnn14" else None,
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
            "random_forest_baseline": {
                "model_file": "hearhere_rf.joblib",
                "model_sha256": rf_model_hash,
                "parameters": rf_model.get_params(),
                "training_folds": list(TRAIN_FOLDS),
            },
            "panns_logistic_regression": {
                "model_file": "hearhere_panns_lr.joblib",
                "parameters": LOGISTIC_REGRESSION_PARAMETERS,
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "training_folds": list(TRAIN_FOLDS),
                "embedding_dimension": EMBEDDING_DIM,
                "checkpoint_source": CHECKPOINT_URL,
                "checkpoint_md5": CHECKPOINT_MD5,
                "checkpoint_license": "CC BY 4.0",
                "checkpoint_attribution": "PANNs by Qiuqiang Kong et al.; Zenodo record 3987831",
            },
            "selected_candidate": winner,
            "selected_parameters": (
                rf_model.get_params() if selected_feature_type == "mfcc"
                else LOGISTIC_REGRESSION_PARAMETERS
            ),
            "random_seed": RANDOM_SEED,
        },
        "dependency_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "soundfile": importlib.metadata.version("soundfile"),
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "torch": importlib.metadata.version("torch"),
            "torchlibrosa": importlib.metadata.version("torchlibrosa"),
        },
        "evaluation_artifacts": {
            "validation_metrics": "reports/validation_metrics.json",
            "thresholds": "models/thresholds.json",
            "final_test_metrics": "reports/test_metrics.json",
        },
        "alert_policy": {
            "minimum_precision": MINIMUM_ALERT_PRECISION,
            "selection_objective": "Maximize mean recall across all six alert classes, counting disabled classes as zero, subject to at least 85% Fold 4 precision for enabled classes",
            "selection_method": "For each alert class, maximize recall over observed Fold 4 thresholds that meet minimum precision; disable classes with no qualifying threshold",
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
        "score_interpretation": "Uncalibrated classifier score; not a real-world event probability",
        "policy_owner": "backend",
        "minimum_precision": MINIMUM_ALERT_PRECISION,
        "selection_objective": "Maximize mean recall across all six alert classes subject to at least 85% precision per enabled class",
        "selection_method": "For each class, select the observed Fold 4 score threshold with maximum recall among thresholds at or above 85% precision; disable classes with no qualifying threshold",
        "recommended_alert_classes": [label for label in ALERT_CLASSES if label in selected_thresholds],
        "disabled_alert_classes": [label for label in ALERT_CLASSES if label not in selected_thresholds],
        "minimum_scores": {
            label: selected_thresholds[label]["threshold"] for label in selected_thresholds
        },
        "validation_metrics": selected_thresholds,
        "notes": "Thresholds are selected on Fold 4 only. Use Fold 5 once for final evaluation after model lock; do not rerun or tune from it. B owns alert decisions, preferences, cooldowns, event merging, and acknowledgements. Enabled classes each meet the Fold 4 precision criterion; this does not establish real-world microphone precision.",
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
            key: selected_metrics[key]
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
        "per_class": selected_metrics["per_class"],
        "important_sound_classes": selected_metrics["important_sound_classes"],
        "confusion_matrix": selected_matrix.tolist(),
        "confusion_matrix_axis_order": "rows=true_class,columns=predicted_class; both follow class_order",
        "thresholds": threshold_report["minimum_scores"],
        "threshold_note": "Per-class thresholds selected on Fold 4 for at least 85% precision; classes without a qualifying threshold are disabled by default.",
        "model_comparison": candidate_results,
        "selected_candidate": winner,
    }

    write_json(model_dir / "model_metadata.json", metadata)
    write_json(model_dir / "thresholds.json", threshold_report)
    write_json(report_dir / "validation_metrics.json", validation_report)
    write_confusion_matrix(report_dir / "confusion_matrix.csv", selected_matrix)
    with (report_dir / "fold4_predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["filename", "true_class", "predicted_class", *selected_model.classes_])
        for row, true_label, predicted_label, scores in zip(
            validation_rows, y_validation, selected_predictions, selected_scores
        ):
            writer.writerow(
                [row["filename"], true_label, predicted_label, *[float(score) for score in scores]]
            )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {model_dir / 'model_metadata.json'}")
    print(f"Saved threshold recommendations: {model_dir / 'thresholds.json'}")
    print(f"Saved validation report: {report_dir / 'validation_metrics.json'}")
    print(f"Saved used-data manifest: {manifest_path}")
    print(f"Selected candidate: {winner}")
    print(f"Mean alert recall: {candidate_results[winner]['mean_alert_recall']:.3f}")
    print(f"Macro F1: {selected_metrics['macro_f1']:.3f}")
    print(json.dumps(threshold_report, indent=2))
    print(json.dumps(selected_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    run_training(args.dataset.resolve(), args.model_dir.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
