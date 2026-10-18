"""Run the reserved ESC-50 fold 5 evaluation once on frozen artifacts."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hearhere_audio.config import ALERT_CLASSES, CLASSES
from hearhere_audio.inference import HearHereClassifier
from scripts.train import build_validation_metrics, write_confusion_matrix, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "ESC-50"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_fold5(dataset_root: Path, report_dir: Path) -> None:
    output_path = report_dir / "test_metrics.json"
    if output_path.exists():
        raise FileExistsError(
            f"Fold 5 has already been evaluated: {output_path}; do not rerun or tune from it"
        )

    model_metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    thresholds_path = PROJECT_ROOT / "models" / "thresholds.json"
    validation_path = report_dir / "validation_metrics.json"
    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_status") != "trained_fold4_validated":
        raise ValueError("Active model is not locked after Fold 4 validation")
    if thresholds.get("artifact_status") != "fold4_thresholds_selected":
        raise ValueError("Active thresholds are not locked after Fold 4 selection")
    if metadata.get("model_version") != thresholds.get("model_version"):
        raise ValueError("Model and threshold artifacts do not match")
    if validation.get("selected_candidate") != metadata["training"]["selected_candidate"]:
        raise ValueError("Fold 4 report does not match the selected model")
    model_path = model_metadata_path.parent / metadata["model_file"]
    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise ValueError("Selected model checksum does not match its metadata")

    metadata_path = dataset_root / "meta" / "esc50.csv"
    audio_dir = dataset_root / "audio"
    with metadata_path.open(newline="", encoding="utf-8") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["category"] in CLASSES and row["fold"] == "5"
        ]
    rows.sort(key=lambda row: row["filename"])
    expected = len(CLASSES) * 8
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} Fold 5 rows, found {len(rows)}")
    counts = {label: sum(row["category"] == label for row in rows) for label in CLASSES}
    if any(count != 8 for count in counts.values()):
        raise ValueError(f"Unexpected Fold 5 class counts: {counts}")

    classifier = HearHereClassifier()
    y_true = []
    y_predicted = []
    score_rows = []
    for index, row in enumerate(rows, start=1):
        result = classifier.predict_bytes((audio_dir / row["filename"]).read_bytes())
        y_true.append(row["category"])
        y_predicted.append(result["label"])
        score_rows.append(result["scores"])
        if index % 16 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)} Fold 5 WAV files")

    metrics, matrix = build_validation_metrics(y_true, y_predicted)
    fixed_threshold_metrics = {}
    for label in thresholds["recommended_alert_classes"]:
        threshold = thresholds["minimum_scores"][label]
        is_alert = np.asarray([scores[label] >= threshold for scores in score_rows])
        is_positive = np.asarray(y_true) == label
        tp = int(np.logical_and(is_alert, is_positive).sum())
        fp = int(np.logical_and(is_alert, ~is_positive).sum())
        fn = int(np.logical_and(~is_alert, is_positive).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fixed_threshold_metrics[label] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(is_positive.sum()),
            "alert_count": int(is_alert.sum()),
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
        }

    evaluated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schema_version": "1.0",
        "artifact_status": "final_fold5_evaluation",
        "model_version": metadata["model_version"],
        "selected_candidate": metadata["training"]["selected_candidate"],
        "model_sha256": metadata["model_sha256"],
        "thresholds_sha256": sha256_file(thresholds_path),
        "evaluation": {
            "dataset": "ESC-50",
            "dataset_metadata_sha256": sha256_file(metadata_path),
            "split": "reserved_final_test",
            "folds": [5],
            "sample_count": len(rows),
            "class_counts": counts,
            "evaluated_at_utc": evaluated_at,
        },
        "class_order": list(CLASSES),
        "summary": {
            key: metrics[key]
            for key in (
                "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",
            )
        },
        "per_class": metrics["per_class"],
        "important_sound_classes": {
            label: metrics["per_class"][label] for label in ALERT_CLASSES
        },
        "fixed_fold4_threshold_metrics": fixed_threshold_metrics,
        "disabled_alert_classes": thresholds["disabled_alert_classes"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axis_order": "rows=true_class,columns=predicted_class; both follow class_order",
        "notes": "This report is a single final Fold 5 evaluation. No model or threshold selection used Fold 5. Each class has eight examples; results do not establish real-world microphone accuracy.",
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    write_confusion_matrix(report_dir / "test_confusion_matrix.csv", matrix)
    with (report_dir / "fold5_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["filename", "true_class", "predicted_class", *classifier.model.classes_])
        for row, actual, predicted, scores in zip(rows, y_true, y_predicted, score_rows):
            writer.writerow([row["filename"], actual, predicted, *[scores[c] for c in classifier.model.classes_]])
    print(f"Saved final Fold 5 report: {output_path}")
    print(json.dumps(report["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    run_fold5(args.dataset.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
