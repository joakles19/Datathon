import csv
import json
from collections import Counter

import numpy as np

from hearhere_audio.config import CLASSES, FEATURE_ORDER
from scripts import train


def write_synthetic_dataset(dataset_root):
    audio_dir = dataset_root / "audio"
    metadata_dir = dataset_root / "meta"
    audio_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    metadata_path = metadata_dir / "esc50.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["filename", "target", "fold", "category", "esc10", "src_file", "take"],
        )
        writer.writeheader()
        for fold in range(1, 6):
            for class_index, category in enumerate(CLASSES):
                for clip_index in range(8):
                    filename = f"{fold}-{class_index}-{clip_index}.wav"
                    writer.writerow(
                        {
                            "filename": filename,
                            "target": class_index,
                            "fold": fold,
                            "category": category,
                            "esc10": "False",
                            "src_file": f"source-{class_index}",
                            "take": clip_index,
                        }
                    )
                    (audio_dir / filename).touch()


def test_manifest_records_all_folds_without_merging_fold5_into_training(tmp_path):
    dataset_root = tmp_path / "ESC-50"
    write_synthetic_dataset(dataset_root)

    rows = train.read_rows(dataset_root)
    manifest_path = tmp_path / "training_manifest.csv"
    train.write_manifest(manifest_path, rows)

    assert len(rows) == 440
    assert Counter(int(row["fold"]) for row in rows) == {fold: 88 for fold in range(1, 6)}
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        manifest_rows = list(csv.DictReader(manifest_file))
    assert len(manifest_rows) == 440
    assert {row["split"] for row in manifest_rows if row["fold"] == "5"} == {"test"}
    assert {row["split"] for row in manifest_rows if row["fold"] in {"1", "2", "3"}} == {"train"}
    assert {row["split"] for row in manifest_rows if row["fold"] == "4"} == {"validation"}


def test_training_feature_extraction_excludes_fold5(tmp_path, monkeypatch):
    dataset_root = tmp_path / "ESC-50"
    write_synthetic_dataset(dataset_root)

    extracted_fold_sets = []

    def fake_extract_matrix(rows, _audio_dir):
        folds = {int(row["fold"]) for row in rows}
        extracted_fold_sets.append(folds)
        return np.stack(
            [
                np.full(len(FEATURE_ORDER), CLASSES.index(row["category"]), dtype=np.float32)
                for row in rows
            ]
        )

    monkeypatch.setattr(train, "extract_matrix", fake_extract_matrix)
    train.run_training(
        dataset_root,
        tmp_path / "models",
        tmp_path / "reports",
    )

    assert extracted_fold_sets == [{1, 2, 3}, {4}]
    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert metadata["dataset"]["sample_counts"] == {
        "train": 264,
        "validation": 88,
        "final_test": 88,
    }
