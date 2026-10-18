import numpy as np

from hearhere_audio.config import ALERT_CLASSES, CLASSES
from scripts.train import select_alert_thresholds


def test_alert_thresholds_maximize_recall_subject_to_precision_target():
    labels = np.asarray([label for label in CLASSES for _ in range(8)])
    scores = np.full((len(labels), len(CLASSES)), 0.01, dtype=np.float64)
    class_indices = {label: index for index, label in enumerate(CLASSES)}
    door_index = class_indices["door_wood_knock"]
    door_rows = np.flatnonzero(labels == "door_wood_knock")
    scores[door_rows[:7], door_index] = 0.9
    scores[door_rows[7:], door_index] = 0.7
    negative_rows = np.flatnonzero(labels != "door_wood_knock")
    scores[negative_rows[:3], door_index] = 0.7

    thresholds = select_alert_thresholds(labels, scores, np.asarray(CLASSES))

    assert thresholds["door_wood_knock"]["threshold"] == 0.9
    assert thresholds["door_wood_knock"]["recall"] == 7 / 8
    assert thresholds["door_wood_knock"]["precision"] == 1.0
    assert "siren" not in thresholds
    assert set(thresholds).issubset(ALERT_CLASSES)


def test_alert_threshold_selection_validates_score_dimensions():
    labels = np.asarray(["car_horn"])

    try:
        select_alert_thresholds(labels, np.zeros((1, 1)), np.asarray(CLASSES))
    except ValueError as error:
        assert "one column per model class" in str(error)
    else:
        raise AssertionError("score dimension mismatch should fail")


def test_alert_threshold_selection_rejects_nonfinite_scores():
    labels = np.asarray(CLASSES)
    scores = np.zeros((len(labels), len(CLASSES)), dtype=np.float64)
    scores[0, 0] = np.nan

    try:
        select_alert_thresholds(labels, scores, np.asarray(CLASSES))
    except ValueError as error:
        assert "finite values" in str(error)
    else:
        raise AssertionError("non-finite scores should fail")
