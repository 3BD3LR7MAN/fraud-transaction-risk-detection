import numpy as np
import pandas as pd
import pytest

from src.score_transactions import create_review_queue
from src.train_fraud import choose_threshold, metrics_at, read_dataset


class StubModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        positive = frame["risk_feature"].to_numpy(dtype=float)
        return np.column_stack([1 - positive, positive])


def test_metrics_at_threshold_returns_expected_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.4])

    result = metrics_at(y_true, scores, threshold=0.5)

    assert result["confusion_matrix_tn_fp_fn_tp"] == [1, 1, 1, 1]
    assert result["flagged_count"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_threshold_is_chosen_on_target_recall_constraint():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.4])

    threshold, validation_metrics, target_met = choose_threshold(
        y_true, scores, target_recall=0.5
    )

    assert target_met is True
    assert threshold == pytest.approx(0.8)
    assert validation_metrics["recall"] >= 0.5
    assert validation_metrics["precision"] == pytest.approx(1.0)


def test_read_dataset_accepts_binary_class_csv(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"Time": [1, 2], "Amount": [4.0, 5.0], "Class": [0, 1]}).to_csv(
        path, index=False
    )

    frame = read_dataset(path)

    assert frame["Class"].tolist() == [0, 1]
    assert frame["Class"].dtype.kind in "iu"


def test_review_queue_sorts_scores_and_blocks_label_leakage():
    frame = pd.DataFrame({"risk_feature": [0.2, 0.9, 0.6], "Amount": [10, 20, 30]})
    package = {"model": StubModel(), "threshold": 0.5, "feature_columns": ["risk_feature"]}

    queue = create_review_queue(frame, package)

    assert queue["Input_Row"].tolist() == [2, 3, 1]
    assert queue["Flag_For_Human_Review"].tolist() == [True, True, False]
    assert queue["Fraud_Risk_Score"].is_monotonic_decreasing
    assert not ({"Class", "Actual_Fraud", "Outcome"} & set(queue.columns))

    with pytest.raises(ValueError, match="Remove label/outcome columns"):
        create_review_queue(frame.assign(Class=[0, 1, 0]), package)
