import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]


def test_published_test_metrics_match_prediction_export():
    predictions = pd.read_csv(ROOT / "reports/powerbi/test_evaluation_predictions.csv")
    reported = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    y_true = predictions["Actual_Fraud"]
    scores = predictions["Fraud_Risk_Score"]
    flags = predictions["Flag_For_Human_Review"]

    assert len(predictions) == reported["split_rows"]["test"]
    assert int(flags.sum()) == reported["flagged_count"]
    assert confusion_matrix(y_true, flags, labels=[0, 1]).ravel().tolist() == reported[
        "confusion_matrix_tn_fp_fn_tp"
    ]
    assert precision_score(y_true, flags) == pytest.approx(reported["precision"])
    assert recall_score(y_true, flags) == pytest.approx(reported["recall"])
    assert f1_score(y_true, flags) == pytest.approx(reported["f1"])
    assert average_precision_score(y_true, scores) == pytest.approx(
        reported["average_precision_pr_auc"]
    )
    assert roc_auc_score(y_true, scores) == pytest.approx(reported["roc_auc"])
