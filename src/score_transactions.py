#!/usr/bin/env python3
"""Score a label-free CSV and rank rows for human review."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FORBIDDEN_LABEL_COLUMNS = {"Class", "Actual_Fraud", "Outcome"}


def create_review_queue(frame: pd.DataFrame, model_package: dict) -> pd.DataFrame:
    """Return a descending-score review export; reject target/outcome leakage."""
    if frame.empty:
        raise ValueError("Input CSV contains no transaction rows.")
    present_labels = FORBIDDEN_LABEL_COLUMNS.intersection(frame.columns)
    if present_labels:
        names = ", ".join(sorted(present_labels))
        raise ValueError(f"Remove label/outcome columns before scoring: {names}")

    feature_columns = model_package.get("feature_columns")
    model = model_package.get("model")
    threshold = model_package.get("threshold")
    if not feature_columns or model is None or threshold is None:
        raise ValueError("Model package must contain feature_columns, model, and threshold.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing trained feature columns: {missing}")

    probabilities = model.predict_proba(frame.loc[:, feature_columns])
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Loaded model has no positive class labelled 1.")
    scores = np.asarray(probabilities)[:, classes.index(1)]

    result = frame.copy()
    if "Input_Row" in result.columns:
        raise ValueError("Input already contains reserved column 'Input_Row'.")
    result.insert(0, "Input_Row", np.arange(1, len(result) + 1))
    result["Fraud_Risk_Score"] = scores
    result["Flag_For_Human_Review"] = scores >= float(threshold)
    result["Threshold_Used"] = float(threshold)
    return result.sort_values("Fraud_Risk_Score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Label-free transaction CSV")
    parser.add_argument("--output", default="reports/powerbi/scored_review_queue.csv")
    parser.add_argument("--model", default="models/fraud_review_model.joblib")
    args = parser.parse_args()

    package = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    queue = create_review_queue(frame, package)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)
    flagged = int(queue["Flag_For_Human_Review"].sum())
    print(f"Scored {len(queue):,} rows; {flagged:,} flagged for human review.")
    print(f"Saved ranked queue to {output.resolve()}")
    print("Review flags are not fraud findings or automatic decline decisions.")


if __name__ == "__main__":
    main()
