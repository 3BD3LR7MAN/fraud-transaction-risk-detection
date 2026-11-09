#!/usr/bin/env python3
"""Train and evaluate a fraud-review prioritization pipeline on OpenML 42175."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_RECALL = 0.70


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        data, _meta = arff.loadarff(path)
        frame = pd.DataFrame(data)
    for col in frame.select_dtypes(include=[object]).columns:
        frame[col] = frame[col].map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    if "Class" not in frame.columns:
        raise ValueError(f"ARFF file has no Class target; columns: {list(frame.columns)}")
    frame["Class"] = pd.to_numeric(frame["Class"], errors="raise").astype(int)
    if set(frame["Class"].unique()) - {0, 1}:
        raise ValueError("Expected binary Class labels 0/1")
    return frame


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def metrics_at(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "flagged_count": int(pred.sum()),
        "fraud_count": int(np.asarray(y_true).sum()),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).ravel().tolist(),
    }


def choose_threshold(y_true, probs, target_recall=TARGET_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve returns one more P/R pair than thresholds.
    p, r, t = precision[:-1], recall[:-1], thresholds
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    feasible = np.flatnonzero(r >= target_recall)
    target_met = len(feasible) > 0
    if target_met:
        best_idx = max(feasible, key=lambda i: (p[i], f1[i], t[i]))
    else:
        best_idx = max(range(len(t)), key=lambda i: (r[i], p[i], f1[i]))
    chosen = float(t[best_idx])
    pred = (np.asarray(probs) >= chosen).astype(int)
    table = pd.DataFrame([{
        "threshold": chosen,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flagged_count": int(pred.sum()),
        "false_positive_count": int(((np.asarray(y_true) == 0) & (pred == 1)).sum()),
        "false_negative_count": int(((np.asarray(y_true) == 1) & (pred == 0)).sum()),
    }])
    feasible = table if target_met else pd.DataFrame()
    if not target_met:
        # The fallback threshold is already selected by maximum attainable recall.
        met = False
        best = table.iloc[0]
        return chosen, best.to_dict(), met
    if feasible.empty:
        best = table.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0]
        met = False
    else:
        best = feasible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
        met = target_met
    return float(best.threshold), best.to_dict(), met


def threshold_scenarios(y_true, probs, thresholds):
    rows = []
    for threshold in thresholds:
        row = metrics_at(y_true, probs, float(threshold))
        tn, fp, fn, tp = row["confusion_matrix_tn_fp_fn_tp"]
        rows.append({
            "threshold": row["threshold"], "precision": row["precision"],
            "recall": row["recall"], "f1": row["f1"],
            "flagged_count": row["flagged_count"],
            "false_positives_for_review": fp, "missed_frauds": fn,
            "true_frauds_flagged": tp, "true_negatives_not_flagged": tn,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='OpenML 42175 CSV or ARFF file')
    parser.add_argument('--output', required=True, help='Project directory')
    parser.add_argument('--target-recall', type=float, default=TARGET_RECALL)
    parser.add_argument('--rf-trees', type=int, default=160)
    args = parser.parse_args()

    root = Path(args.output).resolve()
    report_dir = root / 'reports'
    figure_dir = report_dir / 'figures'
    model_dir = root / 'models'
    powerbi_dir = report_dir / 'powerbi'
    for d in [report_dir, figure_dir, model_dir, powerbi_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = read_dataset(Path(args.data))
    df = df.dropna(subset=['Class']).copy()
    # Preserve transaction order: the source's Time column is elapsed seconds from the beginning.
    df = df.sort_values('Time', kind='stable').reset_index(drop=True)
    y = df['Class'].astype(int)
    X = df.drop(columns=['Class'])
    n = len(df)
    unique_times, counts_by_time = np.unique(df['Time'].to_numpy(), return_counts=True)
    cumulative = np.cumsum(counts_by_time)
    train_candidates = np.arange(0, max(1, len(unique_times) - 2))
    train_group = int(train_candidates[np.argmin(np.abs(cumulative[train_candidates] - 0.70 * n))])
    validation_candidates = np.arange(train_group + 1, max(train_group + 2, len(unique_times) - 1))
    if len(validation_candidates) == 0:
        raise ValueError('Not enough unique Time values for three chronological partitions.')
    validation_group = int(validation_candidates[np.argmin(np.abs(cumulative[validation_candidates] - 0.85 * n))])
    train_cutoff = unique_times[train_group]
    validation_cutoff = unique_times[validation_group]
    time_values = df['Time'].to_numpy()
    train_mask = time_values <= train_cutoff
    validation_mask = (time_values > train_cutoff) & (time_values <= validation_cutoff)
    test_mask = time_values > validation_cutoff
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if not (X_train['Time'].max() < X_val['Time'].min() and X_val['Time'].max() < X_test['Time'].min()):
        raise ValueError('Chronological split boundaries overlap; inspect Time values.')
    if min(y_train.sum(), y_val.sum(), y_test.sum()) == 0:
        raise ValueError('A chronological split has no fraud cases in a partition; use a different documented split strategy.')

    # EDA: label prevalence, amount distribution, chronological fraud counts.
    sns.set_theme(style='whitegrid')
    counts = df['Class'].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(['Legitimate', 'Fraud'], [int(counts.get(0, 0)), int(counts.get(1, 0))], color=['#4C78A8', '#E45756'])
    axes[0].set_yscale('log')
    axes[0].set_title('Class counts (log scale)')
    axes[0].set_ylabel('Transactions')
    sns.histplot(data=df, x='Amount', hue='Class', bins=60, log_scale=(True, False), element='step', stat='density', common_norm=False, ax=axes[1])
    axes[1].set_title('Transaction amounts (log x scale)')
    fig.tight_layout()
    fig.savefig(figure_dir / 'eda_imbalance_amount.png', dpi=170)
    plt.close(fig)

    # Daily/chronological bins in elapsed time; not calendar dates.
    df['time_bin'] = pd.cut(df['Time'], bins=24, include_lowest=True)
    temporal = df.groupby('time_bin', observed=True).agg(transactions=('Class', 'size'), frauds=('Class', 'sum')).reset_index()
    temporal['fraud_rate'] = temporal['frauds'] / temporal['transactions']
    temporal.to_csv(report_dir / 'time_bin_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(temporal)), temporal['fraud_rate'] * 100, marker='o', color='#E45756')
    ax.set(title='Fraud rate across 24 chronological bins', xlabel='Time bin (ordered)', ylabel='Fraud rate (%)')
    fig.tight_layout()
    fig.savefig(figure_dir / 'fraud_rate_over_time.png', dpi=170)
    plt.close(fig)

    baseline = make_pipeline(LogisticRegression(
        max_iter=1500, class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE
    ))
    rf = make_pipeline(RandomForestClassifier(
        n_estimators=args.rf_trees, max_depth=14, min_samples_leaf=3,
        max_features='sqrt', class_weight='balanced_subsample',
        n_jobs=-1, random_state=RANDOM_STATE
    ))
    baseline.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    val_probs = {
        'Logistic Regression baseline': baseline.predict_proba(X_val)[:, 1],
        'Random Forest': rf.predict_proba(X_val)[:, 1],
    }
    val_ap = {name: float(average_precision_score(y_val, p)) for name, p in val_probs.items()}
    chosen_name = max(val_ap, key=val_ap.get)
    models = {'Logistic Regression baseline': baseline, 'Random Forest': rf}
    selected_model = models[chosen_name]
    selected_val_probs = val_probs[chosen_name]
    threshold, threshold_row, recall_goal_met = choose_threshold(y_val, selected_val_probs, args.target_recall)

    test_probs = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_probs, threshold)
    test_pred = (test_probs >= threshold).astype(int)
    test_metrics.update({
        'selected_model': chosen_name,
        'target_recall_validation': float(args.target_recall),
        'target_recall_met_on_validation': bool(recall_goal_met),
        'validation_selected_threshold_metrics': {k: float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v for k, v in threshold_row.items()},
        'validation_pr_auc': val_ap,
        'dataset_id': 42175,
        'rows': int(len(df)),
        'fraud_rows': int(y.sum()),
        'fraud_rate': float(y.mean()),
        'split_method': 'chronological by unique Time groups: approximately 70% train, 15% validation, final 15% test; identical Time values stay in one partition',
        'time_cutoffs': {'train_through': float(train_cutoff), 'validation_through': float(validation_cutoff)},
        'split_rows': {'train': int(len(X_train)), 'validation': int(len(X_val)), 'test': int(len(X_test))},
        'split_frauds': {'train': int(y_train.sum()), 'validation': int(y_val.sum()), 'test': int(y_test.sum())},
    })
    (report_dir / 'metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    pd.DataFrame([
        {'model': name, 'validation_average_precision_pr_auc': ap}
        for name, ap in val_ap.items()
    ]).to_csv(report_dir / 'model_comparison.csv', index=False)
    pd.DataFrame(classification_report(y_test, test_pred, labels=[0, 1], target_names=['Legitimate', 'Fraud'], output_dict=True, zero_division=0)).transpose().to_csv(report_dir / 'classification_report.csv')

    # Chosen operating point and pre-declared fixed thresholds for comparison.
    selected_thresholds = sorted(set([0.01, 0.05, 0.10, 0.20, 0.50, threshold]))
    val_scenarios = threshold_scenarios(y_val, selected_val_probs, selected_thresholds)
    val_scenarios.insert(0, 'split', 'validation')
    test_scenarios = threshold_scenarios(y_test, test_probs, selected_thresholds)
    test_scenarios.insert(0, 'split', 'test')
    pd.concat([val_scenarios, test_scenarios], ignore_index=True).to_csv(report_dir / 'threshold_tradeoffs.csv', index=False)

    # Separate retrospective test truth from a blinded, label-free review-queue demonstration.
    evaluation = X_test.copy()
    evaluation.insert(0, 'row_id', X_test.index.to_numpy())
    evaluation['Actual_Fraud'] = y_test.to_numpy()
    evaluation['Fraud_Risk_Score'] = test_probs
    evaluation['Flag_For_Human_Review'] = test_pred
    evaluation['Threshold_Used'] = threshold
    evaluation['Outcome'] = np.select(
        [(y_test.to_numpy() == 1) & (test_pred == 1), (y_test.to_numpy() == 0) & (test_pred == 1), (y_test.to_numpy() == 1) & (test_pred == 0)],
        ['TP', 'FP', 'FN'], default='TN'
    )
    evaluation.sort_values('Fraud_Risk_Score', ascending=False).to_csv(powerbi_dir / 'test_evaluation_predictions.csv', index=False)
    review_queue = X_test.loc[test_pred == 1].copy()
    review_queue.insert(0, 'row_id', X_test.index.to_numpy()[test_pred == 1])
    review_queue['Fraud_Risk_Score'] = test_probs[test_pred == 1]
    review_queue['Threshold_Used'] = threshold
    review_queue.sort_values('Fraud_Risk_Score', ascending=False).to_csv(powerbi_dir / 'blinded_review_queue_demo.csv', index=False)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    cm_rows = []
    for actual in [0, 1]:
        for predicted in [0, 1]:
            cm_rows.append({'Actual_Fraud': actual, 'Flag_For_Review': predicted, 'Count': int(((y_test.to_numpy() == actual) & (test_pred == predicted)).sum())})
    pd.DataFrame(cm_rows).to_csv(powerbi_dir / 'confusion_matrix.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not flagged', 'Flagged'], yticklabels=['Legitimate', 'Fraud'], ax=ax)
    ax.set(title=f'Test set: human-review flags (threshold={threshold:.4f})', xlabel='Model flag', ylabel='Actual class')
    fig.tight_layout()
    fig.savefig(figure_dir / 'test_confusion_matrix.png', dpi=170)
    plt.close(fig)

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(pr_recall, pr_precision, color='#4C78A8', label=f'AP = {test_metrics["average_precision_pr_auc"]:.3f}')
    ax.axhline(float(y_test.mean()), color='#E45756', linestyle='--', label=f'Fraud prevalence = {y_test.mean():.3%}')
    ax.set(title='Precision–Recall curve on chronological test set', xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(figure_dir / 'test_precision_recall_curve.png', dpi=170)
    plt.close(fig)

    # Use a reproducible validation sample to keep permutation importance affordable.
    sample_n = min(10000, len(X_val))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_val), size=sample_n, replace=False)
    X_imp = X_val.iloc[sample_idx]
    y_imp = y_val.iloc[sample_idx]
    perm = permutation_importance(selected_model, X_imp, y_imp, scoring='average_precision', n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    importances = pd.DataFrame({'feature': X.columns, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std}).sort_values('importance_mean', ascending=False)
    importances.to_csv(report_dir / 'permutation_importance.csv', index=False)
    top = importances.head(15).sort_values('importance_mean')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.feature, top.importance_mean, xerr=top.importance_std, color='#4C78A8')
    ax.set(title='Permutation importance on validation sample', xlabel='Decrease in average precision when shuffled')
    fig.tight_layout()
    fig.savefig(figure_dir / 'permutation_importance.png', dpi=170)
    plt.close(fig)

    package = {'model': selected_model, 'threshold': threshold, 'model_name': chosen_name, 'feature_columns': X.columns.tolist(), 'target_recall_validation': args.target_recall}
    joblib.dump(package, model_dir / 'fraud_review_model.joblib')

    cm_tn, cm_fp, cm_fn, cm_tp = cm.ravel()
    top_features = ', '.join(f"{r.feature} ({r.importance_mean:.3f})" for r in importances.head(5).itertuples())
    report = f'''# Fraud / Transaction Risk Detection — Evaluation Report

## Dataset and split

OpenML dataset 42175 is the ULB/Worldline European credit-card transaction dataset. The downloaded CSV is a public mirror of the Kaggle release [4]. It contains {len(df):,} transactions and {int(y.sum()):,} frauds ({y.mean():.3%}). Inputs are anonymized PCA features V1–V28 plus elapsed Time and transaction Amount. This is a historical two-day dataset, not a live operational feed.[1] [2]

Chronological split by unique Time groups (identical Time values remain together): train {len(X_train):,} rows ({int(y_train.sum())} frauds), validation {len(X_val):,} rows ({int(y_val.sum())} frauds), test {len(X_test):,} rows ({int(y_test.sum())} frauds). The test set was held out until model/threshold selection was complete.

## Models and validation selection

The baseline is class-weighted Logistic Regression. The improved comparison is a class-weighted Random Forest. Both use imputation and standardization in a pipeline. Model selection uses validation Average Precision / PR-AUC, which is more informative than accuracy for this extreme imbalance; the dataset source also recommends AUPRC.[1]

- Logistic Regression validation PR-AUC: **{val_ap['Logistic Regression baseline']:.4f}**
- Random Forest validation PR-AUC: **{val_ap['Random Forest']:.4f}**
- Selected model: **{chosen_name}**
- Review threshold selected on validation for recall ≥ {args.target_recall:.0%}: **{threshold:.6f}** (goal met on validation: **{recall_goal_met}**)

## Final held-out test results

- Precision: **{test_metrics['precision']:.4f}**
- Recall: **{test_metrics['recall']:.4f}**
- F1: **{test_metrics['f1']:.4f}**
- Average Precision / PR-AUC: **{test_metrics['average_precision_pr_auc']:.4f}**
- ROC-AUC: **{test_metrics['roc_auc']:.4f}**
- Flagged for human review: **{test_metrics['flagged_count']:,} / {len(X_test):,}**
- Confusion counts: TN={cm_tn:,}, FP={cm_fp:,}, FN={cm_fn:,}, TP={cm_tp:,}

The threshold is a ranking-to-review policy, not an automatic fraud finding or card-decline rule. The model score is not claimed to be a calibrated probability. Test metrics may differ from validation because transaction patterns and fraud prevalence vary over time.

## Imbalance, false positives, and operational cost

Only {y.mean():.3%} of the historical transactions are labeled fraud. Accuracy is therefore not a meaningful success criterion by itself. `threshold_tradeoffs.csv` reports the number of alerts, false positives sent to review, missed frauds, precision, recall, and F1 at several candidate thresholds on both validation and test data.

No review cost, fraud loss, recovery rate, or reviewer capacity was supplied. Therefore the project does **not** claim a financially optimal threshold. For a real operation, estimate expected cost as `false_positive_reviews × review_cost + missed_frauds × average_unrecovered_fraud_loss`, then select the operating point subject to analyst capacity and customer-impact constraints. The `Amount` feature is transaction value, not an observed loss or recoverable amount; it must not be treated as actual fraud loss without additional evidence.

## Human review and model explanation

`blinded_review_queue_demo.csv` contains only transactions flagged by the model, sorted by `Fraud_Risk_Score`, and deliberately excludes the true label and outcome so a reviewer cannot see the answer. `test_evaluation_predictions.csv` contains test truth for retrospective metrics only and must not be used as an operational review queue. A trained analyst should inspect alerts and use independent evidence before action. The model must not autonomously accuse a customer, block a card, or deny a transaction based on this demo.

Top validation permutation-importance features: {top_features}. They indicate predictive contribution on a validation sample and are not causal explanations. V1–V28 are anonymized components, so business meaning cannot be assigned to an individual component from this dataset.

## Limitations

The data represent two days in September 2013 and include no cardholder identifiers or full business context. The chronological holdout is preferable to a random split for checking temporal drift, but it still cannot establish performance on today's population. Before production use, validate on recent labeled transactions, calibrate scores, estimate costs with business owners, assess privacy/fairness, and monitor drift and reviewer outcomes.

## References

[1]: https://www.openml.org/d/42175 "OpenML CreditCardFraudDetection dataset"
[2]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "ULB and Worldline Credit Card Fraud Detection dataset on Kaggle"
[3]: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Baseline_RealWorldData.html "Fraud Detection Handbook real-world baseline"
[4]: https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection "Public CSV mirror of the Kaggle ULB dataset"
'''
    (report_dir / 'Evaluation_Report_EN.md').write_text(report, encoding='utf-8')
    risk_memo = f'''# Risk Threshold Memo

**Purpose:** prioritize transactions for trained human review. This demonstration does not make automatic fraud or decline decisions.

**Selected rule:** flag for review when model score ≥ **{threshold:.6f}**. It was selected on validation data to maximize precision among thresholds achieving at least {args.target_recall:.0%} recall. Validation goal met: **{recall_goal_met}**.

**Held-out test:** {test_metrics['flagged_count']:,} of {len(X_test):,} transactions flagged; precision {test_metrics['precision']:.3%}; recall {test_metrics['recall']:.3%}; false positives {cm_fp:,}; missed fraud labels {cm_fn:,}.

See `threshold_tradeoffs.csv` for alternate thresholds. Raising the threshold generally reduces review volume and false positives but can miss more fraud; lowering it generally catches more fraud while requiring more manual review. Real costs and review capacity were not provided, so no threshold is represented as financially optimal.

**Cost worksheet:** for a candidate threshold, calculate `FP × cost_per_review + FN × expected_unrecovered_loss_per_missed_fraud`. Estimate these inputs with finance and operations, and run sensitivity analysis. Transaction `Amount` is not itself the unrecovered loss.

**Guardrail:** scores only order a review queue and are not claimed to be calibrated probabilities. A human analyst must verify context; the demo must not automatically decline or block a customer's transaction.
'''
    (report_dir / 'Risk_Threshold_Memo_EN.md').write_text(risk_memo, encoding='utf-8')
    print(json.dumps(test_metrics, indent=2))
    print(f'Artifacts saved to {root}')


if __name__ == '__main__':
    main()
