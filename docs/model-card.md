# Model Card: Fraud / Transaction Risk Detection

## Summary

An educational binary classifier that ranks card transactions for a trained analyst's review. It is not designed to decide whether a person committed fraud, block a card, or decline a transaction.

## Intended use

- Demonstrate imbalanced classification, time-aware validation, threshold selection, and business-aware evaluation.
- Create a ranked human-review queue for controlled educational demonstrations using the bundled benchmark schema.

## Out-of-scope use

- Production fraud decisions or automated account/card actions.
- Customer accusation, account denial, or adverse action based solely on a model score.
- Use of scores as calibrated probabilities or direct estimates of financial loss.
- Claims about performance on current customers or contemporary transactions.

## Data and features

The model uses the ULB / Worldline European card-transaction benchmark mirrored publicly as a CSV. The 284,807 rows contain 492 fraud labels. Predictors include `Time`, `Amount`, and anonymized PCA features `V1`–`V28`; the target is `Class`. The dataset is historical (September 2013, approximately two days) and has no cardholder identifiers or full business context. See [`dataset-card.md`](dataset-card.md).

## Training and evaluation

- Preprocessing is in a scikit-learn pipeline with median imputation and standardization.
- Baseline: Logistic Regression with `class_weight='balanced'`.
- Candidate: Random Forest with `class_weight='balanced_subsample'`.
- Split: chronological by unique `Time` values; identical timestamps remain in one partition. Approximate sizes are 70% train, 15% validation, and 15% test.
- Model selection: higher validation Average Precision (AP).
- Threshold selection: validation threshold chosen for recall at least 70%, with precision maximized among feasible points. The selected threshold is `0.336665`; validation recall was 78.6%.
- The threshold was not selected using the held-out test labels.

## Reported held-out test performance

At the frozen threshold on 42,721 later-time test records (52 fraud labels):

| Metric | Result |
|---|---:|
| Average Precision | 0.7798 |
| ROC-AUC | 0.9616 |
| Precision | 0.8478 |
| Recall | 0.7500 |
| F1 | 0.7959 |
| Confusion matrix (TN, FP, FN, TP) | (42,662, 7, 13, 39) |
| Review flags | 46 |

Average Precision is the summarization of the precision-recall curve used in the code. It is sometimes referred to informally as PR-AUC; it is not the trapezoidal PR area.

## Threshold meaning

The `0.336665` score threshold is a model-score cutoff selected for a validation recall target. It is not a probability cutoff because calibration was not performed. On the test partition it flagged 46 transactions; 39 were labeled fraud and 7 were false positives, while 13 fraud labels were missed. These counts are historical benchmark results, not operating forecasts.

## Risks and limitations

- Only 52 positive examples occur in the test partition, so precision and recall can change materially with a few cases.
- The data are old and temporally narrow; changes in fraud behavior, channels, geography, or customer base can make the model stale.
- `V1`–`V28` are anonymized components; feature importance is predictive, not causal or directly actionable.
- The model's class weighting and score distribution do not establish calibrated risk probabilities.
- No reviewer capacity, review cost, average unrecovered fraud loss, recovery rate, customer-impact estimate, or fairness audit was available.
- The model artifact uses Python pickle/joblib serialization. Load only artifacts obtained from a trusted source.

## Human oversight and monitoring needed before any operational use

Any real deployment would require fresh representative labeled data, independent validation, calibration assessment, privacy and fairness review, cost/capacity analysis, human adjudication procedures, drift monitoring, and an accountable owner. Model alerts must remain advisory and subject to independent evidence and review.
