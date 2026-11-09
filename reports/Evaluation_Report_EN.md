# Fraud / Transaction Risk Detection — Evaluation Report

## Dataset and split

OpenML dataset 42175 is the ULB/Worldline European credit-card transaction dataset. The downloaded CSV is a public mirror of the Kaggle release [4]. It contains 284,807 transactions and 492 frauds (0.173%). Inputs are anonymized PCA features V1–V28 plus elapsed Time and transaction Amount. This is a historical two-day dataset, not a live operational feed.[1] [2]

Chronological split by unique Time groups (identical Time values remain together): train 199,364 rows (384 frauds), validation 42,722 rows (56 frauds), test 42,721 rows (52 frauds). The test set was held out until model/threshold selection was complete.

## Models and validation selection

The baseline is class-weighted Logistic Regression. The improved comparison is a class-weighted Random Forest. Both use imputation and standardization in a pipeline. Model selection uses validation Average Precision / PR-AUC, which is more informative than accuracy for this extreme imbalance; the dataset source also recommends AUPRC.[1]

- Logistic Regression validation PR-AUC: **0.8394**
- Random Forest validation PR-AUC: **0.8727**
- Selected model: **Random Forest**
- Review threshold selected on validation for recall ≥ 70%: **0.336665** (goal met on validation: **True**)

## Final held-out test results

- Precision: **0.8478**
- Recall: **0.7500**
- F1: **0.7959**
- Average Precision / PR-AUC: **0.7798**
- ROC-AUC: **0.9616**
- Flagged for human review: **46 / 42,721**
- Confusion counts: TN=42,662, FP=7, FN=13, TP=39

The threshold is a ranking-to-review policy, not an automatic fraud finding or card-decline rule. The model score is not claimed to be a calibrated probability. Test metrics may differ from validation because transaction patterns and fraud prevalence vary over time.

## Imbalance, false positives, and operational cost

Only 0.173% of the historical transactions are labeled fraud. Accuracy is therefore not a meaningful success criterion by itself. `threshold_tradeoffs.csv` reports the number of alerts, false positives sent to review, missed frauds, precision, recall, and F1 at several candidate thresholds on both validation and test data.

No review cost, fraud loss, recovery rate, or reviewer capacity was supplied. Therefore the project does **not** claim a financially optimal threshold. For a real operation, estimate expected cost as `false_positive_reviews × review_cost + missed_frauds × average_unrecovered_fraud_loss`, then select the operating point subject to analyst capacity and customer-impact constraints. The `Amount` feature is transaction value, not an observed loss or recoverable amount; it must not be treated as actual fraud loss without additional evidence.

## Human review and model explanation

`blinded_review_queue_demo.csv` contains only transactions flagged by the model, sorted by `Fraud_Risk_Score`, and deliberately excludes the true label and outcome so a reviewer cannot see the answer. `test_evaluation_predictions.csv` contains test truth for retrospective metrics only and must not be used as an operational review queue. A trained analyst should inspect alerts and use independent evidence before action. The model must not autonomously accuse a customer, block a card, or deny a transaction based on this demo.

Top validation permutation-importance features: V4 (0.089), V14 (0.040), V12 (0.033), V3 (0.020), V11 (0.019). They indicate predictive contribution on a validation sample and are not causal explanations. V1–V28 are anonymized components, so business meaning cannot be assigned to an individual component from this dataset.

## Limitations

The data represent two days in September 2013 and include no cardholder identifiers or full business context. The chronological holdout is preferable to a random split for checking temporal drift, but it still cannot establish performance on today's population. Before production use, validate on recent labeled transactions, calibrate scores, estimate costs with business owners, assess privacy/fairness, and monitor drift and reviewer outcomes.

## References

[1]: https://www.openml.org/d/42175 "OpenML CreditCardFraudDetection dataset"
[2]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "ULB and Worldline Credit Card Fraud Detection dataset on Kaggle"
[3]: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Baseline_RealWorldData.html "Fraud Detection Handbook real-world baseline"
[4]: https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection "Public CSV mirror of the Kaggle ULB dataset"
