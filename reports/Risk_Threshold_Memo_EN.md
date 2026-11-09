# Risk Threshold Memo

**Purpose:** prioritize transactions for trained human review. This demonstration does not make automatic fraud or decline decisions.

**Selected rule:** flag for review when model score ≥ **0.336665**. It was selected on validation data to maximize precision among thresholds achieving at least 70% recall. Validation goal met: **True**.

**Held-out test:** 46 of 42,721 transactions flagged; precision 84.783%; recall 75.000%; false positives 7; missed fraud labels 13.

See `threshold_tradeoffs.csv` for alternate thresholds. Raising the threshold generally reduces review volume and false positives but can miss more fraud; lowering it generally catches more fraud while requiring more manual review. Real costs and review capacity were not provided, so no threshold is represented as financially optimal.

**Cost worksheet:** for a candidate threshold, calculate `FP × cost_per_review + FN × expected_unrecovered_loss_per_missed_fraud`. Estimate these inputs with finance and operations, and run sensitivity analysis. Transaction `Amount` is not itself the unrecovered loss.

**Guardrail:** scores only order a review queue and are not claimed to be calibrated probabilities. A human analyst must verify context; the demo must not automatically decline or block a customer's transaction.
