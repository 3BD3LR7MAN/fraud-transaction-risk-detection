# Dataset Card: ULB / Worldline Credit-Card Fraud Detection

## Source and provenance

This project uses the widely circulated European card-transaction benchmark associated with ULB Machine Learning Group and Worldline. Project references include [OpenML dataset 42175](https://www.openml.org/d/42175) and the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The included downloader retrieves a CSV from a public [Hugging Face dataset mirror](https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection); it does not use the OpenML API directly.

The mirror is a third-party copy. Verify the source, terms, and integrity of any copy you use. This repository does not include or grant a license for the dataset. Check the source's terms before redistribution.

## Snapshot structure

The benchmark CSV used for this project contains 284,807 transactions with 492 positive fraud labels (approximately 0.173%). It contains:

- `Time`: elapsed seconds from the start of the dataset's observation period; not a wall-clock timestamp.
- `Amount`: transaction amount in the original dataset's units.
- `V1`–`V28`: anonymized principal-component features derived through PCA.
- `Class`: target label (`1` for fraud, `0` otherwise).

The labels and transactions represent a short historical period in September 2013, approximately two days. The data are not representative of current payment behavior by default.

## Intended project use

A teaching benchmark for demonstrating severe class imbalance, chronological splitting, ranking metrics, threshold trade-offs, and human-review prioritization. It is not a live feed and should not be treated as a production validation set.

## Known limitations

- The observation window is very short and historical.
- The anonymized PCA features do not expose their original business meaning.
- The dataset does not supply cardholder identities, reviewer decisions, chargeback recovery, confirmed loss, review cost, or customer impact.
- Class prevalence and transaction behavior may not match another institution or later period.
- `Amount` is transaction value, not measured unrecovered fraud loss.

## Download and handling

Raw data are intentionally excluded from version control because the CSV is large and has separate source terms. From the repository root, run:

```bash
python src/download_data.py --output data/raw/creditcard.csv
```

The download script checks byte-range responses and chunk sizes. These transport checks do not replace independent source verification or a cryptographic checksum.
