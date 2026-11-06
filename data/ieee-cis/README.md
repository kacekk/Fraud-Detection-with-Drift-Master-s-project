# IEEE-CIS Fraud Detection data goes here

1. Sign in to Kaggle and open <https://www.kaggle.com/competitions/ieee-fraud-detection/data>
2. Accept the competition rules (Kaggle asks once; the data may be used for non-commercial and academic purposes under those rules).
3. Download `train_transaction.csv` (required) and `train_identity.csv` (optional) and put them in this folder:

```
data/ieee-cis/train_transaction.csv
data/ieee-cis/train_identity.csv
```

The files are not stored on GitHub (about 700 MB, and the competition rules do not allow redistribution).
The first load converts them into a small cached file (`_cache_stream_14d.parquet`) so later runs start fast.
