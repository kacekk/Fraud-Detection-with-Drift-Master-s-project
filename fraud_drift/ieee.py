"""IEEE-CIS Fraud Detection data (Vesta Corporation, Kaggle 2019).

Real card-not-present (e-commerce) transactions: 590,540 labelled transactions,
3.5% fraud, covering about six months. Download it yourself (Kaggle login and
accepting the competition rules are required):

    https://www.kaggle.com/competitions/ieee-fraud-detection/data

and put the files here:

    data/ieee-cis/train_transaction.csv      (required)
    data/ieee-cis/train_identity.csv         (optional, adds device information)

Only the labelled training files are used; Kaggle's test files have no labels.

Time: `TransactionDT` is seconds from an undisclosed reference point (it starts
at 86,400 = day 1). We cut the timeline into equal periods (default 14 days,
giving 13 periods). A last period shorter than half a period is merged into the
one before it, so no period is tiny.

Features: a compact, documented subset of about 50 columns instead of all 434,
so the experiments run in minutes on a laptop. Missing values become -1;
categories become integer codes (sorted, so the encoding never depends on the
order of rows) and small categoricals are also one-hot encoded. None of this
uses the labels, so there is no information leak from the future.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "ieee-cis"
SECONDS_PER_DAY = 86_400

NUMERIC = (
    ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + ["D1", "D2", "D3", "D4", "D10", "D11", "D15"]
)
CATEGORICAL_ONEHOT = ["ProductCD", "card4", "card6", "M4"]   # few distinct values
CATEGORICAL_CODES = ["P_emaildomain", "R_emaildomain"]       # many distinct values
BINARY_TF = ["M1", "M2", "M3", "M5", "M6"]                   # 'T' / 'F' / missing
IDENTITY = ["DeviceType"]

DESCRIPTION = (
    "IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019): 590,540 real e-commerce card "
    "transactions with labels, about 3.5% fraud, covering roughly six months. Most columns are "
    "anonymised by the provider. The drift here is natural, so its timing is unknown."
)


def find_files(data_dir: str | Path = DEFAULT_DIR) -> tuple[Path | None, Path | None]:
    d = Path(data_dir)
    tx = next((p for p in [d / "train_transaction.csv", d / "train_transaction.parquet"] if p.exists()), None)
    ident = next((p for p in [d / "train_identity.csv", d / "train_identity.parquet"] if p.exists()), None)
    return tx, ident


def is_available(data_dir: str | Path = DEFAULT_DIR) -> bool:
    return find_files(data_dir)[0] is not None


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    header = pd.read_csv(path, nrows=0).columns
    use = [c for c in columns if c in header] if columns else None
    return pd.read_csv(path, usecols=use, low_memory=False)


def assign_periods(transaction_dt: pd.Series, period_days: int = 14) -> pd.Series:
    """Equal-length periods from the start of the data; a short tail joins the previous period."""
    days = (transaction_dt - transaction_dt.min()) / SECONDS_PER_DAY
    period = np.floor(days / period_days).astype(int)
    last = period.max()
    tail_days = days.max() - last * period_days
    if last > 0 and tail_days < period_days / 2:
        period = period.where(period < last, last - 1)
    return period.astype(int)


def prepare(tx: pd.DataFrame, identity: pd.DataFrame | None = None, period_days: int = 14
            ) -> tuple[pd.DataFrame, list[str]]:
    """Turn raw IEEE-CIS tables into the stream format used by the strategies."""
    df = tx.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    if identity is not None and "DeviceType" in identity:
        df = df.merge(identity[["TransactionID", "DeviceType"]], on="TransactionID", how="left")

    out = pd.DataFrame({"month": assign_periods(df["TransactionDT"], period_days)})
    out["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(int)   # hour of day (relative clock)

    features = ["hour"]
    for c in NUMERIC:
        if c in df:
            out[c] = df[c].astype(float).fillna(-1.0)
            features.append(c)
    for c in BINARY_TF:
        if c in df:
            out[c] = df[c].map({"T": 1, "F": 0}).fillna(-1).astype(int)
            features.append(c)
    for c in CATEGORICAL_CODES:
        if c in df:
            cats = sorted(df[c].dropna().unique())
            out[c] = pd.Categorical(df[c], categories=cats).codes.astype(int)   # missing -> -1
            features.append(c)
    onehot_cols = CATEGORICAL_ONEHOT + ([*IDENTITY] if "DeviceType" in df else [])
    for c in onehot_cols:
        if c in df:
            for v in sorted(df[c].dropna().unique()):
                name = f"{c}={v}"
                out[name] = (df[c] == v).astype(int)
                features.append(name)

    out["amount"] = df["TransactionAmt"].astype(float).fillna(0.0)   # used by the cost model
    out["is_fraud"] = df["isFraud"].astype(int)
    out["TransactionID"] = df["TransactionID"].astype(int)
    if "TransactionAmt" in features:                          # avoid a duplicate amount column
        features[features.index("TransactionAmt")] = "amount"
        out = out.drop(columns=["TransactionAmt"])
    # fixed column order: the cached parquet must give the models exactly the same feature order
    out = out[["month"] + features + ["is_fraud", "TransactionID"]]
    return out, features


def load_ieee(data_dir: str | Path = DEFAULT_DIR, period_days: int = 14, use_cache: bool = True
              ) -> tuple[pd.DataFrame, list[str]]:
    """Load IEEE-CIS as a stream. The first call caches a compact parquet file (fast reloads)."""
    data_dir = Path(data_dir)
    tx_path, id_path = find_files(data_dir)
    if tx_path is None:
        raise FileNotFoundError(
            f"No train_transaction.csv in {data_dir}. Download it from "
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data (see data/README.md)."
        )
    cache = data_dir / f"_cache_stream_{period_days}d.parquet"
    if use_cache and cache.exists() and cache.stat().st_mtime >= tx_path.stat().st_mtime:
        stream = pd.read_parquet(cache)
        return stream, [c for c in stream.columns if c not in ("month", "is_fraud", "TransactionID")]

    wanted = (["TransactionID", "TransactionDT", "isFraud"] + NUMERIC + BINARY_TF + CATEGORICAL_CODES
              + CATEGORICAL_ONEHOT)
    tx = _read(tx_path, wanted)
    identity = _read(id_path, ["TransactionID", "DeviceType"]) if id_path else None
    stream, features = prepare(tx, identity, period_days)
    if use_cache:
        try:
            stream.to_parquet(cache, index=False)
        except Exception:  # cache is only a speed-up; never fail because of it
            pass
    return stream, features
