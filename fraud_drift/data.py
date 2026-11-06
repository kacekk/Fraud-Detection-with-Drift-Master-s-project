"""Transaction streams.

1. A synthetic stream where we decide *when* and *how* the fraud pattern
   changes (known drift point -> we can measure recovery time).
2. A loader for the public Kaggle credit-card dataset (ULB) as a realism check.

All randomness comes from one numpy Generator seeded with `cfg.seed`, so the
same config always produces the identical stream on every machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig

SYNTHETIC_FEATURES = ["amount", "hour", "n_txn_1h", "foreign", "device_new"]

PATTERNS = {
    "A": "old fraud pattern: large amounts, late at night (00:00-04:59)",
    "B": "new fraud pattern: small amounts, bursts of transactions, foreign merchant, new device",
}


def share_of_new_pattern(month: int, cfg: ExperimentConfig) -> float:
    """Share of fraud cases in a given month that follow the NEW pattern (B)."""
    if cfg.drift_type == "none" or month < cfg.drift_month:
        return 0.0
    if cfg.drift_type == "sudden":
        return 1.0
    if cfg.drift_type == "gradual":
        # linear hand-over: 1/T, 2/T, ..., 1 over `transition_months` months
        return min(1.0, (month - cfg.drift_month + 1) / cfg.transition_months)
    if cfg.drift_type == "recurring":
        return 1.0 if month < cfg.recur_month else 0.0
    raise ValueError(cfg.drift_type)


def make_month(n: int, fraud_rate: float, p_new: float, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one month of transactions.

    Legitimate behaviour never changes; only what fraud *looks like* changes
    (this is real concept drift: P(fraud | features) moves).
    """
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    hour = rng.integers(0, 24, size=n)
    n_txn_1h = rng.poisson(lam=1.5, size=n)
    foreign = rng.binomial(1, 0.15, size=n)
    device_new = rng.binomial(1, 0.20, size=n)

    n_fraud = int(round(n * fraud_rate))
    is_fraud = np.zeros(n, dtype=int)
    pattern = np.array(["-"] * n, dtype=object)

    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    is_new = rng.random(n_fraud) < p_new          # drawn every month -> stable RNG sequence
    idx_a, idx_b = fraud_idx[~is_new], fraud_idx[is_new]
    is_fraud[fraud_idx] = 1

    # Pattern A (old): large amounts, at night
    amount[idx_a] *= rng.uniform(3, 6, size=idx_a.size)
    hour[idx_a] = rng.integers(0, 5, size=idx_a.size)
    pattern[idx_a] = "A"

    # Pattern B (new): many small rapid payments, foreign merchant, new device
    amount[idx_b] *= rng.uniform(0.3, 0.7, size=idx_b.size)
    n_txn_1h[idx_b] += rng.integers(4, 9, size=idx_b.size)
    foreign[idx_b] = 1
    device_new[idx_b] = 1
    pattern[idx_b] = "B"

    return pd.DataFrame({
        "amount": np.round(amount, 2),
        "hour": hour,
        "n_txn_1h": n_txn_1h,
        "foreign": foreign,
        "device_new": device_new,
        "is_fraud": is_fraud,
        "pattern": pattern,
    })


def build_stream(cfg: ExperimentConfig) -> pd.DataFrame:
    """Full synthetic stream, month by month."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    months = []
    for m in range(cfg.n_months):
        d = make_month(cfg.per_month, cfg.fraud_rate, share_of_new_pattern(m, cfg), rng)
        d.insert(0, "month", m)
        months.append(d)
    return pd.concat(months, ignore_index=True)


def load_kaggle(path: str | Path, n_months: int = 12) -> tuple[pd.DataFrame, list[str]]:
    """Load Kaggle `creditcard.csv` (ULB, 284,807 transactions, 492 frauds).

    Download it yourself from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    and place it in the `data/` folder. The data covers two days, so the
    "months" here are simply `n_months` equal-size time slices.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from Kaggle and put it in the data/ folder."
        )
    df = pd.read_csv(path)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df = df.rename(columns={"Class": "is_fraud", "Amount": "amount"})
    df["month"] = pd.qcut(df["Time"], q=n_months, labels=False).astype(int)
    features = [f"V{i}" for i in range(1, 29)] + ["amount"]
    cols = ["month"] + features + ["is_fraud"]
    return df[cols].reset_index(drop=True), features
