"""A small stand-in for the IEEE-CIS files with the same column names and types.

The real data needs a Kaggle login, so the automatic tests use this file to
check the loader and the pipeline. It is NOT used for any thesis result.
"""

import numpy as np
import pandas as pd

from fraud_drift.ieee import BINARY_TF, NUMERIC


def make_fake_ieee(n: int = 20_000, days: int = 183, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(86_400, 86_400 * (days + 1), size=n))
    tx = pd.DataFrame({"TransactionID": np.arange(2_987_000, 2_987_000 + n), "TransactionDT": dt})
    for c in NUMERIC:
        tx[c] = rng.gamma(2.0, 50.0, size=n).round(2)
        if c != "TransactionAmt":
            tx.loc[rng.random(n) < 0.2, c] = np.nan
    tx["ProductCD"] = rng.choice(["W", "C", "R", "H", "S"], size=n)
    tx["card4"] = rng.choice(["visa", "mastercard", "american express", None], size=n)
    tx["card6"] = rng.choice(["debit", "credit", None], size=n)
    tx["M4"] = rng.choice(["M0", "M1", "M2", None], size=n)
    for c in BINARY_TF:
        tx[c] = rng.choice(["T", "F", None], size=n)
    tx["P_emaildomain"] = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], size=n)
    tx["R_emaildomain"] = rng.choice(["gmail.com", "outlook.com", None], size=n)
    # fraud depends on product and amount, and the dependence changes halfway (natural drift)
    late = dt > dt.max() / 2
    score = (np.where(late, tx.ProductCD == "C", tx.ProductCD == "W") * 2.0
             + (tx.TransactionAmt.fillna(0) > 150) * 1.0)
    tx["isFraud"] = (rng.random(n) < 0.01 + 0.04 * score / 3).astype(int)
    identity = tx.sample(frac=0.3, random_state=seed)[["TransactionID"]].copy()
    identity["DeviceType"] = rng.choice(["desktop", "mobile"], size=len(identity))
    return tx, identity
