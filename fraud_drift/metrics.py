"""Evaluation: detection quality, recovery after drift and money."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def month_metrics(y_true, proba, amount, threshold: float = 0.5) -> dict:
    """Quality of one month of predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    amt = np.asarray(amount, dtype=float)
    flag = p >= threshold

    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    both_classes = len(np.unique(y)) == 2
    return {
        "auc": roc_auc_score(y, p) if both_classes else np.nan,
        "pr_auc": average_precision_score(y, p) if both_classes else np.nan,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_fraud": int(y.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "missed_amount": float(amt[~flag & (y == 1)].sum()),
    }


def add_costs(monthly: pd.DataFrame, investigation_cost: float, retraining_cost: float) -> pd.DataFrame:
    """Monthly cost = money lost to missed fraud + checking false alarms + retraining."""
    monthly = monthly.copy()
    monthly["cost_missed"] = monthly["missed_amount"]
    monthly["cost_false_alarms"] = monthly["fp"] * investigation_cost
    monthly["cost_retraining"] = monthly["retrained"].astype(int) * retraining_cost
    monthly["total_cost"] = monthly[["cost_missed", "cost_false_alarms", "cost_retraining"]].sum(axis=1)
    return monthly


def recovery_delay(months, values, drift_month: int, target: float, end: int | None = None):
    """Months from a change point until the metric is back at/above `target`.

    Only months in [drift_month, end) are considered (end = next change point).
    0    -> the model never dropped below the target after the change
    None -> it dropped and did not recover before `end` / the end of the stream
    """
    post = [(m, a) for m, a in zip(months, values)
            if m >= drift_month and (end is None or m < end) and not np.isnan(a)]
    dip = next((i for i, (_, a) in enumerate(post) if a < target), None)
    if dip is None:
        return 0
    back = next((m for m, a in post[dip:] if a >= target), None)
    return None if back is None else back - drift_month


def fingerprint(tables: dict[str, pd.DataFrame], decimals: int = 6) -> str:
    """Short SHA-256 of all monthly results.

    Same config + same package versions -> same fingerprint. Rounding and a
    fixed line ending make it identical on Windows, macOS and Linux.
    """
    h = hashlib.sha256()
    for name in sorted(tables):
        t = tables[name].round(decimals)
        h.update(name.encode())
        h.update(t.to_csv(index=False, lineterminator="\n", float_format=f"%.{decimals}f").encode())
    return h.hexdigest()[:16]
