"""Run all strategies on one stream and summarise the results."""

from __future__ import annotations

import platform
import random
import sys
from importlib import metadata

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import SYNTHETIC_FEATURES, build_stream
from .metrics import fingerprint, recovery_delay
from .strategies import RUNNERS, STRATEGIES, StrategyResult

TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "river", "matplotlib", "streamlit"]


def seed_everything(seed: int) -> None:
    """Belt and braces: our code uses explicit seeds everywhere, but we also
    fix the global generators in case a library falls back to them."""
    random.seed(seed)
    np.random.seed(seed)


def run_experiment(
    cfg: ExperimentConfig,
    stream: pd.DataFrame | None = None,
    features: list[str] | None = None,
    strategies: list[str] | None = None,
    progress=None,
) -> dict[str, StrategyResult]:
    """Run the chosen strategies (default: all four) on the same stream."""
    cfg.validate()
    seed_everything(cfg.seed)
    if stream is None:
        stream = build_stream(cfg)
    features = features or SYNTHETIC_FEATURES
    strategies = strategies or list(RUNNERS)

    results = {}
    for i, key in enumerate(strategies):
        if progress:
            progress(i, len(strategies), STRATEGIES[key])
        results[key] = RUNNERS[key](stream, features, cfg)
    if progress:
        progress(len(strategies), len(strategies), "done")
    return results


def summary_table(results: dict[str, StrategyResult], cfg: ExperimentConfig, recovery_level: float = 0.8) -> pd.DataFrame:
    """One row per strategy.

    Recovery delay uses PR-AUC (more honest than ROC-AUC when fraud is rare):
    months from the drift until PR-AUC is back to `recovery_level` (80%) of
    the strategy's own average level before the drift. For recurring drift the
    same is measured a second time from the month the old pattern returns.
    """
    has_drift = cfg.drift_type != "none"
    rows = []
    for key, r in results.items():
        t = r.monthly
        row = {"Strategy": r.label}
        if has_drift:
            pre, post = t[t.month < cfg.drift_month], t[t.month >= cfg.drift_month]
            row["AUC before drift"] = pre.auc.mean() if len(pre) else np.nan
            row["AUC after drift"] = post.auc.mean()
            row["PR-AUC after drift"] = post.pr_auc.mean()
            target = recovery_level * pre.pr_auc.mean() if len(pre) else np.nan
            recurring = cfg.drift_type == "recurring"
            d = recovery_delay(t.month, t.pr_auc, cfg.drift_month, target,
                               end=cfg.recur_month if recurring else None)
            row["Recovery delay (months)"] = "not recovered" if d is None else str(d)
            if recurring:
                d2 = recovery_delay(t.month, t.pr_auc, cfg.recur_month, target)
                row["Recovery after old pattern returns"] = "not recovered" if d2 is None else str(d2)
        else:
            row["AUC"] = t.auc.mean()
            row["PR-AUC"] = t.pr_auc.mean()
        row["F1"] = t.f1.mean()
        row["Retrainings"] = "continuous" if key == "online" else r.n_retrainings
        row["Missed frauds"] = int(t.fn.sum())
        row["False alarms"] = int(t.fp.sum())
        row["Missed fraud (EUR)"] = round(t.cost_missed.sum(), 2)
        row["Total cost (EUR)"] = round(t.total_cost.sum(), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.difference(["Missed frauds", "False alarms", "Retrainings"])
    out[num] = out[num].round(3)
    return out


def results_fingerprint(results: dict[str, StrategyResult]) -> str:
    return fingerprint({k: r.monthly for k, r in results.items()})


def environment_info() -> dict:
    versions = {}
    for p in TRACKED_PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }
