"""Charts shared by the app, the notebook and the command-line script."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig

COLORS = {
    "static": "#C0392B",
    "sliding": "#2E86C1",
    "triggered": "#8E44AD",
    "online": "#1E8449",
}
MARKERS = {"static": "o", "sliding": "s", "triggered": "^", "online": "D"}
METRIC_LABELS = {"auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "total_cost": "Cost per month (EUR)"}


def mark_drift(ax, cfg: ExperimentConfig, upto: int | None = None) -> None:
    """Draw where and how the fraud pattern changes."""
    if cfg.drift_type == "none":
        return
    end = cfg.n_months - 0.5
    if upto is not None and upto < cfg.drift_month:
        return
    if cfg.drift_type == "sudden":
        ax.axvline(cfg.drift_month - 0.5, color="black", ls="--", lw=1)
        ax.text(cfg.drift_month - 0.4, 0.02, "sudden drift", transform=ax.get_xaxis_transform(), fontsize=9)
    elif cfg.drift_type == "gradual":
        stop = min(cfg.drift_month + cfg.transition_months - 1, cfg.n_months - 1) + 0.5
        ax.axvspan(cfg.drift_month - 0.5, stop, color="grey", alpha=0.15, lw=0)
        ax.text(cfg.drift_month - 0.4, 0.02, "gradual drift", transform=ax.get_xaxis_transform(), fontsize=9)
    elif cfg.drift_type == "recurring":
        ax.axvspan(cfg.drift_month - 0.5, min(cfg.recur_month - 0.5, end), color="grey", alpha=0.15, lw=0)
        ax.text(cfg.drift_month - 0.4, 0.02, "new pattern", transform=ax.get_xaxis_transform(), fontsize=9)
        ax.text(cfg.recur_month - 0.4, 0.02, "old pattern returns", transform=ax.get_xaxis_transform(), fontsize=9)


def plot_metric(results, cfg: ExperimentConfig, metric: str = "auc", ax=None, title: str | None = None,
                xlim_months: list[int] | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    for key, r in results.items():
        t = r.monthly
        ax.plot(t.month, t[metric], marker=MARKERS[key], color=COLORS[key], lw=2, ms=5, label=r.label)
        if key == "triggered":
            for m in r.alarms:
                ax.axvline(m + 0.5, color=COLORS[key], ls=":", alpha=0.5)
    months = xlim_months or cfg.eval_months
    ax.set_xlim(min(months) - 0.6, max(months) + 0.6)
    if metric in ("auc", "pr_auc", "f1"):
        lo = np.nanmin([r.monthly[metric].min() for r in results.values()])
        ax.set_ylim(max(0.0, min(0.45, lo - 0.05)), 1.02)
    mark_drift(ax, cfg)
    ax.set_xlabel("Month")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} over time ({cfg.drift_type} drift)")
    return ax


def plot_costs(results, ax=None, title: str = "Total cost over the whole stream"):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r.label for r in results.values()]
    parts = [("cost_missed", "Missed fraud", 1.0), ("cost_false_alarms", "False alarms", 0.6),
             ("cost_retraining", "Retraining", 0.3)]
    bottom = np.zeros(len(labels))
    for col, name, alpha in parts:
        vals = np.array([r.monthly[col].sum() for r in results.values()])
        ax.bar(labels, vals, bottom=bottom, label=name, color=[COLORS[k] for k in results], alpha=alpha,
               edgecolor="white")
        bottom += vals
    for i, v in enumerate(bottom):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("EUR")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#555555", alpha=a, label=n) for _, n, a in parts]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    return ax
