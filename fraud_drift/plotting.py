"""Charts shared by the app, the notebook and the command-line script."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import ExperimentConfig

# muted, print-friendly palette (distinguishable for common colour-vision deficiencies)
COLORS = {
    "static": "#B4503F",
    "sliding": "#3F6FA8",
    "triggered": "#7B5EA7",
    "online": "#3D8B65",
}
MARKERS = {"static": "o", "sliding": "s", "triggered": "^", "online": "D"}
METRIC_LABELS = {"auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "total_cost": "Cost per month (EUR)"}

INK = "#1F2329"
MUTED = "#6B7079"
GRID = "#E6E4DF"
SHADE = "#ECEAE4"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#C9C6BF",
    "axes.labelcolor": MUTED,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def apply_style() -> None:
    """Use the project's chart style everywhere (app, notebook, saved PNGs)."""
    mpl.rcParams.update(STYLE)


apply_style()


def mark_drift(ax, cfg: ExperimentConfig) -> None:
    """Shade where the fraud pattern is changing or changed."""
    if cfg.drift_type == "none":
        return
    end = cfg.n_months - 0.5
    trans = ax.get_xaxis_transform()
    # labels sit just above the plot area so they never cover the lines
    label_kw = dict(transform=trans, fontsize=8.5, color=MUTED, va="bottom", clip_on=False)
    if cfg.drift_type == "sudden":
        ax.axvline(cfg.drift_month - 0.5, color=INK, ls=(0, (4, 3)), lw=1)
        ax.text(cfg.drift_month - 0.5, 1.01, "sudden drift", ha="center", **label_kw)
    elif cfg.drift_type == "gradual":
        stop = min(cfg.drift_month + cfg.transition_months - 1, cfg.n_months - 1) + 0.5
        ax.axvspan(cfg.drift_month - 0.5, stop, color=SHADE, lw=0, zorder=0)
        ax.text(cfg.drift_month - 0.45, 1.01, "gradual drift", **label_kw)
    elif cfg.drift_type == "recurring":
        ax.axvspan(cfg.drift_month - 0.5, min(cfg.recur_month - 0.5, end), color=SHADE, lw=0, zorder=0)
        ax.text(cfg.drift_month - 0.45, 1.01, "new pattern", **label_kw)
        ax.text(cfg.recur_month - 0.45, 1.01, "old pattern returns", **label_kw)


def plot_metric(results, cfg: ExperimentConfig, metric: str = "auc", ax=None, title: str | None = None,
                xlim_months: list[int] | None = None, xlabel: str = "Month"):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.8))
    for key, r in results.items():
        t = r.monthly
        ax.plot(t.month, t[metric], marker=MARKERS[key], color=COLORS[key], lw=2, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.8, label=r.label, zorder=3)
        if key == "triggered":
            for m in r.alarms:
                ax.axvline(m + 0.5, color=COLORS[key], ls=":", lw=1.2, alpha=0.7, zorder=1)
    months = xlim_months or cfg.eval_months
    ax.set_xlim(min(months) - 0.6, max(months) + 0.6)
    ax.set_xticks(months)
    if metric in ("auc", "pr_auc", "f1"):
        lo = np.nanmin([r.monthly[metric].min() for r in results.values()])
        ax.set_ylim(max(0.0, min(0.45, lo - 0.05)), 1.04)
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    mark_drift(ax, cfg)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=4, fontsize=9, handlelength=2.2,
              columnspacing=1.8)
    ax.set_title(title if title is not None else f"{METRIC_LABELS.get(metric, metric)} by month · {cfg.drift_type} drift")
    return ax


def plot_costs(results, ax=None, title: str = "Total cost over the stream (EUR)"):
    """Horizontal stacked bars: missed fraud, false alarms, retraining."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.4))
    keys = list(results)[::-1]
    labels = [results[k].label for k in keys]
    parts = [("cost_missed", "Missed fraud", 1.0), ("cost_false_alarms", "False alarms", 0.55),
             ("cost_retraining", "Retraining", 0.25)]
    left = np.zeros(len(keys))
    for col, _, alpha in parts:
        vals = np.array([results[k].monthly[col].sum() for k in keys])
        ax.barh(labels, vals, left=left, color=[COLORS[k] for k in keys], alpha=alpha, height=0.58,
                edgecolor="white", linewidth=1)
        left += vals
    for i, v in enumerate(left):
        ax.text(v + left.max() * 0.01, i, f"{v:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_axisbelow(True)
    ax.set_xlim(0, left.max() * 1.12)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    ax.set_title(title)
    handles = [Patch(facecolor="#4A4E55", alpha=a, label=n) for _, n, a in parts]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=9)
    return ax
