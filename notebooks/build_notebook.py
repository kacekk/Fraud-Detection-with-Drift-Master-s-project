"""Builds fraud_concept_drift.ipynb from the cells below.

Run:  python notebooks/build_notebook.py
(Only needed if you change the cells here; the .ipynb is already in the repo.)
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# When Fraud Patterns Change
### Benchmarking drift adaptation methods for financial transactions

This notebook runs the experiments for my M.Sc. thesis. The question is simple: **when fraudsters change
their tactics, which way of updating a fraud model copes best?**

I compare four strategies on a stream of card transactions in which the fraud pattern changes at a month
I choose. Because I control the drift, I know exactly when it happened and can measure how long each
strategy needs to recover.

| Strategy | Idea | Type of adaptation |
|---|---|---|
| Static baseline | Random Forest trained once, never updated | none (reference) |
| Sliding window | Random Forest retrained every month on the latest months | blind |
| Detector-triggered | Random Forest retrained only when ADWIN/DDM raises a drift alarm | informed |
| Online learning | model updated after every labelled transaction | continuous |

All code lives in the `fraud_drift` package, so this notebook, the Streamlit app and `run_experiment.py`
give exactly the same numbers.
""")

md("""
## 1. Setup and reproducibility

Every random step (data generation, Random Forest, Adaptive Random Forest) takes its seed from the config,
and every model runs single-threaded. Running this notebook again gives **the same numbers every time**.
The last section checks this with a fingerprint (a hash of all results).
""")

code("""
import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fraud_drift import (ExperimentConfig, PATTERNS, build_stream, run_experiment, summary_table,
                         results_fingerprint, environment_info, load_kaggle)
from fraud_drift.plotting import plot_metric, plot_costs, COLORS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

cfg = ExperimentConfig.from_json(ROOT / "configs" / "default.json")
print("Seed:", cfg.seed)
print(environment_info())
cfg
""")

md("""
## 2. The data: a stream with a known change in fraud

Each month has the same number of transactions with five features: amount, hour of the day, number of
transactions in the last hour, foreign merchant (0/1) and new device (0/1).

Normal customers behave the same throughout. Only **what fraud looks like** changes:
""")

code("""
for k, v in PATTERNS.items():
    print(f"Pattern {k}: {v}")
""")

md("""
This is *real* concept drift: the link between the features and the label changes, P(fraud | features).
I simulate three kinds of change, following the taxonomy of Gama et al. (2014) and Lu et al. (2019):

- **sudden**: pattern A switches to pattern B from one month to the next
- **gradual**: pattern B takes over step by step over a few months
- **recurring**: pattern B appears, and later pattern A comes back
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for ax, d in zip(axes, ["sudden", "gradual", "recurring"]):
    s = build_stream(replace(cfg, drift_type=d))
    share = s[s.is_fraud == 1].groupby("month")["pattern"].apply(lambda p: (p == "B").mean())
    ax.bar(share.index, share.values, color=COLORS["triggered"], alpha=0.8)
    ax.set_title(f"{d} drift")
    ax.set_xlabel("Month")
axes[0].set_ylabel("Share of fraud using new pattern B")
plt.tight_layout()
plt.show()

stream = build_stream(cfg)
print(f"{len(stream):,} transactions, {stream.is_fraud.sum():,} frauds ({stream.is_fraud.mean():.1%})")
stream.head()
""")

code("""
# what an average fraud looks like before and after the drift (sudden)
fraud = stream[stream.is_fraud == 1]
fraud.assign(period=np.where(fraud.month < cfg.drift_month, "before drift", "after drift")) \\
     .groupby("period")[["amount", "hour", "n_txn_1h", "foreign", "device_new"]].mean().round(2)
""")

md("""
## 3. How the strategies are evaluated

- The first two months are historical, fully labelled data used for the first training (warm-up).
- From month 2 onward, each month is scored **before** the model sees its labels (no peeking).
- **ROC-AUC** and **PR-AUC** measure ranking quality. PR-AUC is more honest when fraud is rare
  (Dal Pozzolo et al., 2014; He & Garcia, 2009).
- **F1** uses a fixed alarm threshold of 0.5.
- **Recovery delay** = months after the drift until PR-AUC is back to 80% of the strategy's pre-drift level (a smaller tolerance would count normal month-to-month noise as a dip).
- The detector-triggered strategy uses ADWIN watching the confirmed fraud cases (section 9 explains why).
- **Cost (EUR)** = value of missed fraud + 15 EUR per false alarm checked + 500 EUR per retraining.
""")

md("## 4. Sudden drift")

code("""
results = {}
summaries = {}
for d in ["sudden", "gradual", "recurring"]:
    c = replace(cfg, drift_type=d)
    results[d] = run_experiment(c)
    summaries[d] = summary_table(results[d], c)

c = replace(cfg, drift_type="sudden")
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
plot_metric(results["sudden"], c, "auc", ax=axes[0])
plot_metric(results["sudden"], c, "pr_auc", ax=axes[1])
plt.tight_layout(); plt.show()
summaries["sudden"]
""")

md("""
**What happens.** The static model collapses at the drift and never comes back: it keeps looking for
large night-time payments while the fraudsters have moved to small, rapid foreign payments. The adaptive
strategies all recover; the differences are in *how fast* and *at what price* (number of retrainings,
false alarms).
""")

md("## 5. Gradual drift")

code("""
c = replace(cfg, drift_type="gradual")
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
plot_metric(results["gradual"], c, "auc", ax=axes[0])
plot_metric(results["gradual"], c, "pr_auc", ax=axes[1])
plt.tight_layout(); plt.show()
summaries["gradual"]
""")

md("""
Gradual drift is harder for a drift detector: the error rate rises slowly, so the alarm comes later
than with sudden drift. Strategies that adapt all the time (sliding window, online learning) do not need
to notice the change first.
""")

md("## 6. Recurring drift")

code("""
c = replace(cfg, drift_type="recurring")
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
plot_metric(results["recurring"], c, "auc", ax=axes[0])
plot_metric(results["recurring"], c, "pr_auc", ax=axes[1])
plt.tight_layout(); plt.show()
summaries["recurring"]
""")

md("""
Here the old pattern returns. A model that has completely "forgotten" pattern A has to adapt a second time,
which is the stability-plasticity trade-off discussed in the literature review.
""")

md("## 7. Costs")

code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, d in zip(axes, ["sudden", "gradual", "recurring"]):
    plot_costs(results[d], ax=ax, title=f"{d} drift")
    ax.tick_params(axis="x", labelrotation=15)
plt.tight_layout(); plt.show()

pd.concat({d: s.set_index("Strategy")[["Missed frauds", "False alarms", "Retrainings", "Total cost (EUR)"]]
           for d, s in summaries.items()}, axis=1)
""")

md("""
## 8. Label delay

In practice a fraud is confirmed only when the customer disputes the payment, often weeks later
(Dal Pozzolo et al., 2018). Here I repeat the sudden-drift experiment with labels arriving 0, 1 and 2
months late.
""")

code("""
rows = []
for delay in [0, 1, 2]:
    c = replace(cfg, drift_type="sudden", label_delay=delay)
    s = summary_table(run_experiment(c), c)
    s.insert(0, "Label delay (months)", delay)
    rows.append(s[["Label delay (months)", "Strategy", "PR-AUC after drift", "Recovery delay (months)", "Total cost (EUR)"]])
delay_table = pd.concat(rows, ignore_index=True)
delay_table.pivot(index="Strategy", columns="Label delay (months)", values=["PR-AUC after drift", "Recovery delay (months)"])
""")

md("""
## 9. What should the drift detector watch?

A detector needs a signal. The textbook choice is the model's overall error rate (Gama et al., 2004;
Bifet & Gavaldà, 2007). With only 5% fraud, though, a model that misses every new fraud still gets about
95% of transactions right, so the overall error rate barely moves. Watching only the confirmed fraud cases
(was each one caught or missed?) is a recall signal, similar in spirit to class-imbalance-aware detectors.
I compare both signals, with ADWIN and DDM, on a smaller stream where the difference is clearer.
""")

code("""
rows = []
for det in ["adwin", "ddm"]:
    for mon in ["fraud_recall", "error_rate"]:
        for d in ["sudden", "gradual"]:
            c = replace(cfg, drift_type=d, detector=det, monitor=mon, per_month=1500)
            r = run_experiment(c, strategies=["triggered"])["triggered"]
            post = r.monthly[r.monthly.month >= c.drift_month]
            rows.append({"Detector": det.upper(), "Watches": mon, "Drift": d,
                         "First alarm (months after drift)": (min(r.alarms) - c.drift_month) if r.alarms else None,
                         "Alarms": len(r.alarms), "PR-AUC after drift": round(post.pr_auc.mean(), 3)})
pd.DataFrame(rows)
""")

md("""
## 10. Are the differences stable? (10 seeds)

One run with one seed could be lucky. I repeat the sudden-drift benchmark with 10 different seeds
(smaller stream to keep it quick) and report the mean and standard deviation. The seeds are fixed
(0-9), so this table is also reproducible.
""")

code("""
rows = []
for s in range(10):
    c = replace(cfg, drift_type="sudden", per_month=2000, seed=s)
    t = summary_table(run_experiment(c), c)
    t["seed"] = s
    rows.append(t)
seeds = pd.concat(rows, ignore_index=True)
seeds.groupby("Strategy", sort=False)[["AUC after drift", "PR-AUC after drift", "Total cost (EUR)"]] \\
     .agg(["mean", "std"]).round(3)
""")

md("""
## 11. Realism check on the Kaggle credit-card data (optional)

The ULB credit-card dataset (284,807 transactions, 492 frauds) is not in the repository. Download
`creditcard.csv` from Kaggle and place it in the `data/` folder; this cell will then run automatically.
There is no injected drift here: the goal is to check that the strategies behave sensibly on real data.
""")

code("""
kaggle_path = ROOT / "data" / "creditcard.csv"
if kaggle_path.exists():
    real, feats = load_kaggle(kaggle_path, n_months=12)
    ck = replace(cfg, drift_type="none", n_months=12)
    real_results = run_experiment(ck, stream=real, features=feats)
    plot_metric(real_results, ck, "pr_auc", title="Kaggle data: PR-AUC per time slice")
    plt.show()
    display(summary_table(real_results, ck))
else:
    print("data/creditcard.csv not found - skipping (see README).")
""")

md("""
## 12. Reproducibility check

The fingerprint is a hash of every monthly number. If it matches the one saved by
`python run_experiment.py` in `results/`, the run is identical.
""")

code("""
import json
for d in ["sudden", "gradual", "recurring"]:
    fp = results_fingerprint(results[d])
    info_file = ROOT / "results" / d / "run_info.json"
    saved = json.loads(info_file.read_text())["fingerprint"] if info_file.exists() else "no saved run"
    print(f"{d:<10} this run: {fp}   saved: {saved}   {'identical' if fp == saved else ''}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent / "fraud_concept_drift.ipynb"
nbf.write(nb, out)
print("written", out)
