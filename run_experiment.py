"""Run the full benchmark from the command line and save every result.

Examples
--------
    python run_experiment.py                          # all three drift types, default settings
    python run_experiment.py --drift sudden           # only one drift type
    python run_experiment.py --config configs/default.json
    python run_experiment.py --quick                  # small, fast run (used by the tests)
    python run_experiment.py --kaggle data/creditcard.csv
    python run_experiment.py --ieee                   # real IEEE-CIS data in data/ieee-cis/

Output goes to results/<name>/:
    monthly.csv     one row per strategy and month (AUC, PR-AUC, F1, costs, alarms, ...)
    summary.csv     one row per strategy
    auc.png, pr_auc.png, costs.png
    run_info.json   config + Python/package versions + results fingerprint

The fingerprint is a hash of all monthly numbers. Running the same command
again (on Windows, macOS or Linux, with the pinned package versions) gives
the same fingerprint, which proves the results are reproducible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed; works the same on every OS
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_drift import (  # noqa: E402
    ExperimentConfig,
    environment_info,
    load_kaggle,
    results_fingerprint,
    run_experiment,
    summary_table,
)
from fraud_drift import ieee  # noqa: E402
from fraud_drift.plotting import plot_costs, plot_metric  # noqa: E402

ROOT = Path(__file__).resolve().parent
QUICK = dict(per_month=1000, n_months=10, drift_month=5, recur_month=8, transition_months=3, n_trees=30)


def save_run(name: str, cfg: ExperimentConfig, results, out_dir: Path, xlabel: str = "Month") -> str:
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)

    monthly = pd.concat([r.monthly.assign(strategy=k) for k, r in results.items()], ignore_index=True)
    monthly.insert(0, "strategy", monthly.pop("strategy"))
    monthly.round(6).to_csv(out / "monthly.csv", index=False, lineterminator="\n")
    summary = summary_table(results, cfg)
    summary.to_csv(out / "summary.csv", index=False, lineterminator="\n")

    for metric in ("auc", "pr_auc"):
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_metric(results, cfg, metric, ax=ax, xlabel=xlabel,
                    title=None if name not in ("ieee_cis", "kaggle") else f"{metric.upper().replace('_', '-')} by period · {name}")
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_costs(results, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "costs.png", dpi=150)
    plt.close(fig)

    fp = results_fingerprint(results)
    info = {"name": name, "fingerprint": fp, "config": cfg.to_dict(), "environment": environment_info()}
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== {name} ===  fingerprint: {fp}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"saved to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return fp


def main(argv=None) -> dict[str, str]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    ap.add_argument("--drift", choices=["sudden", "gradual", "recurring", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="small fast run")
    ap.add_argument("--seed", type=int, help="override the seed from the config")
    ap.add_argument("--kaggle", type=Path, help="path to Kaggle creditcard.csv (realism check)")
    ap.add_argument("--ieee", nargs="?", type=Path, const=ieee.DEFAULT_DIR,
                    help="run on the real IEEE-CIS data (folder with train_transaction.csv)")
    ap.add_argument("--period-days", type=int, default=14, help="IEEE-CIS: length of one period in days")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)

    base = ExperimentConfig.from_json(args.config) if args.config.exists() else ExperimentConfig()
    if args.quick:
        base = replace(base, **QUICK)
    if args.seed is not None:
        base = replace(base, seed=args.seed)

    fingerprints = {}
    if args.ieee:
        stream, features = ieee.load_ieee(args.ieee, period_days=args.period_days)
        cfg = replace(base, drift_type="none", n_months=int(stream.month.max()) + 1).validate()
        print(f"IEEE-CIS: {len(stream):,} transactions, {cfg.n_months} periods of {args.period_days} days, "
              f"{len(features)} features")
        results = run_experiment(cfg, stream=stream, features=features,
                                 progress=lambda i, n, name: print(f"  [{i}/{n}] {name}", flush=True))
        fingerprints["ieee_cis"] = save_run("ieee_cis", cfg, results, args.out,
                                            xlabel=f"Period ({args.period_days} days)")
        return fingerprints

    if args.kaggle:
        stream, features = load_kaggle(args.kaggle, n_months=12)
        cfg = replace(base, drift_type="none", n_months=12).validate()
        results = run_experiment(cfg, stream=stream, features=features)
        fingerprints["kaggle"] = save_run("kaggle", cfg, results, args.out)
        return fingerprints

    drifts = ["sudden", "gradual", "recurring"] if args.drift == "all" else [args.drift]
    for d in drifts:
        cfg = replace(base, drift_type=d).validate()
        results = run_experiment(cfg)
        name = f"{d}_quick" if args.quick else d
        fingerprints[name] = save_run(name, cfg, results, args.out)
    return fingerprints


if __name__ == "__main__":
    main()
