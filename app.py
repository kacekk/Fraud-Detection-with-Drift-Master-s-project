"""
When Fraud Patterns Change - interactive demo
=============================================
Streamlit app for the M.Sc. thesis "When Fraud Patterns Change: Benchmarking
Drift Adaptation Methods for Financial Transactions".

Start it with   python run.py            (Windows and macOS, sets everything up)
or directly     streamlit run app.py

All numbers come from the `fraud_drift` package, the same code used by the
notebook and by run_experiment.py. Same settings + same seed = same results.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fraud_drift import (
    PATTERNS,
    ExperimentConfig,
    build_stream,
    load_kaggle,
    results_fingerprint,
    run_experiment,
    summary_table,
)
from fraud_drift.plotting import COLORS, plot_costs, plot_metric

ROOT = Path(__file__).resolve().parent
KAGGLE_PATH = ROOT / "data" / "creditcard.csv"

st.set_page_config(page_title="When Fraud Patterns Change", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------
# cached computations (cache key = the full config, so results never mix)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_stream(cfg_dict: dict) -> pd.DataFrame:
    return build_stream(ExperimentConfig.from_dict(cfg_dict))


@st.cache_data(show_spinner=False)
def cached_kaggle(n_months: int):
    return load_kaggle(KAGGLE_PATH, n_months)


@st.cache_resource(show_spinner=False)
def cached_results(cfg_dict: dict, source: str):
    cfg = ExperimentConfig.from_dict(cfg_dict)
    if source == "kaggle":
        stream, features = cached_kaggle(cfg.n_months)
        return run_experiment(cfg, stream=stream, features=features)
    return run_experiment(cfg, stream=cached_stream(cfg_dict))


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

sources = ["Synthetic stream (known drift)"]
if KAGGLE_PATH.exists():
    sources.append("Kaggle credit-card data")
source_label = st.sidebar.radio("Data", sources)
source = "kaggle" if source_label.startswith("Kaggle") else "synthetic"
if source == "synthetic" and not KAGGLE_PATH.exists():
    st.sidebar.caption("Put `creditcard.csv` in the `data/` folder to unlock the Kaggle option.")

defaults = ExperimentConfig()
st.sidebar.subheader("Stream")
if source == "synthetic":
    drift_type = st.sidebar.selectbox("Type of drift", ["sudden", "gradual", "recurring", "none"],
                                      help="How the fraud pattern changes over time.")
    per_month = st.sidebar.slider("Transactions per month", 1000, 20000, defaults.per_month, step=1000)
    n_months = st.sidebar.slider("Number of months", 8, 36, defaults.n_months)
    drift_month = st.sidebar.slider("Drift starts in month", 2, n_months - 2, min(defaults.drift_month, n_months - 2),
                                    disabled=drift_type == "none")
    transition = defaults.transition_months
    recur_month = min(max(defaults.recur_month, drift_month + 1), n_months - 1)
    if drift_type == "gradual":
        transition = st.sidebar.slider("Months until the new pattern fully takes over", 1, 8, defaults.transition_months)
    if drift_type == "recurring":
        recur_month = st.sidebar.slider("Old pattern returns in month", drift_month + 1, n_months - 1, recur_month)
    fraud_rate = st.sidebar.slider("Fraud rate", 0.01, 0.20, defaults.fraud_rate, step=0.01)
else:
    drift_type, per_month, n_months, drift_month = "none", defaults.per_month, 12, defaults.drift_month
    transition, recur_month, fraud_rate = defaults.transition_months, defaults.recur_month, defaults.fraud_rate
    st.sidebar.caption("Kaggle data: 284,807 real transactions split into 12 time slices. No injected drift.")

seed = st.sidebar.number_input("Random seed", 0, 99999, defaults.seed,
                               help="Same seed + same settings = exactly the same results.")

st.sidebar.subheader("Models")
n_trees = st.sidebar.slider("Random Forest trees", 20, 300, defaults.n_trees, step=20)
window = st.sidebar.slider("Retraining window (months)", 1, 6, defaults.window)
detector = st.sidebar.selectbox("Drift detector", ["adwin", "ddm"], format_func=str.upper)
delta = defaults.adwin_delta
if detector == "adwin":
    delta = st.sidebar.select_slider("ADWIN sensitivity (delta)", [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
                                     value=defaults.adwin_delta,
                                     help="Smaller delta = fewer false alarms but slower detection.")
monitor = st.sidebar.selectbox("Detector watches", ["fraud_recall", "error_rate"],
                               format_func={"fraud_recall": "Missed frauds (recall)",
                                            "error_rate": "All errors"}.get,
                               help="With rare fraud, the overall error rate barely moves when fraud changes. "
                                    "Watching only confirmed frauds reacts much faster.")
online_model = st.sidebar.selectbox("Online model", ["logreg", "arf"],
                                    format_func={"logreg": "Logistic regression (fast)",
                                                 "arf": "Adaptive Random Forest (slow)"}.get)
threshold = st.sidebar.slider("Alarm threshold (probability)", 0.05, 0.95, defaults.threshold, step=0.05)
label_delay = st.sidebar.slider("Label delay (months)", 0, 3, defaults.label_delay,
                                help="Fraud is confirmed weeks later (chargebacks). Models only learn from labelled months.")

st.sidebar.subheader("Costs (EUR)")
investigation_cost = st.sidebar.number_input("Checking one false alarm", 0.0, 1000.0, defaults.investigation_cost, step=5.0)
retraining_cost = st.sidebar.number_input("One model retraining", 0.0, 10000.0, defaults.retraining_cost, step=100.0)

try:
    cfg = ExperimentConfig(
        per_month=per_month, n_months=n_months, fraud_rate=fraud_rate, drift_type=drift_type,
        drift_month=drift_month, transition_months=transition, recur_month=recur_month,
        n_trees=n_trees, window=window, detector=detector, adwin_delta=delta, monitor=monitor,
        online_model=online_model,
        threshold=threshold, label_delay=label_delay, investigation_cost=investigation_cost,
        retraining_cost=retraining_cost, seed=int(seed),
    ).validate()
except ValueError as e:
    st.error(f"Invalid settings: {e}")
    st.stop()

if source == "synthetic":
    st.sidebar.metric("Total transactions", f"{per_month * n_months:,}")
    if online_model == "arf" and per_month * n_months > 60_000:
        st.sidebar.warning("ARF on a large stream can take a few minutes.")


# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.title("🛡️ When Fraud Patterns Change")
st.markdown(
    "Fraudsters change tactics, so a model trained on yesterday's fraud slowly stops working. "
    "This is **concept drift**. The app simulates a stream of card transactions in which the fraud "
    "pattern changes at a known month and compares four ways of keeping the model up to date."
)
with st.expander("What are the four strategies?"):
    st.markdown(
        "- **Static baseline**: Random Forest trained once on the first months, never updated.\n"
        "- **Sliding window**: Random Forest retrained every month on the latest labelled months.\n"
        "- **Detector-triggered**: Random Forest retrained only when a drift detector (ADWIN or DDM) "
        "notices that the model has started missing more frauds.\n"
        "- **Online learning**: a model that updates itself after every labelled transaction."
    )

tab_compare, tab_live, tab_data = st.tabs(["📊 Compare strategies", "🎬 Live stream", "🔎 What does the fraud look like?"])


def run_with_progress():
    return cached_results(cfg.to_dict(), source)


# ----------------------------------------------------------------------------
# tab 1: compare
# ----------------------------------------------------------------------------
with tab_compare:
    st.caption("Change the settings on the left, then press Run. Results are cached, so re-running the same settings is instant.")
    if st.button("▶️ Run benchmark", type="primary", key="run_compare"):
        st.session_state["show_compare"] = True

    if st.session_state.get("show_compare"):
        with st.spinner("Training and evaluating the four strategies…"):
            results = run_with_progress()

        c1, c2 = st.columns([3, 2])
        metric = c1.radio("Metric", ["auc", "pr_auc", "f1", "total_cost"], horizontal=True,
                          format_func={"auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
                                       "total_cost": "Cost per month"}.get)
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_metric(results, cfg, metric, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        alarms = results["triggered"].alarms
        st.caption(f"Dotted purple lines: {cfg.detector.upper()} drift alarms (months {alarms})." if alarms
                   else f"{cfg.detector.upper()} raised no drift alarm with these settings.")

        st.subheader("Summary")
        summary = summary_table(results, cfg)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "Recovery delay = months after the drift until PR-AUC is back to 80% of its pre-drift level. "
            "Total cost = missed fraud amount + false alarms × checking cost + retrainings × retraining cost."
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        plot_costs(results, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        monthly = pd.concat([r.monthly.assign(strategy=k) for k, r in results.items()], ignore_index=True)
        fp = results_fingerprint(results)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Monthly results (CSV)", monthly.to_csv(index=False), "monthly.csv", "text/csv")
        d2.download_button("⬇️ Summary (CSV)", summary.to_csv(index=False), "summary.csv", "text/csv")
        d3.download_button("⬇️ Settings (JSON)", pd.Series(cfg.to_dict()).to_json(indent=2), "config.json")
        st.caption(f"Reproducibility fingerprint: `{fp}` — the same settings always give this same code.")
    else:
        st.info("Press **Run benchmark** to start.")


# ----------------------------------------------------------------------------
# tab 2: live stream
# ----------------------------------------------------------------------------
with tab_live:
    st.markdown(
        "The stream is played back **one month at a time**: each month's transactions arrive, every "
        "strategy scores them, and the chart grows. The numbers are the same as in the comparison tab."
    )
    speed = st.slider("Seconds per month", 0.1, 2.0, 0.6, step=0.1)
    if st.button("▶️ Start live stream", type="primary", key="run_live"):
        with st.spinner("Preparing the stream…"):
            results = run_with_progress()
        stream = cached_stream(cfg.to_dict()) if source == "synthetic" else None

        chart_ph, status_ph, metric_ph, feed_ph = st.empty(), st.empty(), st.empty(), st.empty()
        for i, m in enumerate(cfg.eval_months):
            shown = {k: replace(r, monthly=r.monthly.iloc[: i + 1],
                                alarms=[a for a in r.alarms if a <= m]) for k, r in results.items()}
            fig, ax = plt.subplots(figsize=(11, 5))
            plot_metric(shown, cfg, "auc", ax=ax, title=f"Live stream — month {m}")
            fig.tight_layout()
            chart_ph.pyplot(fig)
            plt.close(fig)

            if cfg.drift_type == "none":
                phase = "stream"
            elif m < cfg.drift_month:
                phase = "🟢 old fraud pattern"
            elif cfg.drift_type == "recurring" and m >= cfg.recur_month:
                phase = "🔁 old pattern is back"
            else:
                phase = "🔴 new fraud pattern"
            alarm = m in results["triggered"].alarms
            status_ph.markdown(f"**Month {m}** — {phase}" + ("  ·  🟣 drift alarm → retraining" if alarm else ""))

            cols = metric_ph.columns(4)
            for col, (k, r) in zip(cols, results.items()):
                row = r.monthly.iloc[i]
                col.metric(r.label, f"AUC {row.auc:.3f}", f"{int(row.tp)} caught / {int(row.fn)} missed",
                           delta_color="off")

            if stream is not None:
                sample = stream[stream.month == m].head(8)[["amount", "hour", "n_txn_1h", "foreign",
                                                            "device_new", "is_fraud", "pattern"]]
                feed_ph.dataframe(sample, use_container_width=True, hide_index=True)
            time.sleep(speed)
        st.success("Stream finished. Open **Compare strategies** for the summary table and costs.")
    else:
        st.caption("Press **Start live stream** to watch the months play out.")


# ----------------------------------------------------------------------------
# tab 3: data
# ----------------------------------------------------------------------------
with tab_data:
    if source != "synthetic":
        st.info("This view explains the synthetic generator. Switch the data source to Synthetic.")
    else:
        stream = cached_stream(cfg.to_dict())
        st.markdown(
            f"- **Pattern A** — {PATTERNS['A']}\n- **Pattern B** — {PATTERNS['B']}\n\n"
            "Normal customers behave the same all the time. Only the way fraud looks changes, which is "
            "exactly what makes it *concept* drift."
        )
        fraud = stream[stream.is_fraud == 1]
        share = fraud.groupby("month")["pattern"].apply(lambda s: (s == "B").mean()).rename("Share of new pattern B")
        st.bar_chart(share, color=COLORS["triggered"])
        st.markdown("**Average fraud transaction per month**")
        st.dataframe(fraud.groupby("month")[["amount", "hour", "n_txn_1h", "foreign", "device_new"]].mean().round(2),
                     use_container_width=True)

st.markdown("---")
st.caption("M.Sc. thesis demo · Kozminski University · Kacper Matejczuk · synthetic data with injected concept drift")
