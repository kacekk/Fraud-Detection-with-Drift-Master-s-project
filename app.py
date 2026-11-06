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
    ExperimentConfig,
    build_stream,
    load_kaggle,
    results_fingerprint,
    run_experiment,
    summary_table,
)
from fraud_drift import ieee
from fraud_drift.data import share_of_new_pattern
from fraud_drift.plotting import COLORS, INK, MUTED, plot_costs, plot_metric

ROOT = Path(__file__).resolve().parent
KAGGLE_PATH = ROOT / "data" / "creditcard.csv"

st.set_page_config(page_title="When Fraud Patterns Change", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# look and feel
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #1F2329; --muted: #6B7079; --line: #E3E1DB; --paper: #FAF9F6; --panel: #FFFFFF;
  --accent: #1F3A5F;
}
html, body, [class*="css"], .stMarkdown, .stText, button, input, label, select, textarea {
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
}
.stApp { background: var(--paper); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

section[data-testid="stSidebar"] { background: #F2F0EB; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.side-title { font-size: 0.72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin: 0.4rem 0 0.2rem 0; }

.eyebrow { font-size: 0.74rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin-bottom: .35rem; }
h1.title { font-size: 2.35rem; line-height: 1.12; font-weight: 700; color: var(--ink); margin: 0 0 .6rem 0;
  letter-spacing: -0.02em; }
.lede { font-size: 1.02rem; color: #3E434B; max-width: 780px; line-height: 1.6; margin-bottom: 1.2rem; }

.stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .6rem 0; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }

.stButton > button[kind="primary"], .stDownloadButton > button {
  border-radius: 6px; font-weight: 600; padding: .5rem 1.2rem; }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
.stDownloadButton > button { background: var(--panel); color: var(--ink); border: 1px solid var(--line); }

.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: .6rem 0 1.4rem 0; }
@media (max-width: 900px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px 12px 16px;
  border-top: 3px solid var(--c); }
.card .name { font-size: .82rem; font-weight: 600; color: var(--ink); margin-bottom: .55rem; }
.card .big { font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 500; color: var(--ink);
  line-height: 1.1; }
.card .big small { font-family: 'Inter', sans-serif; font-size: .72rem; color: var(--muted); font-weight: 500;
  margin-left: .35rem; }
.card .row { display: flex; justify-content: space-between; font-size: .8rem; color: var(--muted);
  border-top: 1px solid #F0EEE9; margin-top: .5rem; padding-top: .45rem; }
.card .row b { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: var(--ink); }

.note { font-size: .84rem; color: var(--muted); line-height: 1.55; }
.badge { display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .04em; padding: .2rem .6rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); margin-right: .4rem; }
.badge.old { color: #3D8B65; border-color: #CFE3D8; background: #F1F8F4; }
.badge.new { color: #B4503F; border-color: #EBD3CE; background: #FBF3F1; }
.badge.alarm { color: #7B5EA7; border-color: #DDD3EB; background: #F6F2FA; }
.fp { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: var(--ink); background: #EFEDE8;
  padding: .1rem .45rem; border-radius: 4px; }

.datagrid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: .4rem 0 1rem 0; }
@media (max-width: 900px) { .datagrid { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.panel h4 { margin: 0 0 .5rem 0; font-size: .95rem; color: var(--ink); }
.panel p, .panel li { font-size: .88rem; color: #3E434B; line-height: 1.55; }
</style>
""",
    unsafe_allow_html=True,
)

METRIC_NAMES = {"auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "total_cost": "Cost per month"}


def show(fig) -> None:
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


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
def cached_ieee():
    return ieee.load_ieee()


@st.cache_resource(show_spinner=False)
def cached_results(cfg_dict: dict, source: str):
    cfg = ExperimentConfig.from_dict(cfg_dict)
    if source == "kaggle":
        stream, features = cached_kaggle(cfg.n_months)
        return run_experiment(cfg, stream=stream, features=features)
    if source == "ieee":
        stream, features = cached_ieee()
        return run_experiment(cfg, stream=stream, features=features)
    return run_experiment(cfg, stream=cached_stream(cfg_dict))


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
defaults = ExperimentConfig()
sb = st.sidebar
sb.markdown("### Settings")

SOURCES = {"synthetic": "Synthetic stream (controlled drift)", "ieee": "IEEE-CIS (real e-commerce)",
           "kaggle": "ULB credit-card (real, 2 days)"}
available = ["synthetic"] + (["ieee"] if ieee.is_available() else []) + (["kaggle"] if KAGGLE_PATH.exists() else [])
source = sb.radio("Data source", available, format_func=SOURCES.get)
if len(available) == 1:
    sb.caption("Add the IEEE-CIS files to data/ieee-cis/ to unlock real data (see data/README.md).")
unit = "two-week period" if source == "ieee" else "month"
currency, symbol = ("USD", "$") if source == "ieee" else ("EUR", "€")   # IEEE-CIS amounts are in USD

sb.markdown('<div class="side-title">Stream</div>', unsafe_allow_html=True)
if source == "synthetic":
    drift_type = sb.selectbox("Type of drift", ["sudden", "gradual", "recurring", "none"],
                              format_func=str.capitalize, help="How the fraud pattern changes over time.")
    per_month = sb.slider("Transactions per month", 1000, 20000, defaults.per_month, step=1000)
    n_months = sb.slider("Number of months", 8, 36, defaults.n_months)
    drift_month = sb.slider("Drift starts in month", 2, n_months - 2, min(defaults.drift_month, n_months - 2),
                            disabled=drift_type == "none")
    transition = defaults.transition_months
    recur_month = min(max(defaults.recur_month, drift_month + 1), n_months - 1)
    if drift_type == "gradual":
        transition = sb.slider("Months until the new pattern takes over", 1, 8, defaults.transition_months)
    if drift_type == "recurring":
        recur_month = sb.slider("Old pattern returns in month", drift_month + 1, n_months - 1, recur_month)
    fraud_rate = sb.slider("Fraud rate", 0.01, 0.20, defaults.fraud_rate, step=0.01, format="%.2f")
elif source == "ieee":
    with st.spinner("Loading IEEE-CIS (first time: about a minute, then cached)"):
        ieee_stream, _ = cached_ieee()
    drift_type, per_month, n_months, drift_month = "none", defaults.per_month, int(ieee_stream.month.max()) + 1, 2
    transition, recur_month, fraud_rate = defaults.transition_months, defaults.recur_month, defaults.fraud_rate
    sb.caption(f"{len(ieee_stream):,} real transactions in {n_months} two-week periods. "
               "The drift is natural, so its timing is unknown.")
else:
    drift_type, per_month, n_months, drift_month = "none", defaults.per_month, 12, defaults.drift_month
    transition, recur_month, fraud_rate = defaults.transition_months, defaults.recur_month, defaults.fraud_rate
    sb.caption("284,807 real transactions split into 12 time slices. No injected drift.")

seed = sb.number_input("Random seed", 0, 99999, defaults.seed,
                       help="Same seed and settings always give exactly the same results.")

with sb.expander("Models", expanded=False):
    n_trees = st.slider("Random Forest trees", 20, 300, defaults.n_trees, step=20)
    window = st.slider(f"Retraining window ({unit}s)", 1, 6, defaults.window)
    detector = st.selectbox("Drift detector", ["adwin", "ddm"], format_func=str.upper)
    delta = defaults.adwin_delta
    if detector == "adwin":
        delta = st.select_slider("ADWIN sensitivity (delta)", [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
                                 value=defaults.adwin_delta,
                                 help="Smaller delta: fewer false alarms, slower detection.")
    monitor = st.selectbox("Detector watches", ["fraud_recall", "error_rate"],
                           format_func={"fraud_recall": "Missed frauds (recall)", "error_rate": "All errors"}.get,
                           help="With rare fraud the overall error rate barely moves when fraud changes. "
                                "Watching only confirmed frauds reacts much faster.")
    online_model = st.selectbox("Online model", ["logreg", "arf"],
                                format_func={"logreg": "Logistic regression (fast)",
                                             "arf": "Adaptive Random Forest (slow)"}.get)
    threshold = st.slider("Alarm threshold", 0.05, 0.95, defaults.threshold, step=0.05)

with sb.expander("Realism and costs", expanded=False):
    label_delay = st.slider(f"Label delay ({unit}s)", 0, 3, defaults.label_delay,
                            help="Fraud is confirmed weeks later through chargebacks. "
                                 "Models only learn from months whose labels have arrived.")
    investigation_cost = st.number_input(f"Checking one false alarm ({currency})", 0.0, 1000.0,
                                         defaults.investigation_cost, step=5.0)
    retraining_cost = st.number_input(f"One model retraining ({currency})", 0.0, 10000.0, defaults.retraining_cost,
                                      step=100.0)

try:
    cfg = ExperimentConfig(
        per_month=per_month, n_months=n_months, fraud_rate=fraud_rate, drift_type=drift_type,
        drift_month=drift_month, transition_months=transition, recur_month=recur_month,
        n_trees=n_trees, window=window, detector=detector, adwin_delta=delta, monitor=monitor,
        online_model=online_model, threshold=threshold, label_delay=label_delay,
        investigation_cost=investigation_cost, retraining_cost=retraining_cost, seed=int(seed),
    ).validate()
except ValueError as e:
    st.error(f"Invalid settings: {e}")
    st.stop()

if source == "synthetic":
    sb.caption(f"{per_month * n_months:,} transactions in total")
    if online_model == "arf" and per_month * n_months > 60_000:
        sb.warning("Adaptive Random Forest on a large stream can take a few minutes.")


# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="eyebrow">M.Sc. thesis · Kozminski University</div>
<h1 class="title">When Fraud Patterns Change</h1>
<p class="lede">Fraudsters change tactics, so a model trained on yesterday's fraud slowly stops working.
This is <b>concept drift</b>. The app compares four ways of keeping a fraud model up to date, on a synthetic
stream where we control exactly when fraud changes, and on real e-commerce transactions (IEEE-CIS).</p>
""",
    unsafe_allow_html=True,
)

tab_bench, tab_live, tab_data = st.tabs(["Benchmark", "Live stream", "Data"])


def get_results():
    return cached_results(cfg.to_dict(), source)


def strategy_cards(results, summary: pd.DataFrame) -> None:
    has_drift = cfg.drift_type != "none"
    html = ['<div class="cards">']
    for (key, r), (_, row) in zip(results.items(), summary.iterrows()):
        main = row["PR-AUC after drift"] if has_drift else row["PR-AUC"]
        main_label = "PR-AUC after drift" if has_drift else "PR-AUC"
        recov = row["Recovery delay (months)"] if has_drift else "–"
        recov = f"{recov} mo" if str(recov).isdigit() else recov
        html.append(
            f'<div class="card" style="--c:{COLORS[key]}">'
            f'<div class="name">{r.label}</div>'
            f'<div class="big">{main:.3f}<small>{main_label}</small></div>'
            f'<div class="row"><span>Recovery</span><b>{recov}</b></div>'
            f'<div class="row"><span>Retrainings</span><b>{row["Retrainings"]}</b></div>'
            f'<div class="row"><span>Total cost</span><b>{symbol}{row[f"Total cost ({currency})"]:,.0f}</b></div>'
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# tab 1: benchmark
# ----------------------------------------------------------------------------
with tab_bench:
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    if c1.button("Run benchmark", type="primary", key="run_compare", width="stretch"):
        st.session_state["show_compare"] = True
    c2.markdown('<span class="note">Adjust the settings in the sidebar, then run. '
                'Identical settings are served from cache instantly.</span>', unsafe_allow_html=True)

    if st.session_state.get("show_compare"):
        with st.spinner("Training and evaluating the four strategies"):
            results = get_results()
        summary = summary_table(results, cfg, currency=currency)

        strategy_cards(results, summary)

        metric = st.segmented_control("Metric", list(METRIC_NAMES), format_func=METRIC_NAMES.get,
                                      default="pr_auc", label_visibility="collapsed") or "pr_auc"
        fig, ax = plt.subplots(figsize=(11, 4.6))
        plot_metric(results, cfg, metric, ax=ax, xlabel=unit.capitalize(),
                    title=None if source == "synthetic" else f"{METRIC_NAMES[metric]} by {unit} · {SOURCES[source]}")
        show(fig)
        alarms = results["triggered"].alarms
        st.markdown(
            f'<span class="note">Dotted purple lines mark {cfg.detector.upper()} drift alarms '
            f'({", ".join(f"{unit} {m}" for m in alarms)}); the model is retrained after each one.</span>'
            if alarms else f'<span class="note">{cfg.detector.upper()} raised no drift alarm with these settings.</span>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Summary")
        st.dataframe(summary, width="stretch", hide_index=True)
        st.markdown(
            '<span class="note">Recovery delay: months after the drift until PR-AUC is back to 80% of its '
            "pre-drift level. Total cost: value of missed fraud + false alarms × checking cost + "
            "retrainings × retraining cost.</span>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Where the money goes")
        fig, ax = plt.subplots(figsize=(11, 3.3))
        plot_costs(results, ax=ax, title="")
        show(fig)

        monthly = pd.concat([r.monthly.assign(strategy=k) for k, r in results.items()], ignore_index=True)
        d1, d2, d3, d4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
        d1.download_button("Monthly results", monthly.to_csv(index=False), "monthly.csv", "text/csv",
                           width="stretch")
        d2.download_button("Summary table", summary.to_csv(index=False), "summary.csv", "text/csv",
                           width="stretch")
        d3.download_button("Settings", pd.Series(cfg.to_dict()).to_json(indent=2), "config.json",
                           width="stretch")
        d4.markdown(f'<span class="note">Reproducibility fingerprint '
                    f'<span class="fp">{results_fingerprint(results)}</span></span>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# tab 2: live stream
# ----------------------------------------------------------------------------
with tab_live:
    st.markdown('<p class="note">The stream is played back one month at a time: each month\'s transactions '
                'arrive, every strategy scores them, and the chart grows. The numbers are identical to the '
                'Benchmark tab.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    start = c1.button("Start stream", type="primary", key="run_live", width="stretch")
    speed = c2.slider("Seconds per month", 0.1, 2.0, 0.6, step=0.1)

    if start:
        with st.spinner("Preparing the stream"):
            results = get_results()
        stream = cached_stream(cfg.to_dict()) if source == "synthetic" else None
        sample_cols = ["amount", "hour", "n_txn_1h", "foreign", "device_new", "is_fraud", "pattern"]
        if source == "ieee":
            stream = cached_ieee()[0]
            sample_cols = ["TransactionID", "amount", "hour", "ProductCD=W", "C1", "D1", "is_fraud"]

        status_ph, chart_ph, cards_ph, feed_ph = st.empty(), st.empty(), st.empty(), st.empty()
        for i, m in enumerate(cfg.eval_months):
            shown = {k: replace(r, monthly=r.monthly.iloc[: i + 1], alarms=[a for a in r.alarms if a <= m])
                     for k, r in results.items()}

            if cfg.drift_type == "none":
                badge = '<span class="badge">stream</span>'
            elif m < cfg.drift_month:
                badge = '<span class="badge old">old fraud pattern</span>'
            elif cfg.drift_type == "recurring" and m >= cfg.recur_month:
                badge = '<span class="badge old">old pattern is back</span>'
            elif cfg.drift_type == "gradual" and share_of_new_pattern(m, cfg) < 1:
                badge = f'<span class="badge new">new pattern: {share_of_new_pattern(m, cfg):.0%} of fraud</span>'
            else:
                badge = '<span class="badge new">new fraud pattern</span>'
            if m in results["triggered"].alarms:
                badge += '<span class="badge alarm">drift alarm, retraining</span>'
            status_ph.markdown(f"<div><b>{unit.capitalize()} {m}</b>&nbsp;&nbsp;{badge}</div>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(11, 4.6))
            plot_metric(shown, cfg, "pr_auc", ax=ax, title=f"PR-AUC as the stream arrives · {unit} {m}",
                        xlabel=unit.capitalize())
            fig.tight_layout()
            chart_ph.pyplot(fig, width="stretch")
            plt.close(fig)

            html = ['<div class="cards">']
            for k, r in results.items():
                row = r.monthly.iloc[i]
                html.append(
                    f'<div class="card" style="--c:{COLORS[k]}"><div class="name">{r.label}</div>'
                    f'<div class="big">{row.pr_auc:.3f}<small>PR-AUC</small></div>'
                    f'<div class="row"><span>Caught</span><b>{int(row.tp)}</b></div>'
                    f'<div class="row"><span>Missed</span><b>{int(row.fn)}</b></div>'
                    f'<div class="row"><span>False alarms</span><b>{int(row.fp)}</b></div></div>'
                )
            html.append("</div>")
            cards_ph.markdown("".join(html), unsafe_allow_html=True)

            if stream is not None:
                sample = stream[stream.month == m].head(6)[[c for c in sample_cols if c in stream]]
                feed_ph.dataframe(sample, width="stretch", hide_index=True)
            time.sleep(speed)
        st.success("Stream finished. The Benchmark tab has the full summary and costs.")


# ----------------------------------------------------------------------------
# tab 3: data
# ----------------------------------------------------------------------------
with tab_data:
    if source == "ieee":
        st.markdown("#### Where the data comes from")
        st.markdown(
            '<p class="lede" style="font-size:.95rem">Real e-commerce card transactions provided by '
            "<b>Vesta Corporation</b> for the IEEE Computational Intelligence Society fraud detection competition "
            "(Kaggle, 2019). Only the labelled training file is used: 590,540 transactions, about 3.5% fraud, "
            "covering roughly six months. Most columns are anonymised by the provider (card1–6, C1–14, D1–15, "
            "M1–9). The timeline is cut into two-week periods. Unlike the synthetic stream, the fraud patterns "
            "here change naturally, so nobody knows exactly when drift happens.</p>",
            unsafe_allow_html=True,
        )
        stream, features = cached_ieee()
        per = stream.groupby("month").agg(transactions=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"))
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.bar(per.index, per.fraud_rate, color="#1F3A5F", width=0.68)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax.set_title("Fraud rate per two-week period")
            ax.set_xlabel("Two-week period")
            show(fig)
        with c2:
            prod = [c for c in features if c.startswith("ProductCD=")]
            fraud = stream[stream.is_fraud == 1]
            mix = fraud.groupby("month")[prod].mean()
            fig, ax = plt.subplots(figsize=(6, 3.3))
            bottom = None
            shades = ["#1F3A5F", "#3F6FA8", "#7FA3CC", "#B9CCE3", "#DCE5F0"]
            for c, col in zip(prod, shades):
                ax.bar(mix.index, mix[c], bottom=bottom, color=col, width=0.68, label=c.split("=")[1])
                bottom = mix[c] if bottom is None else bottom + mix[c]
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title("Fraud by product type (natural drift)")
            ax.set_xlabel("Two-week period")
            ax.legend(title="ProductCD", loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, title_fontsize=8)
            show(fig)
        st.markdown(f'<p class="note">{len(features)} features are used: amount, hour of day, card and address '
                    "fields, distances, counting features C1–C14, time-delta features D1–D4/D10/D11/D15, match "
                    "flags M1–M6, e-mail domains, product type and device type. Missing values are set to -1 and "
                    "categories are encoded without using the labels.</p>", unsafe_allow_html=True)
        st.markdown("#### Sample")
        st.dataframe(stream.head(200), width="stretch", hide_index=True, height=280)
    elif source == "kaggle":
        st.markdown(
            "#### Kaggle credit-card data\n"
            "Real card transactions made by European cardholders over two days in September 2013, released by "
            "the Machine Learning Group of Université Libre de Bruxelles (Dal Pozzolo et al., 2015). "
            "284,807 transactions, 492 frauds (0.17%). Features V1–V28 are PCA components (the original "
            "features are confidential), plus the amount. Here the stream is cut into 12 equal time slices."
        )
    else:
        st.markdown("#### Where the data comes from")
        st.markdown(
            '<p class="lede" style="font-size:.95rem">The transactions are <b>synthetic</b>: they are generated by '
            "this project's own code (<code>fraud_drift/data.py</code>), not taken from any bank. There are no real "
            "customers in it. Simulating the data is what makes the experiment possible: we decide exactly when "
            "and how fraud changes, so we can measure how fast each strategy notices and recovers. With a real "
            "dataset the moment of drift is unknown. The same seed always produces the same stream.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="datagrid">
  <div class="panel">
    <h4>Normal transactions (never change)</h4>
    <ul>
      <li><b>amount</b>: log-normal, median about €20</li>
      <li><b>hour</b>: uniform over the day, 0–23</li>
      <li><b>n_txn_1h</b>: transactions in the last hour, Poisson with mean 1.5</li>
      <li><b>foreign</b>: foreign merchant, 15% of transactions</li>
      <li><b>device_new</b>: new device, 20% of transactions</li>
    </ul>
  </div>
  <div class="panel">
    <h4>Fraud (a fixed share each month, default 5%)</h4>
    <p><b>Pattern A, old:</b> the amount is multiplied by 3–6 and the payment happens between 00:00 and 04:59.</p>
    <p><b>Pattern B, new:</b> the amount is cut to 30–70%, 4–8 extra transactions in the last hour, always a foreign
    merchant and a new device.</p>
    <p>Only the look of fraud changes, not normal behaviour, which is what makes it <i>concept</i> drift.</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        stream = cached_stream(cfg.to_dict())
        fraud = stream[stream.is_fraud == 1]
        share = fraud.groupby("month")["pattern"].apply(lambda s: (s == "B").mean())

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.bar(share.index, share.values, color="#1F3A5F", width=0.68)
            ax.set_xticks(range(0, cfg.n_months, 2))
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title("Share of fraud using the new pattern B")
            ax.set_xlabel("Month")
            show(fig)
        with c2:
            avg = fraud.groupby("month")[["amount", "n_txn_1h"]].mean()
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.plot(avg.index, avg.amount, color=INK, lw=2, marker="o", ms=4, label="average fraud amount (EUR)")
            ax2 = ax.twinx()
            ax2.plot(avg.index, avg.n_txn_1h, color=MUTED, lw=2, ls="--", label="transactions in last hour")
            ax2.spines["right"].set_visible(True)
            ax2.grid(False)
            ax.set_xticks(range(0, cfg.n_months, 2))
            ax.set_title("What an average fraud looks like")
            ax.set_xlabel("Month")
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc="upper left", bbox_to_anchor=(0, -0.2),
                      ncol=2, fontsize=8.5)
            show(fig)

        st.markdown("#### Sample of the generated stream")
        st.dataframe(stream.head(200), width="stretch", hide_index=True, height=280)
        st.download_button("Download the full stream (CSV)", stream.to_csv(index=False),
                           f"synthetic_stream_seed{cfg.seed}.csv", "text/csv")
        st.markdown(
            '<p class="note">A realism check on real data is built in: put the public Kaggle '
            "<code>creditcard.csv</code> (ULB, 284,807 transactions) in the <code>data/</code> folder and a second "
            "data source appears in the sidebar.</p>",
            unsafe_allow_html=True,
        )

st.markdown('<p class="note" style="font-size:.8rem;margin-top:2.5rem;border-top:1px solid #E3E1DB;padding-top:.8rem">'
            "Kacper Matejczuk · M.Sc. Big Data Science, Kozminski University · "
            "When Fraud Patterns Change: Benchmarking Drift Adaptation Methods for Financial Transactions</p>",
            unsafe_allow_html=True)
