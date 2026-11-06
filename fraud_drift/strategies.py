"""The four drift-adaptation strategies compared in the thesis.

static     - Random Forest trained once on the warm-up months, never updated
sliding    - Random Forest retrained every month on the last `window` labelled months (blind)
triggered  - Random Forest retrained only when a drift detector (ADWIN / DDM) raises an alarm (informed).
             By default the detector watches the stream of confirmed frauds (missed = 1), because
             with rare fraud the overall error rate barely reacts (cf. Wang et al., 2013, DDM-OCI).
online     - model that learns after every labelled transaction (river: logistic regression or ARF)

Label delay: when predicting month m, only months < m - label_delay have
labels (fraud is confirmed weeks later via chargebacks). The warm-up months are
historical data and always count as labelled.

Every model gets `random_state` / `seed` from the config and runs single-
threaded, so results are identical on every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from river import drift, forest, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier

from .config import ExperimentConfig
from .metrics import add_costs, month_metrics

STRATEGIES = {
    "static": "Static baseline",
    "sliding": "Sliding window",
    "triggered": "Detector-triggered",
    "online": "Online learning",
}


@dataclass
class StrategyResult:
    key: str
    label: str
    monthly: pd.DataFrame
    alarms: list[int] = field(default_factory=list)

    @property
    def n_retrainings(self) -> int:
        return int(self.monthly["retrained"].sum())


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _labelled_until(m: int, cfg: ExperimentConfig) -> int:
    """Exclusive upper bound of months whose labels are known when month m starts."""
    return max(m - cfg.label_delay, cfg.warmup_months)


def _months(stream: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    return stream[(stream["month"] >= start) & (stream["month"] < stop)]


def _fit_forest(train: pd.DataFrame, features: list[str], cfg: ExperimentConfig) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=cfg.n_trees, random_state=cfg.seed, n_jobs=1)
    clf.fit(train[features].to_numpy(), train["is_fraud"].to_numpy())
    return clf


def _predict(clf, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    return clf.predict_proba(data[features].to_numpy())[:, 1]


def _row(m, test, proba, cfg, retrained=False, alarm=False) -> dict:
    r = {"month": m, **month_metrics(test["is_fraud"], proba, test["amount"], cfg.threshold)}
    r["retrained"] = retrained
    r["alarm"] = alarm
    return r


def _finish(key, rows, cfg, alarms=None) -> StrategyResult:
    monthly = add_costs(pd.DataFrame(rows), cfg.investigation_cost, cfg.retraining_cost)
    return StrategyResult(key, STRATEGIES[key], monthly, alarms or [])


def _make_detector(cfg: ExperimentConfig):
    if cfg.detector == "adwin":
        return drift.ADWIN(delta=cfg.adwin_delta)
    return drift.binary.DDM()


def _make_online(cfg: ExperimentConfig):
    if cfg.online_model == "arf":
        return forest.ARFClassifier(n_models=10, seed=cfg.seed)
    return preprocessing.StandardScaler() | linear_model.LogisticRegression()


# ----------------------------------------------------------------------------
# strategies
# ----------------------------------------------------------------------------
def run_static(stream, features, cfg) -> StrategyResult:
    clf = _fit_forest(_months(stream, 0, cfg.warmup_months), features, cfg)
    rows = []
    for m in cfg.eval_months:
        test = _months(stream, m, m + 1)
        rows.append(_row(m, test, _predict(clf, test, features), cfg))
    return _finish("static", rows, cfg)


def run_sliding(stream, features, cfg) -> StrategyResult:
    rows = []
    for m in cfg.eval_months:
        stop = _labelled_until(m, cfg)
        clf = _fit_forest(_months(stream, max(0, stop - cfg.window), stop), features, cfg)
        test = _months(stream, m, m + 1)
        rows.append(_row(m, test, _predict(clf, test, features), cfg, retrained=True))
    return _finish("sliding", rows, cfg)


def run_triggered(stream, features, cfg) -> StrategyResult:
    clf = _fit_forest(_months(stream, 0, cfg.warmup_months), features, cfg)
    detector = _make_detector(cfg)
    errors_by_month: dict[int, np.ndarray] = {}
    rows, alarms = [], []
    retrain_now = False

    for m in cfg.eval_months:
        test = _months(stream, m, m + 1)
        proba = _predict(clf, test, features)
        rows.append(_row(m, test, proba, cfg, retrained=retrain_now))
        retrain_now = False
        y = test["is_fraud"].to_numpy()
        wrong = ((proba >= cfg.threshold).astype(int) != y).astype(int)
        # With ~5% fraud, the overall error rate hardly moves when fraud changes, so by
        # default the detector watches only confirmed fraud cases (1 = missed), i.e. recall.
        errors_by_month[m] = wrong[y == 1] if cfg.monitor == "fraud_recall" else wrong

        # End of month m: labels for month m - label_delay have now arrived.
        k = m - cfg.label_delay
        alarm = False
        for e in errors_by_month.get(k, ()):
            detector.update(int(e))
            if detector.drift_detected:
                alarm = True
        if alarm:
            alarms.append(m)
            rows[-1]["alarm"] = True
            stop = _labelled_until(m + 1, cfg)
            clf = _fit_forest(_months(stream, max(0, stop - cfg.window), stop), features, cfg)
            detector = _make_detector(cfg)      # new model -> start monitoring from scratch
            retrain_now = True                  # the new model is used from next month on

    return _finish("triggered", rows, cfg, alarms)


def run_online(stream, features, cfg) -> StrategyResult:
    model = _make_online(cfg)
    cols = features + ["is_fraud"]

    def learn(df):
        for r in df[cols].to_dict("records"):
            model.learn_one({f: float(r[f]) for f in features}, int(r["is_fraud"]))

    learn(_months(stream, 0, cfg.warmup_months))
    learned_until = cfg.warmup_months
    rows = []

    for m in cfg.eval_months:
        test = _months(stream, m, m + 1)
        proba = []
        if cfg.label_delay == 0:
            # prequential: predict each transaction, then learn its label
            for r in test[cols].to_dict("records"):
                x = {f: float(r[f]) for f in features}
                proba.append(model.predict_proba_one(x).get(1, 0.0))
                model.learn_one(x, int(r["is_fraud"]))
        else:
            stop = _labelled_until(m, cfg)
            if stop > learned_until:
                learn(_months(stream, learned_until, stop))
                learned_until = stop
            for r in test[features].to_dict("records"):
                proba.append(model.predict_proba_one({f: float(r[f]) for f in features}).get(1, 0.0))
        rows.append(_row(m, test, np.array(proba), cfg))

    return _finish("online", rows, cfg)


RUNNERS = {
    "static": run_static,
    "sliding": run_sliding,
    "triggered": run_triggered,
    "online": run_online,
}
