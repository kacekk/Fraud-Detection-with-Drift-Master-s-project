"""Sanity checks: the simulation and the strategies behave as the thesis expects."""

from dataclasses import replace

import numpy as np
import pytest

from fraud_drift import ExperimentConfig, build_stream, run_experiment, summary_table
from fraud_drift.data import share_of_new_pattern

CFG = ExperimentConfig(per_month=1500, n_months=10, drift_month=5, recur_month=8, transition_months=3, n_trees=40)


def test_fraud_rate_and_patterns():
    s = build_stream(CFG)
    assert len(s) == CFG.per_month * CFG.n_months
    assert s.is_fraud.mean() == pytest.approx(CFG.fraud_rate, abs=1e-9)
    fraud = s[s.is_fraud == 1]
    assert set(fraud[fraud.month < CFG.drift_month].pattern) == {"A"}
    assert set(fraud[fraud.month >= CFG.drift_month].pattern) == {"B"}


def test_gradual_and_recurring_schedules():
    g = replace(CFG, drift_type="gradual")
    assert [share_of_new_pattern(m, g) for m in range(4, 9)] == pytest.approx([0, 1 / 3, 2 / 3, 1, 1])
    r = replace(CFG, drift_type="recurring")
    assert [share_of_new_pattern(m, r) for m in (4, 5, 7, 8, 9)] == [0, 1, 1, 0, 0]


def test_static_model_breaks_and_adaptive_models_recover():
    res = run_experiment(CFG)
    end = lambda k: res[k].monthly.tail(2).pr_auc.mean()  # noqa: E731  (last two months)
    assert end("static") < 0.3
    for k in ("sliding", "triggered", "online"):
        assert end(k) > 0.7, k
    assert res["triggered"].alarms, "the detector should notice a sudden drift"


@pytest.mark.parametrize("detector", ["adwin", "ddm"])
@pytest.mark.parametrize("monitor", ["fraud_recall", "error_rate"])
@pytest.mark.parametrize("delay", [0, 2])
def test_detectors_and_label_delay_run(detector, monitor, delay):
    cfg = replace(CFG, detector=detector, monitor=monitor, label_delay=delay)
    res = run_experiment(cfg, strategies=["triggered", "online"])
    for r in res.values():
        assert len(r.monthly) == len(cfg.eval_months)
        assert np.isfinite(r.monthly.total_cost).all()


def test_label_delay_slows_recovery():
    fast = run_experiment(CFG, strategies=["sliding"])["sliding"].monthly
    slow = run_experiment(replace(CFG, label_delay=2), strategies=["sliding"])["sliding"].monthly
    m = CFG.drift_month + 1
    assert slow.query("month == @m").pr_auc.item() < fast.query("month == @m").pr_auc.item()


def test_summary_table_columns():
    t = summary_table(run_experiment(CFG), CFG)
    assert list(t.Strategy) == ["Static baseline", "Sliding window", "Detector-triggered", "Online learning"]
    assert {"Recovery delay (months)", "Total cost (EUR)", "PR-AUC after drift"} <= set(t.columns)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        replace(CFG, drift_type="weird").validate()
    with pytest.raises(ValueError):
        replace(CFG, drift_month=CFG.n_months).validate()
