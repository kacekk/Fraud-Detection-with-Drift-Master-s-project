"""IEEE-CIS loader: runs on a stand-in file with the real column names (see fake_ieee.py)."""

from dataclasses import replace

import pandas as pd

from fake_ieee import make_fake_ieee
from fraud_drift import ExperimentConfig, results_fingerprint, run_experiment
from fraud_drift.ieee import assign_periods, is_available, load_ieee, prepare


def _write(tmp_path, n=20_000):
    tx, ident = make_fake_ieee(n)
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    ident.to_csv(tmp_path / "train_identity.csv", index=False)
    return tx, ident


def test_periods_are_two_weeks_and_tail_is_merged():
    dt = pd.Series([86_400, 86_400 * 15, 86_400 * 183.5])   # last point: 0.5 days into period 13
    assert list(assign_periods(dt, 14)) == [0, 1, 12]


def test_prepare_shapes_and_no_missing(tmp_path):
    tx, ident = make_fake_ieee(5_000)
    stream, features = prepare(tx, ident)
    assert len(stream) == len(tx)
    assert stream[features].isna().sum().sum() == 0
    assert stream.month.min() == 0 and stream.month.max() == 12
    assert "amount" in features and "DeviceType=mobile" in features
    assert stream.TransactionID.is_monotonic_increasing


def test_cache_gives_identical_stream(tmp_path):
    _write(tmp_path)
    assert is_available(tmp_path)
    first, f1 = load_ieee(tmp_path)
    second, f2 = load_ieee(tmp_path)          # served from the parquet cache
    assert f1 == f2
    pd.testing.assert_frame_equal(first, second)


def test_pipeline_runs_and_is_reproducible(tmp_path):
    _write(tmp_path)
    stream, features = load_ieee(tmp_path)
    cfg = ExperimentConfig(drift_type="none", n_months=int(stream.month.max()) + 1, n_trees=20)
    a = run_experiment(cfg, stream=stream, features=features)
    b = run_experiment(cfg, stream=stream, features=features)
    assert results_fingerprint(a) == results_fingerprint(b)
    assert len(a["static"].monthly) == cfg.n_months - cfg.warmup_months


def test_forest_does_not_depend_on_cores():
    cfg = ExperimentConfig(per_month=800, n_months=6, drift_month=3, n_trees=20)
    from fraud_drift.strategies import _fit_forest
    from fraud_drift import build_stream, SYNTHETIC_FEATURES

    s = build_stream(cfg)
    p1 = _fit_forest(s, SYNTHETIC_FEATURES, cfg).predict_proba(s[SYNTHETIC_FEATURES].to_numpy())
    p2 = _fit_forest(s, SYNTHETIC_FEATURES, replace(cfg)).predict_proba(s[SYNTHETIC_FEATURES].to_numpy())
    assert (p1 == p2).all()
