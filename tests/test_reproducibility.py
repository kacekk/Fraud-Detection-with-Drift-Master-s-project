"""Same config + same seed must give exactly the same results, every time and on every OS."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fraud_drift import ExperimentConfig, build_stream, results_fingerprint, run_experiment

ROOT = Path(__file__).resolve().parents[1]
SMALL = ExperimentConfig(per_month=800, n_months=8, drift_month=4, recur_month=6, transition_months=2, n_trees=20)


def test_stream_is_identical_for_same_seed():
    pd.testing.assert_frame_equal(build_stream(SMALL), build_stream(SMALL))


def test_stream_changes_with_seed():
    assert not build_stream(SMALL).equals(build_stream(replace(SMALL, seed=SMALL.seed + 1)))


@pytest.mark.parametrize("drift", ["sudden", "gradual", "recurring"])
def test_results_identical_on_rerun(drift):
    cfg = replace(SMALL, drift_type=drift)
    a, b = run_experiment(cfg), run_experiment(cfg)
    for key in a:
        pd.testing.assert_frame_equal(a[key].monthly, b[key].monthly)
        assert a[key].alarms == b[key].alarms
    assert results_fingerprint(a) == results_fingerprint(b)


def test_matches_reference_fingerprints(tmp_path):
    """The quick benchmark must reproduce the fingerprints stored in the repo."""
    import run_experiment

    reference = json.loads((ROOT / "tests" / "reference_fingerprints.json").read_text(encoding="utf-8"))
    got = run_experiment.main(["--quick", "--out", str(tmp_path)])
    for name, fp in got.items():
        assert fp == reference[name], f"{name}: got {fp}, expected {reference[name]}"
