"""The Streamlit app starts and runs the benchmark without errors."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _small(at):
    for s in at.sidebar.slider:
        if s.label == "Transactions per month":
            s.set_value(1000)
        elif s.label == "Number of months":
            s.set_value(8)
        elif s.label == "Random Forest trees":
            s.set_value(20)
    return at.run(timeout=120)


def test_app_loads():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception


def test_app_runs_benchmark():
    at = _small(AppTest.from_file(APP, default_timeout=120).run())
    at.button(key="run_compare").click().run(timeout=180)
    assert not at.exception
    assert len(at.dataframe) >= 1
