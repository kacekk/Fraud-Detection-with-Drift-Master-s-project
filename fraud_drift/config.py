"""Experiment settings.

Every run is described by one ExperimentConfig. The same config (including
`seed`) always produces exactly the same data and exactly the same results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DRIFT_TYPES = ("none", "sudden", "gradual", "recurring")
DETECTORS = ("adwin", "ddm")
MONITORS = ("fraud_recall", "error_rate")
ONLINE_MODELS = ("logreg", "arf")


@dataclass
class ExperimentConfig:
    # --- data stream -------------------------------------------------------
    per_month: int = 5000            # transactions generated per month
    n_months: int = 18               # length of the stream
    warmup_months: int = 2           # historical, fully labelled months used for the first training
    fraud_rate: float = 0.05         # share of fraudulent transactions
    drift_type: str = "sudden"       # none | sudden | gradual | recurring
    drift_month: int = 6             # month when the new fraud pattern appears
    transition_months: int = 4       # gradual drift: months until the new pattern fully takes over
    recur_month: int = 12            # recurring drift: month when the old pattern comes back

    # --- models ------------------------------------------------------------
    n_trees: int = 100               # Random Forest size (static, sliding, triggered)
    window: int = 2                  # months of labelled data used for each retraining
    detector: str = "adwin"          # adwin | ddm
    adwin_delta: float = 0.002       # ADWIN sensitivity (smaller = fewer alarms)
    monitor: str = "fraud_recall"    # what the detector watches: misses among confirmed frauds, or all errors
    online_model: str = "logreg"     # logreg | arf (Adaptive Random Forest, slower)
    threshold: float = 0.5           # probability above which a transaction is flagged

    # --- realism -----------------------------------------------------------
    label_delay: int = 0             # months until fraud labels (chargebacks) arrive

    # --- cost model (in EUR) -------------------------------------------------
    investigation_cost: float = 15.0   # cost of checking one false alarm
    retraining_cost: float = 500.0     # cost of one model retraining (compute + validation)

    # --- reproducibility -------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------------
    def validate(self) -> "ExperimentConfig":
        if self.drift_type not in DRIFT_TYPES:
            raise ValueError(f"drift_type must be one of {DRIFT_TYPES}")
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {MONITORS}")
        if self.online_model not in ONLINE_MODELS:
            raise ValueError(f"online_model must be one of {ONLINE_MODELS}")
        if not 1 <= self.warmup_months < self.n_months:
            raise ValueError("warmup_months must be between 1 and n_months - 1")
        if self.drift_type != "none" and not self.warmup_months <= self.drift_month < self.n_months:
            raise ValueError("drift_month must lie after the warm-up and inside the stream")
        if self.drift_type == "recurring" and not self.drift_month < self.recur_month < self.n_months:
            raise ValueError("recur_month must lie between drift_month and n_months")
        if not 0 < self.fraud_rate < 0.5:
            raise ValueError("fraud_rate must be between 0 and 0.5")
        if self.label_delay < 0 or self.window < 1 or self.transition_months < 1:
            raise ValueError("label_delay >= 0, window >= 1 and transition_months >= 1 are required")
        return self

    @property
    def eval_months(self) -> list[int]:
        return list(range(self.warmup_months, self.n_months))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
