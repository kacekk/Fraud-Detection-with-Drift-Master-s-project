"""fraud_drift - benchmarking drift adaptation methods for fraud detection.

Code for the M.Sc. thesis "When Fraud Patterns Change: Benchmarking Drift
Adaptation Methods for Financial Transactions" (Kozminski University).
"""

from .config import ExperimentConfig
from .data import PATTERNS, SYNTHETIC_FEATURES, build_stream, load_kaggle
from .experiment import environment_info, results_fingerprint, run_experiment, summary_table
from .strategies import STRATEGIES

__version__ = "1.0.0"

__all__ = [
    "ExperimentConfig",
    "PATTERNS",
    "STRATEGIES",
    "SYNTHETIC_FEATURES",
    "build_stream",
    "environment_info",
    "load_kaggle",
    "results_fingerprint",
    "run_experiment",
    "summary_table",
]
