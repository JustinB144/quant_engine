"""
Evaluation Layer (Truth Engine) — Spec 08.

Regime-aware performance slicing, ML diagnostics, fragility analysis,
calibration validation, and walk-forward evaluation with embargo.
"""

from .slicing import PerformanceSlice, SliceRegistry
from .metrics import compute_slice_metrics, decile_spread
from .fragility import (
    pnl_concentration,
    drawdown_distribution,
    recovery_time_distribution,
    detect_critical_slowing_down,
)
from .ml_diagnostics import feature_importance_drift, ensemble_disagreement
from .calibration_analysis import analyze_calibration
from .engine import EvaluationEngine

__all__ = [
    "PerformanceSlice",
    "SliceRegistry",
    "compute_slice_metrics",
    "decile_spread",
    "pnl_concentration",
    "drawdown_distribution",
    "recovery_time_distribution",
    "detect_critical_slowing_down",
    "feature_importance_drift",
    "ensemble_disagreement",
    "analyze_calibration",
    "EvaluationEngine",
]
