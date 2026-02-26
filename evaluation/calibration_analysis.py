"""
Calibration Analysis — prediction confidence calibration diagnostics.

Uses the existing ``ConfidenceCalibrator`` from ``models/calibration.py``
plus the ``compute_ece`` and ``compute_reliability_curve`` helpers to
assess whether model confidence scores are well-calibrated.

Overconfidence detection: flags when the maximum gap between predicted
and actual percentiles exceeds a threshold (default 20%).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import EVAL_CALIBRATION_BINS, EVAL_OVERCONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def analyze_calibration(
    predictions: np.ndarray,
    returns: pd.Series,
    confidence_scores: Optional[np.ndarray] = None,
    bins: int = EVAL_CALIBRATION_BINS,
    overconfidence_threshold: float = EVAL_OVERCONFIDENCE_THRESHOLD,
) -> Dict:
    """Analyze calibration of model predictions and confidence scores.

    If ``confidence_scores`` are provided, calibrates those directly against
    actual outcomes (prediction direction correct or not).  Otherwise, uses
    the prediction magnitude as a proxy for confidence and calibrates
    against the actual return percentile.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (continuous values).
    returns : pd.Series
        Actual forward returns aligned to predictions.
    confidence_scores : np.ndarray, optional
        Model-provided confidence scores in [0, 1].
    bins : int
        Number of calibration bins.
    overconfidence_threshold : float
        Maximum allowed gap between predicted and actual percentile.

    Returns
    -------
    dict
        Keys: calibration_error, ece, overconfident, max_gap, reliability_curve,
        n_samples.
    """
    pred_arr = np.asarray(predictions, dtype=float).ravel()
    ret_arr = np.asarray(returns, dtype=float).ravel()
    min_len = min(len(pred_arr), len(ret_arr))
    pred_arr = pred_arr[:min_len]
    ret_arr = ret_arr[:min_len]

    # Filter NaN
    valid = np.isfinite(pred_arr) & np.isfinite(ret_arr)
    pred_arr = pred_arr[valid]
    ret_arr = ret_arr[valid]
    n = len(pred_arr)

    if n < bins * 2:
        return _empty_calibration_result()

    # Determine confidence and outcomes
    if confidence_scores is not None:
        conf_arr = np.asarray(confidence_scores, dtype=float).ravel()[:min_len]
        conf_arr = conf_arr[valid]

        if len(conf_arr) != n:
            conf_arr = np.abs(pred_arr) / (np.abs(pred_arr).max() + 1e-12)

        # Binary outcome: did the prediction get the direction right?
        outcomes = ((pred_arr > 0) & (ret_arr > 0)) | ((pred_arr <= 0) & (ret_arr <= 0))
        outcomes = outcomes.astype(float)
    else:
        # Use prediction percentile as confidence proxy
        pred_rank = pd.Series(pred_arr).rank(pct=True).values
        conf_arr = pred_rank

        # Outcome: return percentile
        outcomes = pd.Series(ret_arr).rank(pct=True).values

    # Import calibration helpers
    from ..models.calibration import compute_ece, compute_reliability_curve

    # ECE
    ece = compute_ece(conf_arr, outcomes, n_bins=bins)

    # Reliability curve
    reliability = compute_reliability_curve(conf_arr, outcomes, n_bins=bins)

    # Calibration error: mean squared deviation from diagonal
    avg_pred = np.array(reliability["avg_predicted"], dtype=float)
    obs_freq = np.array(reliability["observed_freq"], dtype=float)
    bin_counts = np.array(reliability["bin_counts"], dtype=int)

    valid_bins = ~np.isnan(avg_pred) & ~np.isnan(obs_freq) & (bin_counts > 0)
    if valid_bins.sum() > 0:
        gaps = np.abs(avg_pred[valid_bins] - obs_freq[valid_bins])
        calibration_error = float(np.mean(gaps ** 2))
        max_gap = float(np.max(gaps))
    else:
        calibration_error = 0.0
        max_gap = 0.0

    overconfident = max_gap > overconfidence_threshold

    return {
        "calibration_error": calibration_error,
        "ece": ece,
        "overconfident": overconfident,
        "max_gap": max_gap,
        "reliability_curve": reliability,
        "n_samples": n,
        "overconfidence_threshold": overconfidence_threshold,
    }


def _empty_calibration_result() -> Dict:
    """Return empty calibration result for insufficient data."""
    return {
        "calibration_error": 0.0,
        "ece": 0.0,
        "overconfident": False,
        "max_gap": 0.0,
        "reliability_curve": {
            "bin_centers": [],
            "observed_freq": [],
            "avg_predicted": [],
            "bin_counts": [],
        },
        "n_samples": 0,
        "overconfidence_threshold": EVAL_OVERCONFIDENCE_THRESHOLD,
    }
