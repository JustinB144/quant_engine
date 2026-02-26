"""
ML Diagnostics — feature importance drift and ensemble disagreement.

Monitors the stability of model feature rankings across retraining
periods and measures the degree of consensus among ensemble members.

Feature importance drift: if top features change significantly between
retraining windows, the model's learned relationships are unstable.

Ensemble disagreement: if individual models in an ensemble produce
uncorrelated predictions, the ensemble may be unreliable.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from ..config import (
    EVAL_FEATURE_DRIFT_THRESHOLD,
    EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD,
)

logger = logging.getLogger(__name__)


def feature_importance_drift(
    importance_matrices: Dict[str, np.ndarray],
    top_k: int = 10,
    drift_threshold: float = EVAL_FEATURE_DRIFT_THRESHOLD,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Detect feature importance drift across retraining periods.

    Computes Spearman rank correlation of feature importance rankings
    between consecutive retraining periods.

    Parameters
    ----------
    importance_matrices : dict[str, np.ndarray]
        Mapping from period label (e.g., date string) to 1-D array
        of feature importances.  Each array must have the same length.
    top_k : int
        Number of top features to track for stability analysis.
    drift_threshold : float
        Spearman correlation below this between consecutive periods
        flags significant drift.
    feature_names : list[str], optional
        Feature names for reporting (same order as importance arrays).

    Returns
    -------
    dict
        Keys: drift_detected (bool), correlations_per_period (list[float]),
        mean_correlation, min_correlation, top_k_stability (float),
        top_k_features_per_period (dict), n_periods.
    """
    periods = sorted(importance_matrices.keys())
    n_periods = len(periods)

    if n_periods < 2:
        return {
            "drift_detected": False,
            "correlations_per_period": [],
            "mean_correlation": 1.0,
            "min_correlation": 1.0,
            "top_k_stability": 1.0,
            "top_k_features_per_period": {},
            "n_periods": n_periods,
        }

    correlations = []
    top_k_per_period = {}
    top_k_change_count = 0

    for i in range(n_periods):
        imp = np.asarray(importance_matrices[periods[i]], dtype=float).ravel()

        # Track top-k features
        top_indices = np.argsort(imp)[::-1][:top_k]
        if feature_names and len(feature_names) == len(imp):
            top_k_per_period[periods[i]] = [feature_names[j] for j in top_indices]
        else:
            top_k_per_period[periods[i]] = top_indices.tolist()

        if i == 0:
            continue

        prev_imp = np.asarray(importance_matrices[periods[i - 1]], dtype=float).ravel()

        if len(imp) != len(prev_imp):
            logger.warning(
                "Feature importance length mismatch: %d vs %d for periods %s → %s",
                len(prev_imp), len(imp), periods[i - 1], periods[i],
            )
            correlations.append(0.0)
            continue

        # Spearman rank correlation of importance rankings
        if sp_stats is not None and len(imp) > 2:
            corr, _ = sp_stats.spearmanr(imp, prev_imp)
            corr = float(corr) if np.isfinite(corr) else 0.0
        else:
            corr = _manual_spearman(imp, prev_imp)
        correlations.append(corr)

        # Count top-k changes
        prev_top = set(np.argsort(prev_imp)[::-1][:top_k].tolist())
        curr_top = set(np.argsort(imp)[::-1][:top_k].tolist())
        n_changed = len(prev_top - curr_top)
        top_k_change_count += n_changed

    if not correlations:
        return {
            "drift_detected": False,
            "correlations_per_period": [],
            "mean_correlation": 1.0,
            "min_correlation": 1.0,
            "top_k_stability": 1.0,
            "top_k_features_per_period": top_k_per_period,
            "n_periods": n_periods,
        }

    mean_corr = float(np.mean(correlations))
    min_corr = float(np.min(correlations))

    # Top-k stability: fraction of top-k features that stayed the same across periods
    max_changes = top_k * (n_periods - 1)
    top_k_stability = 1.0 - (top_k_change_count / max_changes) if max_changes > 0 else 1.0

    drift_detected = min_corr < drift_threshold or mean_corr < drift_threshold

    return {
        "drift_detected": drift_detected,
        "correlations_per_period": correlations,
        "mean_correlation": mean_corr,
        "min_correlation": min_corr,
        "top_k_stability": top_k_stability,
        "top_k_features_per_period": top_k_per_period,
        "n_periods": n_periods,
    }


def ensemble_disagreement(
    predictions: Dict[str, np.ndarray],
    disagreement_threshold: float = EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD,
) -> Dict:
    """Measure disagreement among ensemble model predictions.

    Computes pairwise Spearman rank correlation among all model
    predictions.  High disagreement (low correlation) indicates
    the ensemble members are learning different patterns.

    Parameters
    ----------
    predictions : dict[str, np.ndarray]
        Mapping from model name to prediction array.  All arrays
        must have the same length.
    disagreement_threshold : float
        Pairwise correlation below this flags disagreement.

    Returns
    -------
    dict
        Keys: mean_correlation, min_correlation, max_correlation,
        disagreement_pairs (list[tuple]), n_models,
        high_disagreement (bool), pairwise_correlations (dict).
    """
    model_names = sorted(predictions.keys())
    n_models = len(model_names)

    if n_models < 2:
        return {
            "mean_correlation": 1.0,
            "min_correlation": 1.0,
            "max_correlation": 1.0,
            "disagreement_pairs": [],
            "n_models": n_models,
            "high_disagreement": False,
            "pairwise_correlations": {},
        }

    pairwise = {}
    all_corrs = []
    disagreement_pairs = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_i, name_j = model_names[i], model_names[j]
            pred_i = np.asarray(predictions[name_i], dtype=float).ravel()
            pred_j = np.asarray(predictions[name_j], dtype=float).ravel()

            min_len = min(len(pred_i), len(pred_j))
            pred_i = pred_i[:min_len]
            pred_j = pred_j[:min_len]

            valid = np.isfinite(pred_i) & np.isfinite(pred_j)
            if valid.sum() < 3:
                corr = 0.0
            elif sp_stats is not None:
                c, _ = sp_stats.spearmanr(pred_i[valid], pred_j[valid])
                corr = float(c) if np.isfinite(c) else 0.0
            else:
                corr = _manual_spearman(pred_i[valid], pred_j[valid])

            pair_key = f"{name_i}_vs_{name_j}"
            pairwise[pair_key] = corr
            all_corrs.append(corr)

            if corr < disagreement_threshold:
                disagreement_pairs.append((name_i, name_j, corr))

    if not all_corrs:
        return {
            "mean_correlation": 0.0,
            "min_correlation": 0.0,
            "max_correlation": 0.0,
            "disagreement_pairs": [],
            "n_models": n_models,
            "high_disagreement": True,
            "pairwise_correlations": pairwise,
        }

    return {
        "mean_correlation": float(np.mean(all_corrs)),
        "min_correlation": float(np.min(all_corrs)),
        "max_correlation": float(np.max(all_corrs)),
        "disagreement_pairs": [(a, b, round(c, 4)) for a, b, c in disagreement_pairs],
        "n_models": n_models,
        "high_disagreement": len(disagreement_pairs) > 0,
        "pairwise_correlations": pairwise,
    }


def _manual_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Fallback Spearman rank correlation without scipy."""
    if len(a) < 3:
        return 0.0
    rx = pd.Series(a).rank().values.astype(float)
    ry = pd.Series(b).rank().values.astype(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(rx * ry) / denom)
