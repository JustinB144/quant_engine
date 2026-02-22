"""
Walk-Forward Model Selection — expanding-window hyperparameter search
with Deflated Sharpe Ratio penalty for multiple-testing correction.

Architecture:
    - Expanding-window walk-forward split (respecting temporal ordering)
    - Train all configurations in each fold
    - Track OOS Spearman correlation per config across folds
    - Select config with best average OOS performance
    - Apply DSR penalty to penalise excessive search
"""
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover
    sp_stats = None


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (scipy optional)."""
    if sp_stats is not None:
        corr, _ = sp_stats.spearmanr(x, y)
        return float(corr) if np.isfinite(corr) else 0.0

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 2:
        return 0.0
    rx = pd.Series(xa[mask]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(ya[mask]).rank(method="average").to_numpy(dtype=float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _expanding_walk_forward_folds(
    dates: pd.Series,
    n_folds: int,
    horizon: int,
) -> List[tuple]:
    """Generate expanding-window walk-forward folds using unique dates.

    Returns a list of (train_idx, test_idx) tuples where indices reference
    positions in the original Series.
    """
    unique_dates = np.array(sorted(pd.Index(dates.unique())))
    n_dates = len(unique_dates)
    if n_dates < (n_folds + 1) * 2:
        return []

    fold_size = max(1, n_dates // (n_folds + 1))
    purge_gap = horizon
    folds = []

    for i in range(n_folds):
        test_start = (i + 1) * fold_size
        test_end = min(test_start + fold_size, n_dates)

        train_end = max(0, test_start - purge_gap)
        if train_end <= 0 or test_start >= test_end:
            continue

        train_dates = set(unique_dates[:train_end])
        test_dates = set(unique_dates[test_start:test_end])

        train_idx = np.flatnonzero(dates.isin(train_dates).to_numpy())
        test_idx = np.flatnonzero(dates.isin(test_dates).to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))

    return folds


def _extract_dates(index: pd.Index) -> pd.Series:
    """Return row-aligned timestamps from an index (supports panel MultiIndex)."""
    if isinstance(index, pd.MultiIndex):
        dates = pd.to_datetime(index.get_level_values(-1))
    else:
        dates = pd.to_datetime(index)
    return pd.Series(dates, index=index)


def walk_forward_select(
    features: pd.DataFrame,
    targets: pd.Series,
    regimes: pd.Series,
    param_grid: Dict[str, List],
    n_folds: int = 5,
    horizon: int = 10,
) -> Dict[str, Any]:
    """Select the best model configuration via walk-forward cross-validation.

    For each fold in an expanding-window walk-forward split, all
    configurations from *param_grid* are trained and evaluated on OOS
    Spearman rank correlation.  The configuration with the best average
    OOS correlation (after Deflated Sharpe Ratio penalty) is returned.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (rows = observations, possibly panel-indexed).
    targets : pd.Series
        Prediction targets aligned with *features*.
    regimes : pd.Series
        Regime labels aligned with *features*.  Currently unused for
        selection but reserved for future regime-conditional grids.
    param_grid : Dict[str, List]
        Hyperparameter search space.  Each key maps to a list of values.
        Example: ``{"n_estimators": [200, 500], "max_depth": [3, 5]}``.
    n_folds : int, default 5
        Number of walk-forward folds.
    horizon : int, default 10
        Prediction horizon (used for purge gap between train/test folds).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``best_params``: dict of the best hyperparameter combination
        - ``best_score``: adjusted average OOS Spearman correlation
        - ``raw_score``: unadjusted average OOS Spearman correlation
        - ``n_trials``: total number of (config x fold) trials evaluated
        - ``dsr_penalty``: the Deflated Sharpe Ratio penalty factor applied
        - ``fold_scores``: per-fold OOS scores for the best config
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for walk_forward_select. "
            "Install it before calling this function."
        )

    # Drop rows with missing targets
    valid = targets.notna() & features.notna().any(axis=1)
    X = features[valid].copy()
    y = targets[valid].copy()

    dates = _extract_dates(X.index)
    n_obs = len(X)

    # Generate expanding-window folds
    folds = _expanding_walk_forward_folds(dates, n_folds=n_folds, horizon=horizon)
    if not folds:
        # Fallback: return default params
        default_params = {k: v[0] for k, v in param_grid.items()}
        return {
            "best_params": default_params,
            "best_score": 0.0,
            "raw_score": 0.0,
            "n_trials": 0,
            "dsr_penalty": 1.0,
            "fold_scores": [],
        }

    # Enumerate all parameter combinations
    param_keys = sorted(param_grid.keys())
    param_combos = list(product(*(param_grid[k] for k in param_keys)))
    n_configs = len(param_combos)

    # Track OOS scores per config across folds
    # Shape: (n_configs, n_folds)
    scores_matrix = np.full((n_configs, len(folds)), np.nan)
    n_trials = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        for cfg_idx, combo in enumerate(param_combos):
            params = dict(zip(param_keys, combo))

            # Build GBR with this config
            model_params = {
                "n_estimators": params.get("n_estimators", 500),
                "max_depth": params.get("max_depth", 4),
                "min_samples_leaf": params.get("min_samples_leaf", 30),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
                "random_state": 42,
            }

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train.values)
            X_te_scaled = scaler.transform(X_test.values)

            model = GradientBoostingRegressor(**model_params)
            model.fit(X_tr_scaled, y_train.values)

            preds = model.predict(X_te_scaled)
            oos_corr = _spearmanr(y_test.values, preds)
            scores_matrix[cfg_idx, fold_idx] = oos_corr
            n_trials += 1

    # Compute average OOS score per config (ignoring NaN folds)
    avg_scores = np.nanmean(scores_matrix, axis=1)

    # Apply Deflated Sharpe Ratio penalty:
    # adjusted_score = raw_score * (1 - max(0, (n_trials - 1) / (2 * n_obs)))
    n_trials_total = n_configs * n_folds
    dsr_penalty_factor = max(0.0, (n_trials_total - 1) / (2.0 * max(1, n_obs)))
    dsr_multiplier = 1.0 - min(1.0, dsr_penalty_factor)
    adjusted_scores = avg_scores * dsr_multiplier

    # Select best config
    best_idx = int(np.nanargmax(adjusted_scores))
    best_combo = param_combos[best_idx]
    best_params = dict(zip(param_keys, best_combo))
    best_fold_scores = scores_matrix[best_idx, :].tolist()

    return {
        "best_params": best_params,
        "best_score": float(adjusted_scores[best_idx]),
        "raw_score": float(avg_scores[best_idx]),
        "n_trials": n_trials,
        "dsr_penalty": float(dsr_multiplier),
        "fold_scores": [float(s) if np.isfinite(s) else None for s in best_fold_scores],
    }
