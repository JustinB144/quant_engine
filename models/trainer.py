"""
Model Trainer — trains regime-conditional gradient boosting ensemble.

Architecture:
    1. One global model trained on all data
    2. One model per regime (trending_bull, trending_bear, mean_reverting, high_vol)
    3. At prediction time: blend regime-specific + global based on confidence

Anti-overfitting:
    - Purged date-grouped CV folds (gap = prediction horizon to prevent leakage)
    - Embargo period after purge gap
    - Scaler fit INSIDE each CV fold (no leakage from test fold statistics)
    - Permutation importance on held-out validation portion
    - IS/OOS gap monitoring
    - Holdout set never touched during training
    - Adaptive max features based on sample size
    - Median imputation (not zero-fill)
    - Shallow trees (max_depth=4) + subsampling
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover - optional dependency fallback
    sp_stats = None
try:
    from scipy.optimize import minimize as sp_minimize
except ImportError:  # pragma: no cover - optional dependency fallback
    sp_minimize = None
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - optional dependency fallback
    GradientBoostingRegressor = None
    RandomForestRegressor = None
    ElasticNet = None
    permutation_importance = None
    StandardScaler = None

    def r2_score(y_true, y_pred):
        """r2 score."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot <= 1e-12:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def mean_absolute_error(y_true, y_pred):
        """mean absolute error."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

# Optional boosting libraries — graceful fallback if not installed
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

from ..config import (
    MODEL_PARAMS, MAX_FEATURES_SELECTED, MAX_IS_OOS_GAP,
    CV_FOLDS, HOLDOUT_FRACTION, MODEL_DIR, MIN_REGIME_SAMPLES,
    MIN_REGIME_DAYS,
    REGIME_NAMES, RECENCY_DECAY, REGIME_SOFT_ASSIGNMENT_THRESHOLD,
    ENSEMBLE_DIVERSIFY, WF_MAX_TRAIN_DATES,
    STRUCTURAL_WEIGHT_ENABLED, STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY,
    STRUCTURAL_WEIGHT_JUMP_PENALTY, STRUCTURAL_WEIGHT_STRESS_PENALTY,
)
from .feature_stability import FeatureStabilityTracker
from .versioning import ModelVersion, ModelRegistry


class IdentityScaler:
    """No-op scaler that passes data through unchanged.

    Drop-in replacement for StandardScaler when the model handles its own
    scaling internally (e.g., DiverseEnsemble with per-constituent scalers).
    """

    def fit(self, X, y=None):
        """Fit the transformer to the provided data."""
        return self

    def transform(self, X):
        """Transform the provided data using the fitted parameters."""
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        """Fit the transformer and return transformed data."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        """Inverse-transform data back to the original scale."""
        return np.asarray(X)


class DiverseEnsemble:
    """
    Lightweight ensemble wrapper that combines predictions from multiple
    (model, scaler_or_none) pairs using optimized weights.  Exposes a
    ``predict`` method so it is a drop-in replacement for any sklearn
    estimator in the downstream pipeline.
    """

    def __init__(
        self,
        estimators: List[Tuple[object, Optional[object]]],
        weights: Optional[np.ndarray] = None,
    ):
        """
        Args:
            estimators: list of (model, scaler_or_none) tuples.
                        If scaler is not None, X is transformed through it
                        before calling model.predict.
            weights: Optional array of non-negative weights that sum to 1.
                     If None, equal weighting is used.
        """
        self.estimators = estimators
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        else:
            n = len(estimators)
            self.weights = np.full(n, 1.0 / n) if n > 0 else np.array([])
        # Expose feature_importances_ from the first estimator that has it,
        # so downstream code (e.g. feature importance logging) keeps working.
        self.feature_importances_ = self._aggregate_feature_importances()

    def _aggregate_feature_importances(self) -> np.ndarray:
        """Weighted average of feature_importances_ across estimators that expose it."""
        arrays = []
        imp_weights = []
        for i, (model, _) in enumerate(self.estimators):
            if hasattr(model, "feature_importances_"):
                arrays.append(np.asarray(model.feature_importances_))
                imp_weights.append(self.weights[i] if i < len(self.weights) else 1.0)
        if not arrays:
            return np.array([])
        # All arrays must have the same length (same feature set).
        imp_weights = np.asarray(imp_weights, dtype=float)
        if imp_weights.sum() > 0:
            imp_weights = imp_weights / imp_weights.sum()
        else:
            imp_weights = np.full(len(imp_weights), 1.0 / len(imp_weights))
        return np.average(arrays, weights=imp_weights, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted average of all constituent model predictions."""
        preds = []
        for model, scaler in self.estimators:
            X_in = scaler.transform(X) if scaler is not None else X
            preds.append(model.predict(X_in))
        return np.average(preds, weights=self.weights, axis=0)


@dataclass
class TrainResult:
    """Result of training a single model."""
    model_name: str
    n_samples: int
    n_features: int
    selected_features: List[str]
    cv_scores: List[float]  # OOS Spearman correlation per fold
    cv_train_scores: List[float]  # IS Spearman correlation per fold
    cv_gap: float  # mean IS - mean OOS
    holdout_r2: float
    holdout_mae: float
    holdout_correlation: float  # Spearman rank correlation
    feature_importance: Dict[str, float]
    target_std: float = 0.0  # std of training targets (for prediction clipping)
    feature_medians: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Result of training the full regime-conditional ensemble."""
    global_model: TrainResult
    regime_models: Dict[int, TrainResult]
    horizon: int
    train_time_seconds: float
    total_features: int
    total_samples: int


class ModelTrainer:
    """
    Trains a regime-conditional gradient boosting ensemble for
    forward return prediction.
    """

    def __init__(
        self,
        model_params: Optional[dict] = None,
        max_features: int = MAX_FEATURES_SELECTED,
        cv_folds: int = CV_FOLDS,
        holdout_fraction: float = HOLDOUT_FRACTION,
        max_gap: float = MAX_IS_OOS_GAP,
    ):
        """Initialize ModelTrainer."""
        # Truth Layer: validate execution contract preconditions
        from ..validation.preconditions import enforce_preconditions
        enforce_preconditions()

        self.model_params = model_params or MODEL_PARAMS.copy()
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.holdout_fraction = holdout_fraction
        self.max_gap = max_gap

    @staticmethod
    def _spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Spearman correlation with optional SciPy dependency.
        """
        if sp_stats is not None:
            corr, pval = sp_stats.spearmanr(x, y)
            return float(corr), float(pval)

        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        mask = np.isfinite(xa) & np.isfinite(ya)
        if mask.sum() < 2:
            return np.nan, np.nan
        rx = pd.Series(xa[mask]).rank(method="average").to_numpy(dtype=float)
        ry = pd.Series(ya[mask]).rank(method="average").to_numpy(dtype=float)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
        if denom <= 1e-12:
            return np.nan, np.nan
        corr = float(np.sum(rx * ry) / denom)
        return corr, np.nan

    @staticmethod
    def _require_sklearn() -> None:
        """Fail fast when training is requested without scikit-learn."""
        if GradientBoostingRegressor is None or StandardScaler is None:
            raise ImportError(
                "scikit-learn is required for model training. "
                "Install it in the runtime environment before training.",
            )

    @staticmethod
    def _extract_dates(index: pd.Index) -> pd.Series:
        """Return row-aligned timestamps from an index (supports panel MultiIndex)."""
        if isinstance(index, pd.MultiIndex):
            dates = pd.to_datetime(index.get_level_values(-1))
        else:
            dates = pd.to_datetime(index)
        return pd.Series(dates, index=index)

    @classmethod
    def _sort_panel_by_time(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray], pd.Series]:
        """Sort panel rows by date (then security id) to enforce deterministic chronology."""
        dates = cls._extract_dates(X.index)
        if isinstance(X.index, pd.MultiIndex):
            ids = X.index.get_level_values(0).astype(str).to_numpy()
            order = np.lexsort((ids, dates.to_numpy()))
        else:
            order = np.argsort(dates.to_numpy())

        X_sorted = X.iloc[order]
        y_sorted = y.iloc[order]
        w_sorted = sample_weights[order] if sample_weights is not None else None
        dates_sorted = dates.iloc[order]
        return X_sorted, y_sorted, w_sorted, dates_sorted

    @staticmethod
    def _temporal_holdout_masks(
        dates: pd.Series,
        holdout_fraction: float,
        min_dev_rows: int = 200,
        min_hold_rows: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split by unique dates (not raw rows) to avoid cross-asset temporal leakage.
        """
        unique_dates = np.array(sorted(pd.Index(dates.unique())))
        if len(unique_dates) < 10:
            split = max(min_dev_rows, int(len(dates) * (1 - holdout_fraction)))
            split = min(split, max(1, len(dates) - min_hold_rows))
            dev = np.zeros(len(dates), dtype=bool)
            dev[:split] = True
            hold = ~dev
            return dev, hold

        n_hold_dates = max(1, int(len(unique_dates) * holdout_fraction))
        hold_date_set = set(unique_dates[-n_hold_dates:])
        hold = dates.isin(hold_date_set).to_numpy()
        dev = ~hold

        # Guard rails for very small samples.
        if dev.sum() < min_dev_rows or hold.sum() < min_hold_rows:
            split = max(min_dev_rows, int(len(dates) * (1 - holdout_fraction)))
            split = min(split, max(1, len(dates) - min_hold_rows))
            dev = np.zeros(len(dates), dtype=bool)
            dev[:split] = True
            hold = ~dev

        return dev, hold

    @staticmethod
    def _date_purged_folds(
        dates: pd.Series,
        n_folds: int,
        purge_gap: int,
        embargo: int,
        max_train_dates: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward folds using unique dates for panel-safe CV.

        By default the training window expands (all dates up to the fold
        boundary).  When *max_train_dates* is set, the training window
        rolls forward so that only the most recent *max_train_dates*
        unique dates are included, preventing concept-drift from stale
        data and better simulating periodic retraining.
        """
        unique_dates = np.array(sorted(pd.Index(dates.unique())))
        n_dates = len(unique_dates)
        if n_dates < (n_folds + 1) * 2:
            return []

        fold_size = max(1, n_dates // (n_folds + 1))
        folds: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_folds):
            test_start = (i + 1) * fold_size
            test_end = min(test_start + fold_size, n_dates)

            train_end = max(0, test_start - purge_gap)
            embargoed_start = min(test_start + embargo, test_end)
            if train_end <= 0 or embargoed_start >= test_end:
                continue

            # Rolling window: cap training dates to most recent max_train_dates
            train_start = 0
            if max_train_dates is not None and train_end > max_train_dates:
                train_start = train_end - max_train_dates

            train_dates = set(unique_dates[train_start:train_end])
            test_dates = set(unique_dates[embargoed_start:test_end])

            train_idx = np.flatnonzero(dates.isin(train_dates).to_numpy())
            test_idx = np.flatnonzero(dates.isin(test_dates).to_numpy())
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            folds.append((train_idx, test_idx))

        return folds

    @staticmethod
    def _prune_correlated_features(
        X: pd.DataFrame,
        threshold: float = 0.80,
        use_vif: bool = False,
    ) -> List[str]:
        """Remove highly correlated features before permutation importance.

        When two features have |correlation| >= ``threshold``, the one with
        lower absolute mean is dropped.  This prevents permutation importance
        from splitting credit among correlated features and erroneously
        dropping useful predictors.

        The threshold was tightened from 0.90 to 0.80 to remove features
        in the 0.80-0.89 correlation range that cause:
        - Feature importance instability (correlated features split importance)
        - Gradient boosting inefficiency (splits on redundant features)

        Args:
            X: Feature DataFrame.
            threshold: Correlation threshold (default 0.80).
            use_vif: If True, additionally prune features with VIF > 10
                (Variance Inflation Factor) to detect multicollinearity
                not captured by pairwise correlation.
        """
        if X.shape[1] <= 1:
            return X.columns.tolist()

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop: set = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] >= threshold].tolist()
            for c in correlated:
                if c in to_drop:
                    continue
                # Drop the feature with lower absolute mean (less informative)
                if abs(X[col].mean()) >= abs(X[c].mean()):
                    to_drop.add(c)
                else:
                    to_drop.add(col)

        # Optional VIF check for multicollinearity beyond pairwise correlation
        if use_vif and (X.shape[1] - len(to_drop)) > 5:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X_remaining = X.drop(columns=list(to_drop))
                # Replace any NaN/inf for VIF computation
                X_vif = X_remaining.fillna(X_remaining.median())
                X_vif = X_vif.replace([np.inf, -np.inf], 0.0)
                for i, col_name in enumerate(X_vif.columns):
                    try:
                        vif = variance_inflation_factor(X_vif.values, i)
                        if vif > 10:
                            to_drop.add(col_name)
                    except (np.linalg.LinAlgError, ValueError):
                        continue
            except ImportError:
                pass  # statsmodels not available; skip VIF

        kept = [c for c in X.columns if c not in to_drop]
        return kept if kept else X.columns[:1].tolist()

    def _select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_feats: int,
    ) -> Tuple[List[str], pd.Series]:
        """
        Feature selection: correlation pruning then permutation importance.

        Step 1: Remove features with |correlation| >= 0.80 to prevent
                permutation importance from splitting credit.
        Step 2: Train a lightweight GBR on a temporal split.
        Step 3: Compute permutation importance on the validation set.
        """
        self._require_sklearn()
        if len(X_train) < 80 or X_train.shape[1] == 0:
            cols = X_train.columns[:max(1, min(max_feats, X_train.shape[1]))].tolist()
            return cols, pd.Series(0.0, index=X_train.columns)

        # Step 1: Prune highly correlated features
        kept_cols = self._prune_correlated_features(X_train)
        X_pruned = X_train[kept_cols]

        split = int(len(X_pruned) * 0.8)
        split = min(max(split, 50), max(50, len(X_pruned) - 20))
        X_sel_train, X_sel_val = X_pruned.iloc[:split], X_pruned.iloc[split:]
        y_sel_train, y_sel_val = y_train.iloc[:split], y_train.iloc[split:]

        selector = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=30,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        selector.fit(X_sel_train.values, y_sel_train.values)

        if len(X_sel_val) >= 20:
            try:
                perm = permutation_importance(
                    selector,
                    X_sel_val.values,
                    y_sel_val.values,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=1,
                )
                importances = pd.Series(perm.importances_mean, index=X_pruned.columns)
            except (ValueError, RuntimeError):
                # Fall back to tree-based importance if permutation fails
                importances = pd.Series(selector.feature_importances_, index=X_pruned.columns)
        else:
            importances = pd.Series(selector.feature_importances_, index=X_pruned.columns)

        # Expand importances to full column set (pruned columns get 0)
        full_importances = pd.Series(0.0, index=X_train.columns)
        full_importances[importances.index] = importances

        selected = importances[importances > 0].nlargest(max_feats).index.tolist()
        if len(selected) == 0:
            selected = importances.abs().nlargest(max(1, min(max_feats, X_pruned.shape[1]))).index.tolist()

        min_required = min(5, max(1, X_pruned.shape[1]))
        if len(selected) < min_required:
            fallback = importances.abs().nlargest(min_required).index.tolist()
            selected = list(dict.fromkeys(selected + fallback))

        return selected[:max_feats], full_importances

    def _compute_stable_features(
        self,
        fold_feature_sets: List[List[str]],
        max_feats: int,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        stability_threshold: float = 0.80,
    ) -> List[str]:
        """Compute stable features selected in ≥ stability_threshold fraction of folds.

        Features that consistently appear across multiple independent CV folds
        are more likely to generalize than features selected by a single fold.

        Args:
            fold_feature_sets: List of feature name lists, one per CV fold.
            max_feats: Maximum number of features to return.
            X_dev: Development features (used for fallback importance ranking).
            y_dev: Development targets (used for fallback importance ranking).
            stability_threshold: Minimum fraction of folds a feature must appear
                in to be considered "stable". Default 0.80 (≥80% of folds).

        Returns:
            List of stable feature names, capped at max_feats.
        """
        from collections import Counter

        n_folds = len(fold_feature_sets)
        if n_folds == 0:
            return []

        # Count how many folds selected each feature
        feature_counts = Counter()
        for feat_list in fold_feature_sets:
            for feat in feat_list:
                feature_counts[feat] += 1

        min_folds = max(1, int(np.ceil(n_folds * stability_threshold)))
        stable = [
            feat for feat, count in feature_counts.most_common()
            if count >= min_folds
        ]

        if len(stable) >= 3:
            return stable[:max_feats]

        # Fallback: if fewer than 3 stable features, relax to features
        # appearing in ≥50% of folds, then rank by frequency
        relaxed_min = max(1, n_folds // 2)
        relaxed = [
            feat for feat, count in feature_counts.most_common()
            if count >= relaxed_min
        ]

        if len(relaxed) >= 3:
            return relaxed[:max_feats]

        # Last resort: use global selection on full dev set
        selected, _ = self._select_features(X_dev, y_dev, max_feats=max_feats)
        return selected

    @staticmethod
    def _compute_structural_weights(
        index: pd.Index,
        shock_vectors: Dict,
        changepoint_penalty: float = STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY,
        jump_penalty: float = STRUCTURAL_WEIGHT_JUMP_PENALTY,
        stress_penalty: float = STRUCTURAL_WEIGHT_STRESS_PENALTY,
    ) -> np.ndarray:
        """Compute per-sample weights from ShockVector structural signals.

        Samples near regime changepoints, jumps, or high systemic stress
        are downweighted to reduce the influence of structurally unstable
        periods on model training.  Samples in confident, stable regimes
        are upweighted.

        Weight formula per sample i:
            w_i = confidence_factor * changepoint_factor * jump_factor * stress_factor

        Where:
            confidence_factor  = hmm_confidence  (in [0, 1])
            changepoint_factor = 1 - penalty * bocpd_changepoint_prob
            jump_factor        = (1 - penalty) if jump_detected else 1.0
            stress_factor      = 1 - penalty * systemic_stress

        Parameters
        ----------
        index : pd.Index
            Row index of the training data (used to align shock vectors).
        shock_vectors : dict
            Mapping from index key to ShockVector.  Keys may be timestamps
            or (ticker, timestamp) tuples matching the training index.
        changepoint_penalty : float
            Maximum weight reduction near changepoints.
        jump_penalty : float
            Weight multiplier for jump events (0.0=exclude, 1.0=no penalty).
        stress_penalty : float
            Maximum weight reduction for high systemic stress.

        Returns
        -------
        np.ndarray
            Per-sample weights of shape ``(len(index),)``, normalised
            to mean 1.0.
        """
        n = len(index)
        weights = np.ones(n, dtype=float)

        if not shock_vectors:
            return weights

        # Build a lookup-friendly set of shock vectors.
        # The training index may be a MultiIndex (ticker, date) or a simple
        # DatetimeIndex.  shock_vectors may be keyed by date, (ticker, date),
        # or similar.  We try several match strategies.
        for i in range(n):
            idx_val = index[i]

            # Try direct lookup first
            sv = shock_vectors.get(idx_val)

            # For MultiIndex rows: try the date component (last level)
            if sv is None and isinstance(index, pd.MultiIndex):
                date_val = index.get_level_values(-1)[i]
                sv = shock_vectors.get(date_val)

                # Try (ticker, date) tuple
                if sv is None and index.nlevels >= 2:
                    ticker_val = index.get_level_values(0)[i]
                    sv = shock_vectors.get((ticker_val, date_val))

            if sv is None:
                continue

            # Confidence factor: upweight high-confidence regime states
            confidence_factor = float(
                getattr(sv, "hmm_confidence", 0.5)
            )
            confidence_factor = max(0.1, confidence_factor)  # floor at 0.1

            # Changepoint factor: downweight near regime transitions
            cp_prob = float(getattr(sv, "bocpd_changepoint_prob", 0.0))
            changepoint_factor = 1.0 - changepoint_penalty * cp_prob

            # Jump factor: downweight jump events (noisy, outlier-driven)
            jump_detected = bool(getattr(sv, "jump_detected", False))
            jump_factor = (1.0 - jump_penalty) if jump_detected else 1.0

            # Stress factor: downweight during systemic stress
            structural = getattr(sv, "structural_features", {}) or {}
            systemic_stress = float(structural.get("systemic_stress", 0.0))
            stress_factor = 1.0 - stress_penalty * systemic_stress

            weights[i] = (
                confidence_factor
                * changepoint_factor
                * jump_factor
                * stress_factor
            )

        # Clip to prevent negative or zero weights
        weights = np.clip(weights, 0.05, 10.0)

        # Normalise to mean 1.0 so downstream sample_weight interpretation
        # is consistent with the recency-weight path.
        w_sum = weights.sum()
        if w_sum > 1e-10:
            weights = weights * (n / w_sum)

        return weights

    def train_ensemble(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        regimes: pd.Series,
        regime_probabilities: Optional[pd.DataFrame] = None,
        horizon: int = 10,
        verbose: bool = True,
        versioned: bool = True,
        survivorship_mode: bool = False,
        universe_as_of: Optional[str] = None,
        recency_weight: bool = False,
        shock_vectors: Optional[Dict] = None,
    ) -> EnsembleResult:
        """
        Train the full regime-conditional ensemble.

        Args:
            versioned: Save to timestamped versioned directory + register
            survivorship_mode: Universe was constructed with survivorship-bias-free data
            universe_as_of: Point-in-time universe date (for versioning metadata)
            recency_weight: Apply exponential recency weighting to training samples
            shock_vectors: Optional dict of ShockVectors keyed by index value or
                (ticker, date) tuple.  When provided and STRUCTURAL_WEIGHT_ENABLED,
                samples are weighted by structural state signals (BOCPD changepoint
                probability, jump flags, systemic stress, HMM confidence).
        """
        t0 = time.time()

        # Drop rows with NaN targets
        valid = targets.notna() & features.notna().any(axis=1)
        X = features[valid].copy()
        y = targets[valid].copy()
        r = regimes[valid].copy()
        rp = regime_probabilities.loc[valid].copy() if regime_probabilities is not None else None

        # Compute sample weights for recency weighting
        sample_weights = None
        structural_weights_applied = False
        train_dates = self._extract_dates(X.index)
        train_data_start = str(train_dates.min().date()) if len(train_dates) > 0 else None
        train_data_end = str(train_dates.max().date()) if len(train_dates) > 0 else None
        if recency_weight:
            # Exponential decay: weight = exp(-λ * days_from_most_recent)
            max_date = train_dates.max()
            days_ago = (max_date - train_dates).dt.days
            sample_weights = np.exp(-RECENCY_DECAY * days_ago.values)

        # ── Structural sample weighting (SPEC_03 T4) ──
        # Downweight samples near regime changepoints, jumps, and high stress.
        if STRUCTURAL_WEIGHT_ENABLED and shock_vectors:
            struct_weights = self._compute_structural_weights(
                index=X.index,
                shock_vectors=shock_vectors,
            )
            # Combine with recency weights multiplicatively
            if sample_weights is not None:
                sample_weights = sample_weights * struct_weights
            else:
                sample_weights = struct_weights
            structural_weights_applied = True

        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING ENSEMBLE — {horizon}d forward return")
            print(f"{'='*60}")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Regimes: {r.value_counts().to_dict()}")
            if recency_weight:
                print(f"  Recency weighting: ON (λ={RECENCY_DECAY})")
            if structural_weights_applied:
                matched = int(np.sum(sample_weights != 1.0)) if sample_weights is not None else 0
                print(f"  Structural weighting: ON ({matched} samples matched)")
            if survivorship_mode:
                print(f"  Survivorship-bias-free universe")

        # ── Step 1: Train global model ──
        if verbose:
            print(f"\n── Global Model ──")
        global_result, global_model, global_scaler, global_features = self._train_single(
            X, y, model_name="global", horizon=horizon, verbose=verbose,
            sample_weights=sample_weights,
        )

        if global_result is None:
            if verbose:
                print("\n  Global model was REJECTED by quality gates. "
                      "Aborting ensemble training.")
            return EnsembleResult(
                global_model=None,
                regime_models={},
                horizon=horizon,
                train_time_seconds=time.time() - t0,
                total_features=X.shape[1],
                total_samples=len(X),
            )

        # ── Step 2: Train per-regime models ──
        regime_results = {}
        regime_models = {}
        regime_scalers = {}
        regime_feature_sets = {}
        skipped_regimes = {}  # Explicitly track skipped regimes with reasons

        for regime_code, regime_name in REGIME_NAMES.items():
            if rp is not None and f"regime_prob_{regime_code}" in rp.columns:
                prob = rp[f"regime_prob_{regime_code}"].reindex(X.index).fillna(0.0)
                mask = prob >= REGIME_SOFT_ASSIGNMENT_THRESHOLD
            else:
                prob = None
                mask = r == regime_code
            n_samples = mask.sum()

            # Check both minimum samples and minimum distinct days (SPEC_10 T7)
            if mask.any():
                n_days = mask.index[mask].nunique() if hasattr(mask, 'index') else n_samples
            else:
                n_days = 0

            if n_samples < MIN_REGIME_SAMPLES or n_days < MIN_REGIME_DAYS:
                skip_reason = (
                    f"insufficient_samples ({n_samples} < {MIN_REGIME_SAMPLES})"
                    if n_samples < MIN_REGIME_SAMPLES
                    else f"insufficient_days ({n_days} < {MIN_REGIME_DAYS})"
                )
                if verbose:
                    print(f"\n── {regime_name} — SKIPPED ({skip_reason}) ──")
                    print(f"  Global model will be used for regime {regime_code} predictions.")
                skipped_regimes[regime_code] = {
                    "name": regime_name,
                    "n_samples": int(n_samples),
                    "n_days": int(n_days),
                    "min_required_samples": MIN_REGIME_SAMPLES,
                    "min_required_days": MIN_REGIME_DAYS,
                    "reason": skip_reason,
                }
                regime_models[regime_code] = None  # Explicitly mark as unavailable
                continue

            if verbose:
                print(f"\n── {regime_name} ({n_samples} samples) ──")

            # Adaptive max features based on sample size
            adaptive_max = min(self.max_features, n_samples // 20)
            if adaptive_max < 5:
                if verbose:
                    print(f"  SKIPPED: too few samples for feature count (need {5 * 20}+)")
                continue

            regime_weights = sample_weights[mask.values] if sample_weights is not None else None
            if prob is not None:
                p_vals = prob[mask].values
                if regime_weights is None:
                    regime_weights = p_vals
                else:
                    regime_weights = regime_weights * p_vals
            result, model, scaler, selected = self._train_single(
                X[mask], y[mask], model_name=regime_name,
                horizon=horizon, max_features_override=adaptive_max,
                verbose=verbose, sample_weights=regime_weights,
            )

            if result is not None:
                regime_results[regime_code] = result
                regime_models[regime_code] = model
                regime_scalers[regime_code] = scaler
                regime_feature_sets[regime_code] = selected

        elapsed = time.time() - t0

        ensemble_result = EnsembleResult(
            global_model=global_result,
            regime_models=regime_results,
            horizon=horizon,
            train_time_seconds=elapsed,
            total_features=X.shape[1],
            total_samples=len(X),
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training complete in {elapsed:.1f}s")
            self._print_summary(ensemble_result, regimes=r, targets=y)

        # Save artifacts — only if global model passed quality gates
        if global_result is not None and global_model is not None:
            self._save(
                global_model, global_scaler, global_features, global_result,
                regime_models, regime_scalers, regime_feature_sets, regime_results,
                horizon, ensemble_result,
                versioned=versioned,
                survivorship_mode=survivorship_mode,
                universe_as_of=universe_as_of,
                train_data_start=train_data_start,
                train_data_end=train_data_end,
                structural_weights_applied=structural_weights_applied,
            )
            # Fit and save confidence calibrator on holdout predictions vs outcomes
            cal_metrics = self._fit_calibrator(
                global_model, global_scaler, global_features, global_result,
                X, y, horizon, versioned,
            )
            if cal_metrics and verbose:
                ece_str = (f"{cal_metrics['ece']:.4f}"
                           if cal_metrics.get('ece') is not None else "N/A")
                print(f"  Calibration — ECE: {ece_str} "
                      f"(fit={cal_metrics['n_cal_fit']}, "
                      f"val={cal_metrics['n_cal_val']})")
        elif verbose:
            print("  Skipping model save — global model was rejected by quality gates.")

        # ── Record feature importances for stability monitoring ──
        if global_result is not None and global_result.feature_importance:
            try:
                tracker = FeatureStabilityTracker()
                feat_names = list(global_result.feature_importance.keys())
                feat_values = np.array(
                    [global_result.feature_importance[f] for f in feat_names]
                )
                cycle_id = time.strftime("%Y%m%dT%H%M%S")
                tracker.record_importance(cycle_id, feat_values, feat_names)
                stability = tracker.check_stability()
                if verbose:
                    print(f"\n  Feature stability — cycles recorded: {stability.n_cycles}")
                    if stability.spearman_vs_previous is not None:
                        print(f"  Spearman vs previous: {stability.spearman_vs_previous:.4f}")
                    if stability.alert:
                        print(f"  WARNING: {stability.alert_message}")
            except (OSError, ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  Feature stability tracking skipped: {e}")

        return ensemble_result

    def _train_single(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        horizon: int = 10,
        max_features_override: Optional[int] = None,
        verbose: bool = True,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[TrainResult], object, object, List[str]]:
        """
        Train a single gradient boosting model with feature selection.

        Key anti-leakage measures:
        - Median imputation (not fillna(0))
        - Permutation importance on held-out validation split
        - StandardScaler fit inside each CV fold
        - Purge gap = horizon between train and test folds
        - Embargo period after purge gap
        """
        warnings = []
        self._require_sklearn()
        max_feats = max_features_override or self.max_features

        # Sort panel chronologically by date before any split.
        X, y, sample_weights, dates = self._sort_panel_by_time(X, y, sample_weights)

        # Store target std for prediction clipping
        target_std = float(y.std())

        # ── Holdout split by date (panel-safe) — BEFORE imputation ──
        dev_mask, hold_mask = self._temporal_holdout_masks(
            dates=dates,
            holdout_fraction=self.holdout_fraction,
        )
        dev_idx = np.flatnonzero(dev_mask)
        hold_idx = np.flatnonzero(hold_mask)

        # ── Impute NaN with DEV-ONLY medians (no holdout leakage) ──
        dev_medians = X.iloc[dev_idx].median()
        feature_medians = dev_medians.to_dict()
        X_clean = X.fillna(dev_medians)

        X_dev, X_hold = X_clean.iloc[dev_idx], X_clean.iloc[hold_idx]
        y_dev, y_hold = y.iloc[dev_idx], y.iloc[hold_idx]
        dev_dates = dates.iloc[dev_idx]
        w_dev = sample_weights[dev_idx] if sample_weights is not None else None

        if len(X_dev) < 100:
            warnings.append("Insufficient dev samples after temporal split")
            return None, None, None, []

        # ── Cross-validation with panel-safe purged date folds ──
        purge_gap = horizon
        embargo = max(1, horizon // 2)

        n_unique_dates = len(pd.Index(dev_dates.unique()))
        n_folds = min(self.cv_folds, max(2, n_unique_dates // 120))
        folds = self._date_purged_folds(
            dates=dev_dates,
            n_folds=n_folds,
            purge_gap=purge_gap,
            embargo=embargo,
            max_train_dates=WF_MAX_TRAIN_DATES,
        )

        if verbose:
            print(f"  Per-fold feature selection ({X_dev.shape[1]} candidates, "
                  f"{len(folds)} folds)...", flush=True)

        cv_oos_scores = []
        cv_is_scores = []
        fold_feature_sets: List[List[str]] = []

        for fold, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) < 100 or len(test_idx) < 20:
                continue

            # Per-fold feature selection: each fold independently selects
            # its best features from its own training data, preventing
            # fold-1 selection bias from propagating to all folds.
            fold_selected, _ = self._select_features(
                X_dev.iloc[train_idx],
                y_dev.iloc[train_idx],
                max_feats=max_feats,
            )
            if len(fold_selected) < 3:
                continue

            fold_feature_sets.append(fold_selected)

            # Scale features INSIDE the CV fold (fit only on training fold)
            fold_scaler = StandardScaler()
            X_train_fold = fold_scaler.fit_transform(X_dev.iloc[train_idx][fold_selected].values)
            X_test_fold = fold_scaler.transform(X_dev.iloc[test_idx][fold_selected].values)

            model = self._make_model()
            fold_weights = w_dev[train_idx] if w_dev is not None else None
            model.fit(X_train_fold, y_dev.values[train_idx], sample_weight=fold_weights)

            is_pred = model.predict(X_train_fold)
            oos_pred = model.predict(X_test_fold)

            # Use Spearman rank correlation (robust to outliers, appropriate for finance)
            is_corr, _ = self._spearmanr(y_dev.values[train_idx], is_pred)
            oos_corr, _ = self._spearmanr(y_dev.values[test_idx], oos_pred)

            cv_is_scores.append(float(is_corr) if not np.isnan(is_corr) else 0.0)
            cv_oos_scores.append(float(oos_corr) if not np.isnan(oos_corr) else 0.0)

        # ── Compute stable features: selected in ≥80% of CV folds ──
        # This prevents any single fold from dictating the final feature set.
        if fold_feature_sets:
            selected = self._compute_stable_features(
                fold_feature_sets, max_feats, X_dev, y_dev,
            )
        else:
            # Fallback to global selection if no valid folds
            selected, _ = self._select_features(X_dev, y_dev, max_feats=max_feats)

        if len(selected) < 5:
            # Insufficient signal — refuse to train rather than selecting noise
            warnings.append(f"Only {len(selected)} positive-importance features; skipping model")
            if verbose:
                print(f"  INSUFFICIENT SIGNAL ({len(selected)} features)")
            # Return a degenerate result rather than training on noise
            if len(selected) == 0:
                return None, None, None, []
            # If we have at least 1-4, use them but warn heavily
            warnings.append("WARNING: Very few features — high overfitting risk")

        if verbose:
            n_valid_folds = len(fold_feature_sets)
            print(f"  Stable features: {len(selected)} selected "
                  f"(from {n_valid_folds} valid folds)")

        X_dev_sel = X_dev[selected]
        X_hold_sel = X_hold[selected]

        if not cv_oos_scores:
            warnings.append("No valid CV folds")
            cv_gap = 0
        else:
            cv_gap = np.mean(cv_is_scores) - np.mean(cv_oos_scores)
            if cv_gap > self.max_gap:
                warnings.append(f"CV gap {cv_gap:.4f} > threshold {self.max_gap}")

        if verbose and cv_oos_scores:
            print(f"  CV Spearman — IS: {np.mean(cv_is_scores):.4f}, "
                  f"OOS: {np.mean(cv_oos_scores):.4f}, gap: {cv_gap:.4f}")

        # ── HARD BLOCK: Reject severely overfit models ──
        # A CV gap above the rejection threshold means the model memorizes
        # training data and fails to generalize. Deploying such a model is
        # worse than no model at all.
        cv_gap_reject_threshold = self.max_gap * 3  # 0.15 when max_gap=0.05
        if cv_gap > cv_gap_reject_threshold:
            if verbose:
                print(f"  REJECTED: CV gap {cv_gap:.4f} exceeds hard block "
                      f"threshold {cv_gap_reject_threshold:.4f}. Model not saved.")
            return None, None, None, []

        # ── Train final model on full dev set ──
        # Fit scaler on full dev set for the final model
        scaler = StandardScaler()
        X_dev_scaled = scaler.fit_transform(X_dev_sel)
        X_hold_scaled = scaler.transform(X_hold_sel) if len(X_hold_sel) > 0 else np.empty((0, len(selected)))

        # Final GBR model WITHOUT internal validation_fraction
        # (number of iterations already validated via CV)
        gbr_model = self._make_model(use_early_stopping=False)
        gbr_model.fit(X_dev_scaled, y_dev.values, sample_weight=w_dev)

        # ── Optionally build diverse ensemble (GBR + ElasticNet + RF) ──
        if ENSEMBLE_DIVERSIFY and ElasticNet is not None and RandomForestRegressor is not None:
            final_model = self._train_diverse_ensemble(
                X_dev_sel, y_dev, w_dev, gbr_model, scaler, verbose,
            )
            # The DiverseEnsemble handles scaling internally per constituent.
            # Return a no-op (identity) scaler so the predictor's
            # scaler.transform(X) passes data through unchanged.
            scaler = IdentityScaler().fit(X_dev_sel)
        else:
            final_model = gbr_model

        # ── Holdout evaluation (Spearman correlation) ──
        hold_pred = final_model.predict(
            scaler.transform(X_hold_sel) if len(X_hold_sel) > 0 else np.empty((0, len(selected)))
        ) if len(X_hold_sel) > 0 else np.array([])
        hold_r2 = r2_score(y_hold.values, hold_pred) if len(y_hold) > 2 else 0
        hold_mae = mean_absolute_error(y_hold.values, hold_pred) if len(y_hold) > 2 else 0
        if len(y_hold) > 2 and len(hold_pred) > 0:
            hold_corr, _ = self._spearmanr(y_hold.values, hold_pred)
            hold_corr = float(hold_corr) if not np.isnan(hold_corr) else 0
        else:
            hold_corr = 0
            warnings.append("Holdout set too small for robust evaluation")

        if cv_oos_scores:
            degradation = np.mean(cv_oos_scores) - hold_corr
            if degradation > self.max_gap:
                warnings.append(f"Holdout degradation {degradation:.4f}")

        if verbose:
            print(f"  Holdout — R²: {hold_r2:.4f}, MAE: {hold_mae:.6f}, "
                  f"Spearman: {hold_corr:.4f}")

        # ── HARD BLOCK: Reject models with negative holdout R² ──
        # Negative R² means predictions are worse than predicting the mean.
        # Such a model actively destroys value when deployed.
        if len(y_hold) > 10 and hold_r2 < 0:
            if verbose:
                print(f"  REJECTED: Holdout R² = {hold_r2:.4f} < 0. "
                      f"Predictions are worse than the mean. Model not saved.")
            return None, None, None, []

        # Feature importance from final model
        feat_imp = dict(zip(selected, final_model.feature_importances_))

        result = TrainResult(
            model_name=model_name,
            n_samples=len(X_dev),
            n_features=len(selected),
            selected_features=selected,
            cv_scores=cv_oos_scores,
            cv_train_scores=cv_is_scores,
            cv_gap=cv_gap,
            holdout_r2=hold_r2,
            holdout_mae=hold_mae,
            holdout_correlation=hold_corr,
            feature_importance=feat_imp,
            target_std=target_std,
            feature_medians=feature_medians,
            warnings=warnings,
        )

        return result, final_model, scaler, selected

    def _train_diverse_ensemble(
        self,
        X_dev_sel: pd.DataFrame,
        y_dev: pd.Series,
        w_dev: Optional[np.ndarray],
        gbr_model: object,
        gbr_scaler: object,
        verbose: bool = True,
    ) -> DiverseEnsemble:
        """
        Train ElasticNet, RandomForest, and optionally XGBoost and LightGBM
        alongside the already-fitted GBR and return a ``DiverseEnsemble``
        with optimized stacking weights.

        Weight optimization:
            After fitting all constituent models, OOS predictions are collected
            from temporal CV folds. Constrained optimization (weights sum to 1,
            all non-negative) minimises MSE on the stacked OOS predictions.
            Falls back to equal weights if optimisation fails or scipy is
            unavailable.

        Each constituent model stores its own scaler so the ensemble's
        ``predict`` method is self-contained: callers pass *unscaled* features.

        Args:
            X_dev_sel: Development features (selected columns, unscaled).
            y_dev: Development targets.
            w_dev: Optional sample weights.
            gbr_model: Already-trained GradientBoostingRegressor (uses gbr_scaler).
            gbr_scaler: StandardScaler fitted on X_dev_sel for the GBR.
            verbose: Print progress information.
        """
        estimators: List[Tuple[object, Optional[object]]] = []

        # 1) GBR — already trained; pair it with its scaler
        estimators.append((gbr_model, gbr_scaler))

        # 2) ElasticNet — needs its own StandardScaler (linear model is
        #    sensitive to feature scale)
        enet_scaler = StandardScaler()
        X_enet = enet_scaler.fit_transform(X_dev_sel)
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
        enet.fit(X_enet, y_dev.values, sample_weight=w_dev)
        estimators.append((enet, enet_scaler))

        if verbose:
            print("  Diverse ensemble: ElasticNet trained")

        # 3) RandomForestRegressor — tree model, no scaler needed but we
        #    reuse a scaler for consistency (RF is scale-invariant so the
        #    scaler is harmless and keeps the code path uniform).
        rf_scaler = StandardScaler()
        X_rf = rf_scaler.fit_transform(X_dev_sel)
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=30,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_rf, y_dev.values, sample_weight=w_dev)
        estimators.append((rf, rf_scaler))

        if verbose:
            print("  Diverse ensemble: RandomForest trained")

        # 4) XGBoost — handles NaN natively, no special imputation needed.
        #    Uses its own scaler for code-path consistency.
        if XGBRegressor is not None:
            xgb_scaler = StandardScaler()
            X_xgb = xgb_scaler.fit_transform(X_dev_sel)
            xgb = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
            )
            xgb.fit(X_xgb, y_dev.values, sample_weight=w_dev)
            estimators.append((xgb, xgb_scaler))
            if verbose:
                print("  Diverse ensemble: XGBoost trained")

        # 5) LightGBM — fast gradient boosting with leaf-wise growth.
        if LGBMRegressor is not None:
            lgbm_scaler = StandardScaler()
            X_lgbm = lgbm_scaler.fit_transform(X_dev_sel)
            lgbm = LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
            lgbm.fit(X_lgbm, y_dev.values, sample_weight=w_dev)
            estimators.append((lgbm, lgbm_scaler))
            if verbose:
                print("  Diverse ensemble: LightGBM trained")

        n_models = len(estimators)
        model_names = ["GBR", "ElasticNet", "RF"]
        if XGBRegressor is not None:
            model_names.append("XGBoost")
        if LGBMRegressor is not None:
            model_names.append("LightGBM")

        # ── Optimize stacking weights via OOS predictions from CV folds ──
        optimized_weights = self._optimize_ensemble_weights(
            estimators=estimators,
            X_dev_sel=X_dev_sel,
            y_dev=y_dev,
            w_dev=w_dev,
            model_names=model_names,
            verbose=verbose,
        )

        ensemble = DiverseEnsemble(estimators, weights=optimized_weights)

        if verbose:
            weight_strs = [f"{name}={w:.3f}" for name, w in zip(model_names, ensemble.weights)]
            print(f"  Diverse ensemble: {n_models} models "
                  f"({' + '.join(model_names)}), weights: [{', '.join(weight_strs)}]")

        return ensemble

    def _optimize_ensemble_weights(
        self,
        estimators: List[Tuple[object, Optional[object]]],
        X_dev_sel: pd.DataFrame,
        y_dev: pd.Series,
        w_dev: Optional[np.ndarray],
        model_names: List[str],
        verbose: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Find optimal stacking weights by minimising MSE on OOS predictions
        collected from temporal CV folds.

        Constraints:
            - All weights >= 0
            - Weights sum to 1

        Returns optimized weight array, or None to fall back to equal weights.
        """
        n_models = len(estimators)
        if n_models <= 1:
            return None

        if sp_minimize is None:
            if verbose:
                print("  Stacking weights: scipy unavailable, using equal weights")
            return None

        # Collect OOS predictions from temporal CV folds
        dates = self._extract_dates(X_dev_sel.index)
        X_sorted, y_sorted, w_sorted, dates_sorted = self._sort_panel_by_time(
            X_dev_sel, y_dev, w_dev,
        )

        n_unique_dates = len(pd.Index(dates_sorted.unique()))
        n_folds = min(self.cv_folds, max(2, n_unique_dates // 120))
        purge_gap = 10  # conservative purge for weight optimization
        embargo = max(1, purge_gap // 2)

        folds = self._date_purged_folds(
            dates=dates_sorted,
            n_folds=n_folds,
            purge_gap=purge_gap,
            embargo=embargo,
        )

        if len(folds) < 2:
            if verbose:
                print("  Stacking weights: insufficient CV folds, using equal weights")
            return None

        # Collect OOS predictions from each model across all folds
        all_oos_preds = [[] for _ in range(n_models)]  # per-model list of arrays
        all_oos_targets = []

        for train_idx, test_idx in folds:
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            X_fold_train = X_sorted.iloc[train_idx]
            y_fold_train = y_sorted.iloc[train_idx]
            X_fold_test = X_sorted.iloc[test_idx]
            y_fold_test = y_sorted.iloc[test_idx]
            w_fold = w_sorted[train_idx] if w_sorted is not None else None

            fold_preds = []
            fold_ok = True
            for i, (model_template, _) in enumerate(estimators):
                try:
                    # Fit a fresh copy of each model type on fold training data
                    fold_scaler = StandardScaler()
                    X_tr_scaled = fold_scaler.fit_transform(X_fold_train.values)
                    X_te_scaled = fold_scaler.transform(X_fold_test.values)

                    # Clone the model by creating a fresh instance of same type
                    fold_model = self._clone_model(model_template)
                    if fold_model is None:
                        fold_ok = False
                        break

                    if hasattr(fold_model, 'fit'):
                        try:
                            fold_model.fit(X_tr_scaled, y_fold_train.values, sample_weight=w_fold)
                        except TypeError:
                            fold_model.fit(X_tr_scaled, y_fold_train.values)

                    pred = fold_model.predict(X_te_scaled)
                    fold_preds.append(pred)
                except (ValueError, RuntimeError, TypeError):
                    fold_ok = False
                    break

            if not fold_ok or len(fold_preds) != n_models:
                continue

            for i, pred in enumerate(fold_preds):
                all_oos_preds[i].append(pred)
            all_oos_targets.append(y_fold_test.values)

        # Concatenate across folds
        if not all_oos_targets or any(len(p) == 0 for p in all_oos_preds):
            if verbose:
                print("  Stacking weights: no valid OOS predictions, using equal weights")
            return None

        y_oos = np.concatenate(all_oos_targets)
        pred_matrix = np.column_stack([np.concatenate(p) for p in all_oos_preds])
        # pred_matrix shape: (n_oos_samples, n_models)

        if len(y_oos) < 20:
            if verbose:
                print("  Stacking weights: too few OOS samples, using equal weights")
            return None

        # Optimize: minimize MSE = ||y - pred_matrix @ w||^2
        def objective(w):
            blended = pred_matrix @ w
            return float(np.mean((y_oos - blended) ** 2))

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        # Bounds: all weights >= 0
        bounds = [(0.0, 1.0)] * n_models
        # Initial guess: equal weights
        w0 = np.full(n_models, 1.0 / n_models)

        try:
            result = sp_minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if result.success:
                opt_weights = result.x
                # Normalize to ensure exact sum-to-1 (numerical safety)
                opt_weights = np.clip(opt_weights, 0.0, 1.0)
                opt_weights = opt_weights / opt_weights.sum()

                if verbose:
                    equal_mse = objective(w0)
                    opt_mse = objective(opt_weights)
                    improvement = (1.0 - opt_mse / max(equal_mse, 1e-12)) * 100
                    print(f"  Stacking weights optimized: MSE improvement {improvement:.1f}% "
                          f"(equal={equal_mse:.6f}, optimized={opt_mse:.6f})")
                return opt_weights
            else:
                if verbose:
                    print(f"  Stacking weights: optimization did not converge "
                          f"({result.message}), using equal weights")
                return None
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"  Stacking weights: optimization failed ({e}), using equal weights")
            return None

    @staticmethod
    def _clone_model(model: object) -> Optional[object]:
        """
        Create a fresh (unfitted) clone of a model with the same hyperparameters.

        Supports sklearn-style estimators (get_params/set_params) and falls back
        to type-based reconstruction for known model types.
        """
        try:
            # sklearn convention: get_params() returns constructor kwargs
            if hasattr(model, "get_params"):
                params = model.get_params()
                return type(model)(**params)
        except (TypeError, ValueError, RuntimeError):
            pass

        # Fallback: construct a bare instance of the same type
        try:
            return type(model)()
        except (TypeError, RuntimeError):
            return None

    def _make_model(self, use_early_stopping: bool = True) -> GradientBoostingRegressor:
        """Create a fresh GradientBoostingRegressor with configured params."""
        self._require_sklearn()
        params = dict(
            n_estimators=self.model_params.get("n_estimators", 500),
            max_depth=self.model_params.get("max_depth", 4),
            min_samples_leaf=self.model_params.get("min_samples_leaf", 30),
            learning_rate=self.model_params.get("learning_rate", 0.05),
            subsample=self.model_params.get("subsample", 0.8),
            max_features=self.model_params.get("max_features", "sqrt"),
            random_state=42,
        )
        if use_early_stopping:
            params["validation_fraction"] = 0.15
            params["n_iter_no_change"] = 20
        return GradientBoostingRegressor(**params)

    def _save(
        self,
        global_model, global_scaler, global_features, global_result,
        regime_models, regime_scalers, regime_feature_sets, regime_results,
        horizon, ensemble_result,
        versioned: bool = True,
        survivorship_mode: bool = False,
        universe_as_of: Optional[str] = None,
        train_data_start: Optional[str] = None,
        train_data_end: Optional[str] = None,
        structural_weights_applied: bool = False,
    ):
        """Save all model artifacts to disk using joblib (safe serialization)."""
        # Determine save directory
        registry = ModelRegistry() if versioned else None
        if versioned and registry is not None:
            version_id, save_dir = registry.create_version_dir()
        else:
            version_id = None
            save_dir = MODEL_DIR
            save_dir.mkdir(parents=True, exist_ok=True)

        prefix = save_dir / f"ensemble_{horizon}d"

        # Global model
        joblib.dump(global_model, f"{prefix}_global_model.pkl")
        joblib.dump(global_scaler, f"{prefix}_global_scaler.pkl")

        # Regime models (skip None entries from regimes with insufficient samples)
        for code, model in regime_models.items():
            if model is None:
                continue
            joblib.dump(model, f"{prefix}_regime{code}_model.pkl")
            joblib.dump(regime_scalers[code], f"{prefix}_regime{code}_scaler.pkl")

        # Metadata
        meta = {
            "horizon": horizon,
            "train_data_start": train_data_start,
            "train_data_end": train_data_end,
            "global_features": global_features,
            "global_cv_corr": float(np.mean(global_result.cv_scores)) if global_result.cv_scores else 0,
            "global_holdout_r2": float(global_result.holdout_r2),
            "global_holdout_corr": float(global_result.holdout_correlation),
            "global_target_std": float(global_result.target_std),
            "global_feature_medians": {k: float(v) for k, v in global_result.feature_medians.items()
                                        if k in global_features},
            "global_warnings": global_result.warnings,
            "global_feature_importance": {k: float(v) for k, v in global_result.feature_importance.items()},
            "regime_models": {},
            "train_time_seconds": ensemble_result.train_time_seconds,
            "total_samples": ensemble_result.total_samples,
            "structural_weights_applied": structural_weights_applied,
        }
        if structural_weights_applied:
            meta["structural_weight_config"] = {
                "changepoint_penalty": STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY,
                "jump_penalty": STRUCTURAL_WEIGHT_JUMP_PENALTY,
                "stress_penalty": STRUCTURAL_WEIGHT_STRESS_PENALTY,
            }
        for code, result in regime_results.items():
            meta["regime_models"][str(code)] = {
                "name": REGIME_NAMES[code],
                "features": regime_feature_sets[code],
                "cv_corr": float(np.mean(result.cv_scores)) if result.cv_scores else 0,
                "holdout_r2": float(result.holdout_r2),
                "holdout_corr": float(result.holdout_correlation),
                "target_std": float(result.target_std),
                "feature_medians": {k: float(v) for k, v in result.feature_medians.items()
                                     if k in regime_feature_sets[code]},
                "n_samples": result.n_samples,
                "warnings": result.warnings,
                "feature_importance": {k: float(v) for k, v in result.feature_importance.items()},
            }

        if version_id:
            meta["version_id"] = version_id
        if survivorship_mode:
            meta["survivorship_mode"] = True
            meta["universe_as_of"] = universe_as_of

        with open(f"{prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Register version
        if versioned and registry is not None and version_id:
            version = ModelVersion(
                version_id=version_id,
                training_date=time.strftime("%Y-%m-%dT%H:%M:%S"),
                horizon=horizon,
                universe_size=0,  # set by caller
                n_samples=ensemble_result.total_samples,
                n_features=ensemble_result.total_features,
                oos_spearman=float(np.mean(global_result.cv_scores)) if global_result.cv_scores else 0,
                cv_gap=float(global_result.cv_gap),
                holdout_r2=float(global_result.holdout_r2),
                holdout_spearman=float(global_result.holdout_correlation),
                survivorship_mode=survivorship_mode,
                universe_as_of=universe_as_of,
            )
            registry.register_version(version)
            registry.prune_old()
            print(f"\n  Saved versioned model: {save_dir}")
            print(f"  Version ID: {version_id}")
        else:
            # Also save to flat MODEL_DIR for backward compat
            if save_dir != MODEL_DIR:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\n  Saved to {prefix}_*")

    def _fit_calibrator(
        self,
        global_model, global_scaler, global_features, global_result,
        X: pd.DataFrame, y: pd.Series, horizon: int, versioned: bool,
    ) -> Optional[Dict]:
        """Fit isotonic regression calibrator on holdout predictions vs outcomes.

        The holdout is split 50/50 into calibration-fit and calibration-validate
        portions to prevent calibration overfitting. ECE (Expected Calibration
        Error) is computed on the validation portion and returned.

        Saves calibrator.pkl alongside the model so the predictor can load and
        apply it to map raw confidence to empirical probability of correctness.

        Returns:
            Dict with calibration metrics (ece, n_cal_fit, n_cal_val) or None.
        """
        from .calibration import compute_ece

        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            return None  # sklearn not available

        try:
            # Use the holdout portion to calibrate
            holdout_frac = HOLDOUT_FRACTION
            n = len(X)
            n_holdout = max(40, int(n * holdout_frac))
            holdout_X = X.iloc[-n_holdout:]
            holdout_y = y.iloc[-n_holdout:]

            # Prepare features matching global model
            available = [c for c in global_features if c in holdout_X.columns]
            if not available:
                return None
            X_cal = holdout_X[available].copy()
            for col in global_features:
                if col not in X_cal.columns:
                    X_cal[col] = global_result.feature_medians.get(col, 0.0)
            X_cal = X_cal[global_features].fillna(X_cal.median())
            X_cal_scaled = global_scaler.transform(X_cal)

            # Generate raw predictions on holdout
            raw_preds = global_model.predict(X_cal_scaled)
            # Binary outcome: did prediction get direction right?
            direction_correct = ((raw_preds > 0) & (holdout_y.values > 0)) | (
                (raw_preds <= 0) & (holdout_y.values <= 0)
            )
            direction_correct = direction_correct.astype(float)

            # Raw confidence proxy: absolute prediction magnitude, normalized
            abs_preds = np.abs(raw_preds)
            if abs_preds.max() > 1e-10:
                raw_confidence = abs_preds / abs_preds.max()
            else:
                return None

            # ── Split holdout 50/50: calibration-fit vs calibration-validate ──
            # Use temporal ordering (first half for fitting, second half for validation)
            # to prevent look-ahead in the calibration itself.
            cal_split = len(raw_confidence) // 2
            if cal_split < 10:
                # Too few samples to split; fit on all but report no ECE
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(raw_confidence, direction_correct)
                cal_metrics = {"ece": None, "n_cal_fit": len(raw_confidence), "n_cal_val": 0}
            else:
                # Fit on first half
                cal_fit_conf = raw_confidence[:cal_split]
                cal_fit_outcome = direction_correct[:cal_split]

                # Validate on second half
                cal_val_conf = raw_confidence[cal_split:]
                cal_val_outcome = direction_correct[cal_split:]

                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cal_fit_conf, cal_fit_outcome)

                # Compute ECE on validation half
                cal_val_preds = calibrator.transform(cal_val_conf)
                ece = compute_ece(cal_val_preds, cal_val_outcome)

                cal_metrics = {
                    "ece": float(ece),
                    "n_cal_fit": int(cal_split),
                    "n_cal_val": int(len(raw_confidence) - cal_split),
                }

            # Save calibrator
            registry = ModelRegistry() if versioned else None
            if versioned and registry is not None and registry.has_versions():
                save_dir = registry.get_latest_dir()
            else:
                save_dir = MODEL_DIR
            if save_dir is not None:
                calibrator_path = save_dir / f"ensemble_{horizon}d_calibrator.pkl"
                joblib.dump(calibrator, str(calibrator_path))

            return cal_metrics
        except (ValueError, RuntimeError, OSError):
            return None  # Calibration is optional; don't fail training

    def _print_summary(self, result: EnsembleResult, regimes: Optional[pd.Series] = None,
                        targets: Optional[pd.Series] = None):
        """Print comprehensive training summary with gate diagnostics."""
        g = result.global_model
        cv_gap_threshold = self.max_gap * 3

        print(f"\n{'─'*60}")
        print(f"  TRAINING DIAGNOSTICS REPORT")
        print(f"{'─'*60}")

        # ── Gate Status ──
        print(f"\n  Quality Gate Results:")
        if g is not None:
            gap_status = "PASS" if g.cv_gap <= cv_gap_threshold else "FAIL"
            r2_status = "PASS" if g.holdout_r2 >= 0 else "FAIL"
            print(f"    CV gap hard block:   {gap_status}  (gap={g.cv_gap:.4f}, "
                  f"threshold={cv_gap_threshold:.4f})")
            print(f"    Holdout R² >= 0:     {r2_status}  (R²={g.holdout_r2:.4f})")
        else:
            print(f"    Global model: REJECTED (did not survive gates)")

        # ── Global Model ──
        if g is not None:
            print(f"\n  Global Model:")
            print(f"    R²={g.holdout_r2:.4f}, Spearman={g.holdout_correlation:.4f}, "
                  f"MAE={g.holdout_mae:.6f}")
            print(f"    Features: {g.n_features}, CV gap: {g.cv_gap:.4f}, "
                  f"CV OOS mean: {np.mean(g.cv_scores):.4f}")
            if g.warnings:
                for w in g.warnings:
                    print(f"    WARNING: {w}")

        # ── Regime Models ──
        if result.regime_models:
            print(f"\n  Per-Regime Models:")
            for code in sorted(result.regime_models.keys()):
                r = result.regime_models[code]
                name = REGIME_NAMES.get(code, f"regime_{code}")
                gate = "PASS" if r.holdout_r2 >= 0 and r.cv_gap <= cv_gap_threshold else "WARN"
                print(f"    {name}: R²={r.holdout_r2:.4f}, Corr={r.holdout_correlation:.4f}, "
                      f"N={r.n_samples}, Gap={r.cv_gap:.4f} [{gate}]")
                if r.warnings:
                    for w in r.warnings:
                        print(f"      WARNING: {w}")

        # ── Regime Distribution ──
        if regimes is not None and len(regimes) > 0:
            print(f"\n  Regime Distribution:")
            total = len(regimes)
            for code in sorted(regimes.unique()):
                count = int((regimes == code).sum())
                name = REGIME_NAMES.get(int(code), f"regime_{code}")
                pct = 100.0 * count / total
                print(f"    {name}: {count} samples ({pct:.1f}%)")

        # ── Target Statistics (excess return diagnostics) ──
        if targets is not None and len(targets) > 0:
            clean_t = targets.dropna()
            print(f"\n  Target Statistics (excess returns):")
            print(f"    Mean:   {clean_t.mean():.6f}")
            print(f"    Std:    {clean_t.std():.6f}")
            print(f"    Median: {clean_t.median():.6f}")
            print(f"    Skew:   {float(clean_t.skew()):.4f}")
            print(f"    >0:     {(clean_t > 0).mean():.1%}")
            print(f"    NaN:    {targets.isna().sum()} / {len(targets)}")

        print(f"\n  Training time: {result.train_time_seconds:.1f}s")
        print(f"{'─'*60}")

    @staticmethod
    def compute_shared_features(
        features: pd.DataFrame,
        targets_dict: Dict[int, pd.Series],
        max_features: int = MAX_FEATURES_SELECTED,
    ) -> List[str]:
        """Identify features with predictive power across multiple horizons.

        Trains a lightweight model per horizon and selects features that are
        important for ANY horizon.  This finds features with persistent
        predictive power across time scales, enabling multi-horizon
        information sharing.

        Args:
            features: Full feature DataFrame.
            targets_dict: {horizon_days: target_series} for each horizon.
            max_features: Maximum number of shared features to select.

        Returns:
            List of shared feature names ranked by cross-horizon importance.
        """
        if GradientBoostingRegressor is None:
            return features.columns[:max_features].tolist()

        importance_sum = pd.Series(0.0, index=features.columns)
        n_horizons = 0

        for horizon, targets in targets_dict.items():
            valid = targets.notna() & features.notna().any(axis=1)
            X_h = features[valid]
            y_h = targets[valid]
            if len(X_h) < 100:
                continue

            # Quick GBR for feature importance
            quick_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=30,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            X_filled = X_h.fillna(X_h.median())
            quick_model.fit(X_filled.values, y_h.values)
            imp = pd.Series(
                quick_model.feature_importances_, index=features.columns
            )
            importance_sum += imp
            n_horizons += 1

        if n_horizons == 0:
            return features.columns[:max_features].tolist()

        # Average importance across horizons
        avg_importance = importance_sum / n_horizons
        shared = avg_importance.nlargest(max_features).index.tolist()
        return shared
