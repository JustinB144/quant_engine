"""
Meta-labeling model for signal confidence prediction (Spec 04).

Trains a secondary XGBoost classifier to predict whether the primary
trading signal will be correct.  Uses signal magnitude, volatility,
regime state, and market context features as inputs.

The output — P(signal_correct) — is used by the autopilot engine to
filter low-confidence signals before backtesting/promotion.

References:
    de Prado, M.L. (2018). *Advances in Financial Machine Learning*,
    Chapter 3: Meta-Labeling.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    META_LABELING_XGB_PARAMS,
    META_LABELING_CONFIDENCE_THRESHOLD,
    META_LABELING_MIN_SAMPLES,
    META_LABELING_FOLD_COUNT,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    import joblib

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    _HAS_JOBLIB = False


# Canonical ordering of meta-features expected by the trained model.
META_FEATURE_COLUMNS: List[str] = [
    "signal_magnitude",
    "signal_volatility",
    "signal_autocorr",
    "return_volatility",
    "price_momentum",
    "max_dd_realized",
    "regime_0",
    "regime_1",
    "regime_2",
    "regime_3",
]


class MetaLabelingModel:
    """Secondary classifier predicting primary signal correctness.

    The meta-labeling approach trains a classifier on features derived
    from the primary signal and market context.  The output is
    P(signal_correct), which can be used to filter low-confidence
    signals before they enter the backtest or promotion pipeline.

    Lifecycle
    ---------
    1. ``build_meta_features`` — compute meta-features from signals + market
    2. ``build_labels`` — label historical signals as correct/incorrect
    3. ``train`` — fit XGBoost on (meta_features, labels)
    4. ``predict_confidence`` — inference on new meta-features
    5. ``save`` / ``load`` — persist model between cycles
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        confidence_threshold: float = META_LABELING_CONFIDENCE_THRESHOLD,
        min_samples: int = META_LABELING_MIN_SAMPLES,
        n_folds: int = META_LABELING_FOLD_COUNT,
    ):
        self.params = dict(params or META_LABELING_XGB_PARAMS)
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples
        self.n_folds = n_folds
        self.model: object = None
        self.last_train_time: Optional[datetime] = None
        self.feature_importance_: Optional[Dict[str, float]] = None
        self._model_dir = MODEL_DIR / "meta_labeler"

    # ── Feature engineering ──────────────────────────────────────────

    @staticmethod
    def build_meta_features(
        signals: pd.Series,
        returns: pd.Series,
        regime_states: pd.Series,
    ) -> pd.DataFrame:
        """Build meta-features from signal, return, and regime data.

        All rolling windows use ``min_periods`` guards so that the first
        few rows produce 0.0 rather than NaN.

        Args:
            signals: Primary signal values (predicted returns or z-scores).
            returns: Asset returns (daily or per-bar).
            regime_states: Regime labels (integers 0–3).

        Returns:
            DataFrame with columns matching ``META_FEATURE_COLUMNS``,
            aligned to *signals*' index.
        """
        meta = pd.DataFrame(index=signals.index)

        # -- Signal-based --
        meta["signal_magnitude"] = signals.abs()
        meta["signal_volatility"] = (
            signals.rolling(20, min_periods=5).std().fillna(0.0)
        )

        def _lag1_autocorr(x: np.ndarray) -> float:
            s = pd.Series(x)
            if len(s) < 10 or s.std() < 1e-10:
                return 0.0
            ac = s.autocorr(lag=1)
            return float(ac) if np.isfinite(ac) else 0.0

        meta["signal_autocorr"] = (
            signals.rolling(20, min_periods=10)
            .apply(_lag1_autocorr, raw=True)
            .fillna(0.0)
        )

        # -- Market-based --
        returns_clean = returns.reindex(signals.index).fillna(0.0)
        meta["return_volatility"] = (
            returns_clean.rolling(20, min_periods=5).std().fillna(0.0)
        )
        meta["price_momentum"] = (
            returns_clean.rolling(10, min_periods=3).sum().fillna(0.0)
        )

        # Max drawdown over trailing 60 bars
        cum_ret = (1 + returns_clean).cumprod()
        rolling_max = cum_ret.rolling(60, min_periods=10).max()
        dd = (cum_ret - rolling_max) / rolling_max.replace(0, np.nan)
        meta["max_dd_realized"] = dd.fillna(0.0)

        # -- Regime one-hot --
        regime_aligned = (
            regime_states.reindex(signals.index).fillna(0).astype(int)
        )
        for code in range(4):
            meta[f"regime_{code}"] = (regime_aligned == code).astype(float)

        return meta[META_FEATURE_COLUMNS].fillna(0.0)

    # ── Label construction ───────────────────────────────────────────

    @staticmethod
    def build_labels(
        signals: pd.Series,
        actuals: pd.Series,
        entry_threshold: float = 0.005,
    ) -> pd.Series:
        """Build binary labels indicating signal correctness.

        A signal is labelled **correct** (1) when:
        - ``|signal| > entry_threshold``  (actionable)
        - ``sign(signal) == sign(actual)``  (direction correct)

        Non-actionable signals (below threshold) are labelled 0.

        Args:
            signals: Primary signal values.
            actuals: Realised forward returns.
            entry_threshold: Minimum absolute signal for an actionable trade.

        Returns:
            Binary ``pd.Series`` (1 = correct, 0 = incorrect) on the
            inner-join index of *signals* and *actuals*.
        """
        aligned_signals, aligned_actuals = signals.align(actuals, join="inner")
        actionable = aligned_signals.abs() > entry_threshold
        correct_dir = np.sign(aligned_signals) == np.sign(aligned_actuals)
        return (actionable & correct_dir).astype(int)

    # ── Training ─────────────────────────────────────────────────────

    def train(
        self,
        meta_features: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, float]:
        """Train XGBoost classifier on meta-features and binary labels.

        Uses a 90/10 temporal train/validation split with early stopping.
        Class imbalance is handled via ``scale_pos_weight``.

        Args:
            meta_features: DataFrame of meta-features.
            labels: Binary labels (1 = signal correct, 0 = incorrect).

        Returns:
            Dict of training metrics (accuracy, sample counts, etc.).

        Raises:
            RuntimeError: If XGBoost is not installed or insufficient data.
        """
        if not _HAS_XGB:
            raise RuntimeError(
                "xgboost is required for meta-labeling. "
                "Install via: pip install 'quant_engine[ml]'"
            )

        # Align features and labels
        common_idx = meta_features.index.intersection(labels.index)
        X = meta_features.loc[common_idx].copy()
        y = labels.loc[common_idx].copy()

        # Drop rows with NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < self.min_samples:
            raise RuntimeError(
                f"Insufficient samples for meta-labeling training: "
                f"{len(X)} < {self.min_samples}"
            )

        # Select only expected columns
        feature_cols = [c for c in META_FEATURE_COLUMNS if c in X.columns]
        X = X[feature_cols]

        # Handle class imbalance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_neg > 0 and n_pos > 0:
            scale_pos_weight = float(n_neg) / float(n_pos)
        else:
            scale_pos_weight = 1.0
        scale_pos_weight = min(max(scale_pos_weight, 0.1), 10.0)

        xgb_params = dict(self.params)
        xgb_params["scale_pos_weight"] = scale_pos_weight
        xgb_params["eval_metric"] = "logloss"
        xgb_params["random_state"] = 42
        xgb_params["verbosity"] = 0

        # Temporal train/validation split (90/10)
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # XGBoost 2.0+ API: early_stopping_rounds is a constructor param,
        # not a fit() param. eval_set is still passed to fit().
        use_early_stopping = len(X_val) >= 10
        if use_early_stopping:
            xgb_params["early_stopping_rounds"] = 10

        model = XGBClassifier(**xgb_params)

        fit_params: Dict = {}
        if use_early_stopping:
            fit_params["eval_set"] = [(X_val.values, y_val.values)]

        model.fit(X_train.values, y_train.values, **fit_params)

        self.model = model
        self.last_train_time = datetime.now()

        # Feature importance
        importance = model.feature_importances_
        self.feature_importance_ = dict(zip(feature_cols, importance.tolist()))

        # Training metrics
        train_preds = model.predict(X_train.values)
        train_acc = float((train_preds == y_train.values).mean())

        metrics: Dict[str, float] = {
            "train_accuracy": train_acc,
            "n_samples": float(len(X)),
            "n_positive": float(n_pos),
            "n_negative": float(n_neg),
            "scale_pos_weight": scale_pos_weight,
            "n_features": float(len(feature_cols)),
        }

        if len(X_val) >= 10:
            val_preds = model.predict(X_val.values)
            metrics["val_accuracy"] = float(
                (val_preds == y_val.values).mean()
            )
            try:
                val_proba = model.predict_proba(X_val.values)[:, 1]
                from sklearn.metrics import roc_auc_score

                metrics["val_auc"] = float(
                    roc_auc_score(y_val.values, val_proba)
                )
            except (ImportError, ValueError):
                pass

        # Warn on dominant single feature
        max_imp = float(max(importance)) if len(importance) > 0 else 0.0
        if max_imp > 0.50:
            logger.warning(
                "Meta-labeling: single feature dominates importance "
                "(%.2f). Consider increasing regularization.",
                max_imp,
            )

        logger.info(
            "Meta-labeling model trained: %d samples, "
            "train_acc=%.3f, val_acc=%.3f, features=%s",
            len(X),
            train_acc,
            metrics.get("val_accuracy", float("nan")),
            feature_cols,
        )
        return metrics

    # ── Inference ────────────────────────────────────────────────────

    def predict_confidence(
        self,
        meta_features: pd.DataFrame,
    ) -> pd.Series:
        """Predict P(signal_correct) for each sample.

        Args:
            meta_features: DataFrame of meta-features (must include
                columns from ``META_FEATURE_COLUMNS``).

        Returns:
            ``pd.Series`` of confidence scores in [0, 1].

        Raises:
            RuntimeError: If model is not trained/loaded.
        """
        if self.model is None:
            raise RuntimeError("Meta-labeling model not trained or loaded.")

        feature_cols = [
            c for c in META_FEATURE_COLUMNS if c in meta_features.columns
        ]
        X = meta_features[feature_cols].fillna(0.0)

        proba = self.model.predict_proba(X.values)
        # Column 1 = P(class=1) = P(signal_correct)
        if proba.shape[1] >= 2:
            confidence = proba[:, 1]
        else:
            confidence = proba[:, 0]

        return pd.Series(
            confidence, index=meta_features.index, name="meta_confidence"
        )

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save trained model to disk via joblib.

        Writes both a versioned file and a ``meta_labeler_current.joblib``
        pointer so that the next cycle can load the latest model.

        Args:
            filepath: Explicit target path.  If ``None``, a timestamped
                path under ``MODEL_DIR/meta_labeler/`` is used.

        Returns:
            Path where the model was written.
        """
        if not _HAS_JOBLIB:
            raise RuntimeError("joblib is required for model persistence.")
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        if filepath is None:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self._model_dir / f"meta_labeler_v{ts}.joblib"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "params": self.params,
            "confidence_threshold": self.confidence_threshold,
            "last_train_time": self.last_train_time,
            "feature_importance": self.feature_importance_,
        }
        joblib.dump(state, filepath)

        # Update "current" pointer
        self._model_dir.mkdir(parents=True, exist_ok=True)
        current_path = self._model_dir / "meta_labeler_current.joblib"
        joblib.dump(state, current_path)

        logger.info("Meta-labeling model saved to %s", filepath)
        return filepath

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Load a previously trained model from disk.

        Args:
            filepath: Source path.  If ``None``, loads from the default
                ``meta_labeler_current.joblib`` under the model directory.

        Returns:
            ``True`` if the model loaded successfully, ``False`` otherwise.
        """
        if not _HAS_JOBLIB:
            logger.warning(
                "joblib not available; cannot load meta-labeling model."
            )
            return False

        if filepath is None:
            filepath = self._model_dir / "meta_labeler_current.joblib"

        filepath = Path(filepath)
        if not filepath.exists():
            logger.info("No meta-labeling model found at %s", filepath)
            return False

        try:
            state = joblib.load(filepath)
            self.model = state["model"]
            self.params = state.get("params", self.params)
            self.confidence_threshold = state.get(
                "confidence_threshold", self.confidence_threshold
            )
            self.last_train_time = state.get("last_train_time")
            self.feature_importance_ = state.get("feature_importance")
            logger.info("Meta-labeling model loaded from %s", filepath)
            return True
        except Exception as exc:
            logger.error("Failed to load meta-labeling model: %s", exc)
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained or loaded."""
        return self.model is not None
