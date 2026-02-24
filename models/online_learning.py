"""
Online learning module for incremental model updates between full retrains.

Provides lightweight parameter adjustment using recent trade outcomes
without the cost of a full retrain cycle. Uses exponentially-weighted
moving statistics to adapt model confidence scaling and feature weights.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OnlineUpdate:
    """Record of a single online update step."""

    timestamp: str
    n_new_samples: int
    confidence_adjustment: float
    feature_drift_detected: bool
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)


class OnlineLearner:
    """Incremental model updater between full retrains.

    Rather than retraining the full model ensemble, this module:
    1. Tracks prediction errors on new data
    2. Adjusts confidence scaling (multiplicative bias correction)
    3. Detects feature drift via running statistics
    4. Triggers full retrain when drift exceeds threshold

    Parameters
    ----------
    decay : float
        Exponential decay factor for running statistics (0.95 = 20-sample half-life).
    drift_threshold : float
        Z-score threshold for declaring feature drift.
    min_samples : int
        Minimum new samples before performing an update.
    """

    def __init__(
        self,
        decay: float = 0.95,
        drift_threshold: float = 2.5,
        min_samples: int = 20,
        state_path: Optional[Path] = None,
    ):
        self.decay = decay
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.state_path = state_path or Path("data/online_state.json")

        # Running statistics
        self._running_pred_mean = 0.0
        self._running_pred_var = 1.0
        self._running_error_mean = 0.0
        self._running_error_var = 1.0
        self._confidence_scale = 1.0
        self._n_updates = 0
        self._pending_samples: List[Dict[str, float]] = []
        self._update_history: List[OnlineUpdate] = []

        # Feature-level drift tracking
        self._feature_means: Dict[str, float] = {}
        self._feature_vars: Dict[str, float] = {}

    def add_sample(
        self,
        prediction: float,
        actual: float,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a new prediction-outcome pair for online tracking.

        Parameters
        ----------
        prediction : float
            Model's predicted return.
        actual : float
            Realized return.
        features : dict, optional
            Feature values for drift detection.
        """
        self._pending_samples.append({
            "prediction": prediction,
            "actual": actual,
            "error": prediction - actual,
            "features": features or {},
        })

    def update(self) -> Optional[OnlineUpdate]:
        """Perform an online update if enough samples have accumulated.

        Returns
        -------
        OnlineUpdate or None if insufficient samples.
        """
        if len(self._pending_samples) < self.min_samples:
            return None

        samples = self._pending_samples
        self._pending_samples = []

        errors = np.array([s["error"] for s in samples])
        predictions = np.array([s["prediction"] for s in samples])
        actuals = np.array([s["actual"] for s in samples])

        # Metrics before
        metrics_before = {
            "confidence_scale": self._confidence_scale,
            "running_error_mean": self._running_error_mean,
        }

        # Update running error statistics with exponential decay
        for err in errors:
            self._running_error_mean = (
                self.decay * self._running_error_mean + (1 - self.decay) * err
            )
            self._running_error_var = (
                self.decay * self._running_error_var
                + (1 - self.decay) * (err - self._running_error_mean) ** 2
            )

        # Adjust confidence scaling
        # If predictions are systematically too high/low, scale them
        if abs(self._running_error_mean) > 0.001:
            # Multiplicative bias correction
            pred_mean = predictions.mean()
            if abs(pred_mean) > 1e-10:
                bias_ratio = actuals.mean() / pred_mean
                # Smooth adjustment (don't overcorrect)
                self._confidence_scale *= 0.8 + 0.2 * np.clip(bias_ratio, 0.5, 2.0)
                self._confidence_scale = np.clip(self._confidence_scale, 0.1, 5.0)

        # Feature drift detection
        drift_detected = False
        for sample in samples:
            for feat_name, feat_val in sample.get("features", {}).items():
                if feat_name not in self._feature_means:
                    self._feature_means[feat_name] = feat_val
                    self._feature_vars[feat_name] = 1.0
                    continue

                self._feature_means[feat_name] = (
                    self.decay * self._feature_means[feat_name]
                    + (1 - self.decay) * feat_val
                )
                self._feature_vars[feat_name] = (
                    self.decay * self._feature_vars[feat_name]
                    + (1 - self.decay)
                    * (feat_val - self._feature_means[feat_name]) ** 2
                )

                # Z-score test for drift
                std = max(np.sqrt(self._feature_vars[feat_name]), 1e-10)
                z = abs(feat_val - self._feature_means[feat_name]) / std
                if z > self.drift_threshold:
                    drift_detected = True

        self._n_updates += 1

        metrics_after = {
            "confidence_scale": self._confidence_scale,
            "running_error_mean": self._running_error_mean,
        }

        update_record = OnlineUpdate(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_new_samples=len(samples),
            confidence_adjustment=self._confidence_scale,
            feature_drift_detected=drift_detected,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )
        self._update_history.append(update_record)
        self._save_state()

        logger.info(
            "Online update #%d: %d samples, confidence_scale=%.3f, drift=%s",
            self._n_updates,
            len(samples),
            self._confidence_scale,
            drift_detected,
        )

        return update_record

    def adjust_prediction(self, raw_prediction: float) -> float:
        """Apply online-learned confidence scaling to a raw prediction."""
        return raw_prediction * self._confidence_scale

    def should_retrain(self) -> bool:
        """Check if conditions warrant a full model retrain.

        Returns True if:
        - Feature drift has been detected in recent updates
        - Confidence scale has drifted far from 1.0
        - Running error mean is persistently large
        """
        if not self._update_history:
            return False

        recent = self._update_history[-3:]
        drift_count = sum(1 for u in recent if u.feature_drift_detected)
        if drift_count >= 2:
            return True

        if abs(self._confidence_scale - 1.0) > 0.5:
            return True

        if abs(self._running_error_mean) > 0.01:
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Return current online learning status."""
        return {
            "n_updates": self._n_updates,
            "confidence_scale": round(self._confidence_scale, 4),
            "running_error_mean": round(self._running_error_mean, 6),
            "running_error_std": round(np.sqrt(max(self._running_error_var, 0)), 6),
            "pending_samples": len(self._pending_samples),
            "should_retrain": self.should_retrain(),
            "n_tracked_features": len(self._feature_means),
        }

    def _save_state(self) -> None:
        """Persist state to JSON."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "n_updates": self._n_updates,
                "confidence_scale": self._confidence_scale,
                "running_error_mean": self._running_error_mean,
                "running_error_var": self._running_error_var,
                "feature_means": self._feature_means,
                "feature_vars": self._feature_vars,
            }
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save online learning state: %s", e)

    def load_state(self) -> bool:
        """Load persisted state. Returns True if state was loaded."""
        try:
            if not self.state_path.exists():
                return False
            with open(self.state_path) as f:
                state = json.load(f)
            self._n_updates = state.get("n_updates", 0)
            self._confidence_scale = state.get("confidence_scale", 1.0)
            self._running_error_mean = state.get("running_error_mean", 0.0)
            self._running_error_var = state.get("running_error_var", 1.0)
            self._feature_means = state.get("feature_means", {})
            self._feature_vars = state.get("feature_vars", {})
            return True
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load online learning state: %s", e)
            return False
