"""
Conformal Prediction — distribution-free prediction intervals.

Implements split conformal prediction to construct prediction intervals
with guaranteed finite-sample coverage probability, regardless of the
underlying data distribution or model complexity.

Key properties:
    - No distributional assumptions (works with any base model)
    - Guaranteed coverage: P(Y in interval) >= 1 - alpha for finite samples
    - Finite-sample correction via ceil((n+1)*coverage)/n quantile
    - Integrates with position sizing: wider intervals -> smaller positions

References:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005).
      "Algorithmic Learning in a Random World." Springer.
    - Romano, Y., Patterson, E., & Candès, E. (2019).
      "Conformalized Quantile Regression." NeurIPS.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConformalInterval:
    """A single prediction interval."""
    point_prediction: float
    lower: float
    upper: float
    width: float


@dataclass
class ConformalCalibrationResult:
    """Result of conformal calibration on holdout data."""
    n_calibration: int
    coverage_target: float
    quantile: float
    mean_residual: float
    median_residual: float
    max_residual: float


class ConformalPredictor:
    """
    Split conformal prediction for distribution-free prediction intervals.

    Calibration phase:
        Given a set of calibration predictions and actuals, compute the
        non-conformity scores (absolute residuals) and find the quantile
        that achieves the target coverage.

    Prediction phase:
        For new point predictions, construct intervals as:
            [prediction - quantile, prediction + quantile]

    Integration with position sizing:
        Wider prediction intervals indicate more uncertainty, which should
        result in smaller position sizes:

            interval_width = upper - lower
            uncertainty_scalar = 1.0 / (1.0 + interval_width / avg_interval_width)
            position_size *= uncertainty_scalar

    Usage:
        conformal = ConformalPredictor(coverage=0.90)
        conformal.calibrate(holdout_predictions, holdout_actuals)

        # Get intervals for new predictions
        intervals = conformal.predict_intervals_batch(new_predictions)
        # intervals[:, 0] = lower bounds
        # intervals[:, 1] = upper bounds

        # Uncertainty-aware position sizing
        widths = intervals[:, 1] - intervals[:, 0]
        scalars = conformal.uncertainty_scalars(widths)
    """

    def __init__(self, coverage: float = 0.90):
        """
        Args:
            coverage: Target coverage probability (0 < coverage < 1).
                Default 0.90 means 90% of true values should fall within
                the prediction interval.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self._quantile: Optional[float] = None
        self._calibration_residuals: Optional[np.ndarray] = None
        self._calibration_result: Optional[ConformalCalibrationResult] = None

    @property
    def is_calibrated(self) -> bool:
        """Whether the predictor has been calibrated."""
        return self._quantile is not None

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> ConformalCalibrationResult:
        """Compute non-conformity scores from a calibration set.

        The calibration set should be held out from training — these are
        predictions the model made on data it was not trained on.

        Args:
            predictions: Model predictions on calibration data.
            actuals: True target values for calibration data.

        Returns:
            ConformalCalibrationResult with calibration diagnostics.

        Raises:
            ValueError: If fewer than 10 calibration points provided.
        """
        predictions = np.asarray(predictions, dtype=float).ravel()
        actuals = np.asarray(actuals, dtype=float).ravel()

        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions and actuals must have same length, "
                f"got {len(predictions)} and {len(actuals)}"
            )

        # Remove NaN pairs
        mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[mask]
        actuals = actuals[mask]

        if len(predictions) < 10:
            raise ValueError(
                f"Need at least 10 calibration points, got {len(predictions)}"
            )

        # Non-conformity scores = absolute residuals
        residuals = np.abs(actuals - predictions)
        self._calibration_residuals = np.sort(residuals)

        # Finite-sample correction: use ceil((n+1)*coverage)/n quantile
        # This guarantees P(Y in interval) >= coverage for finite samples
        n = len(residuals)
        q_level = np.ceil((n + 1) * self.coverage) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(residuals, q_level))

        self._calibration_result = ConformalCalibrationResult(
            n_calibration=n,
            coverage_target=self.coverage,
            quantile=self._quantile,
            mean_residual=float(np.mean(residuals)),
            median_residual=float(np.median(residuals)),
            max_residual=float(np.max(residuals)),
        )

        return self._calibration_result

    def predict_interval(
        self,
        point_prediction: float,
    ) -> ConformalInterval:
        """Return prediction interval for a single point prediction.

        Args:
            point_prediction: The model's point prediction.

        Returns:
            ConformalInterval with lower, upper bounds and width.

        Raises:
            ValueError: If not calibrated yet.
        """
        if self._quantile is None:
            raise ValueError("Must calibrate before predicting intervals")

        lower = point_prediction - self._quantile
        upper = point_prediction + self._quantile

        return ConformalInterval(
            point_prediction=point_prediction,
            lower=lower,
            upper=upper,
            width=upper - lower,
        )

    def predict_intervals_batch(
        self,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Compute prediction intervals for a batch of predictions.

        Args:
            predictions: Array of point predictions.

        Returns:
            Array of shape (n, 2) with columns [lower, upper].

        Raises:
            ValueError: If not calibrated yet.
        """
        if self._quantile is None:
            raise ValueError("Must calibrate before predicting intervals")

        predictions = np.asarray(predictions, dtype=float).ravel()
        return np.column_stack([
            predictions - self._quantile,
            predictions + self._quantile,
        ])

    def uncertainty_scalars(
        self,
        interval_widths: np.ndarray,
    ) -> np.ndarray:
        """Compute position-sizing scalars from prediction interval widths.

        Wider intervals indicate more uncertainty and should result in
        smaller positions.  The scalar is:

            scalar = 1.0 / (1.0 + width / avg_width)

        This maps:
            - Average-width interval -> scalar ~0.50
            - Narrow interval (half avg) -> scalar ~0.67
            - Wide interval (2x avg) -> scalar ~0.33
            - Zero-width interval -> scalar = 1.0

        Args:
            interval_widths: Array of prediction interval widths.

        Returns:
            Array of scalars in (0, 1] for position sizing.
        """
        widths = np.asarray(interval_widths, dtype=float)
        avg_width = np.mean(widths)
        if avg_width <= 1e-12:
            return np.ones_like(widths)
        return 1.0 / (1.0 + widths / avg_width)

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict:
        """Evaluate empirical coverage on a test set.

        Args:
            predictions: Point predictions on test data.
            actuals: True values on test data.

        Returns:
            Dict with coverage metrics.
        """
        if self._quantile is None:
            raise ValueError("Must calibrate before evaluating coverage")

        predictions = np.asarray(predictions, dtype=float).ravel()
        actuals = np.asarray(actuals, dtype=float).ravel()

        mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[mask]
        actuals = actuals[mask]

        intervals = self.predict_intervals_batch(predictions)
        covered = (actuals >= intervals[:, 0]) & (actuals <= intervals[:, 1])

        return {
            "empirical_coverage": float(covered.mean()),
            "target_coverage": self.coverage,
            "n_test": len(predictions),
            "n_covered": int(covered.sum()),
            "interval_width": float(self._quantile * 2),
            "coverage_gap": float(covered.mean() - self.coverage),
        }

    def to_dict(self) -> Dict:
        """Serialize calibration state for persistence."""
        return {
            "coverage": self.coverage,
            "quantile": self._quantile,
            "n_calibration": (
                self._calibration_result.n_calibration
                if self._calibration_result else 0
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConformalPredictor":
        """Restore from serialized state."""
        cp = cls(coverage=d.get("coverage", 0.90))
        cp._quantile = d.get("quantile")
        return cp
