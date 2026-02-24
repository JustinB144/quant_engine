"""
Distribution Shift Detection — CUSUM and PSI methods.

Detects when market data distributions have shifted away from the training
distribution, providing an early warning signal for model staleness.

Two complementary methods:
    1. CUSUM (Cumulative Sum): detects persistent mean shifts in prediction
       errors.  Best at catching gradual alpha decay.
    2. PSI (Population Stability Index): detects feature distribution changes
       between training and current windows.  Best at catching regime changes
       that invalidate the feature space.

Integration points:
    - RetrainTrigger: shift detection can reduce the retrain interval
    - Health checks: PSI scores feed into the system health dashboard

References:
    - Page, E.S. (1954). "Continuous inspection schemes." Biometrika.
    - Wu, D. & Olson, D.L. (2010). "Enterprise Risk Management in Finance."
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CUSUMResult:
    """Result of CUSUM change-point detection."""
    shift_detected: bool
    cusum_statistic: float
    threshold: float
    direction: str  # "positive", "negative", or "none"
    changepoint_index: Optional[int] = None


@dataclass
class PSIResult:
    """Result of Population Stability Index analysis."""
    shift_detected: bool
    avg_psi: float
    max_psi: float
    threshold: float
    top_shifted_features: List[Tuple[str, float]] = field(default_factory=list)
    psi_scores: Dict[str, float] = field(default_factory=dict)


class DistributionShiftDetector:
    """
    Detect when market data distribution has shifted from training distribution.

    Two methods:
        1. CUSUM: detects mean shift in prediction errors
        2. PSI: detects feature distribution shift vs training data

    Usage:
        detector = DistributionShiftDetector()
        detector.set_reference(training_features)

        # Later, check for shift:
        psi_result = detector.check_psi(current_features)
        cusum_result = detector.check_cusum(recent_prediction_errors)

        if psi_result.shift_detected or cusum_result.shift_detected:
            trigger_early_retrain()
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        psi_threshold: float = 0.25,
        reference_window: int = 252,
        n_bins: int = 10,
    ):
        """
        Args:
            cusum_threshold: Number of standard deviations for CUSUM alarm.
                Higher values reduce false alarms at the cost of slower detection.
            psi_threshold: PSI threshold for declaring distribution shift.
                Industry convention: <0.10 = stable, 0.10-0.25 = moderate,
                >0.25 = significant shift.
            reference_window: Number of observations in reference distribution.
            n_bins: Number of histogram bins for PSI computation.
        """
        self.cusum_threshold = cusum_threshold
        self.psi_threshold = psi_threshold
        self.reference_window = reference_window
        self.n_bins = n_bins
        self._reference_distributions: Dict[str, Dict] = {}
        self._reference_set = False

    def set_reference(self, features: pd.DataFrame) -> None:
        """Store reference distributions from training data.

        Computes histogram bin edges and normalized frequencies for each
        feature column.  These are used as the baseline for PSI comparison.

        Args:
            features: Training feature DataFrame.  Columns are feature names.
        """
        self._reference_distributions = {}
        for col in features.columns:
            vals = features[col].dropna().values
            if len(vals) < 20:
                continue
            hist, bin_edges = np.histogram(vals, bins=self.n_bins)
            freq = hist.astype(float) / max(1, len(vals))
            self._reference_distributions[col] = {
                "bins": bin_edges,
                "hist": freq,
                "n_samples": len(vals),
            }
        self._reference_set = True

    def check_cusum(
        self,
        prediction_errors: pd.Series,
        target_mean: float = 0.0,
    ) -> CUSUMResult:
        """Detect mean shift in prediction errors using two-sided CUSUM.

        The CUSUM algorithm accumulates deviations from the target mean.
        When the accumulated sum exceeds a threshold (scaled by the error
        standard deviation), a shift is declared.

        Uses the two-sided variant to detect both positive and negative
        mean shifts.

        Args:
            prediction_errors: Series of (actual - predicted) errors.
            target_mean: Expected mean of errors under H0 (typically 0).

        Returns:
            CUSUMResult with shift detection outcome and diagnostics.
        """
        if len(prediction_errors) < 10:
            return CUSUMResult(
                shift_detected=False,
                cusum_statistic=0.0,
                threshold=0.0,
                direction="none",
            )

        errors = prediction_errors.dropna().values - target_mean
        n = len(errors)
        std = np.std(errors, ddof=1)
        if std < 1e-12:
            return CUSUMResult(
                shift_detected=False,
                cusum_statistic=0.0,
                threshold=0.0,
                direction="none",
            )

        # Two-sided CUSUM
        # S_pos tracks upward shifts, S_neg tracks downward shifts
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)

        # Allowance parameter (slack): half standard deviation
        # This prevents small random fluctuations from accumulating
        k = 0.5 * std

        for i in range(1, n):
            s_pos[i] = max(0.0, s_pos[i - 1] + errors[i] - k)
            s_neg[i] = min(0.0, s_neg[i - 1] + errors[i] + k)

        threshold = self.cusum_threshold * std
        max_pos = float(np.max(s_pos))
        min_neg = float(np.min(s_neg))

        pos_detected = max_pos > threshold
        neg_detected = abs(min_neg) > threshold

        if pos_detected and (not neg_detected or max_pos >= abs(min_neg)):
            changepoint = int(np.argmax(s_pos))
            return CUSUMResult(
                shift_detected=True,
                cusum_statistic=max_pos,
                threshold=threshold,
                direction="positive",
                changepoint_index=changepoint,
            )
        elif neg_detected:
            changepoint = int(np.argmin(s_neg))
            return CUSUMResult(
                shift_detected=True,
                cusum_statistic=abs(min_neg),
                threshold=threshold,
                direction="negative",
                changepoint_index=changepoint,
            )
        else:
            cusum_stat = max(max_pos, abs(min_neg))
            return CUSUMResult(
                shift_detected=False,
                cusum_statistic=cusum_stat,
                threshold=threshold,
                direction="none",
            )

    def check_psi(self, current_features: pd.DataFrame) -> PSIResult:
        """Detect feature distribution shift using Population Stability Index.

        PSI measures the symmetric KL-divergence between reference and current
        feature distributions:

            PSI = sum((current_i - reference_i) * ln(current_i / reference_i))

        Industry thresholds:
            PSI < 0.10: No significant shift
            0.10 <= PSI < 0.25: Moderate shift (monitor)
            PSI >= 0.25: Significant shift (action needed)

        Args:
            current_features: Current feature DataFrame to compare against
                reference distributions.

        Returns:
            PSIResult with per-feature PSI scores and shift detection.
        """
        if not self._reference_set:
            return PSIResult(
                shift_detected=False,
                avg_psi=0.0,
                max_psi=0.0,
                threshold=self.psi_threshold,
            )

        psi_scores: Dict[str, float] = {}

        for col in current_features.columns:
            if col not in self._reference_distributions:
                continue

            ref = self._reference_distributions[col]
            current_vals = current_features[col].dropna().values
            if len(current_vals) < 10:
                continue

            # Compute histogram using reference bin edges
            current_hist, _ = np.histogram(current_vals, bins=ref["bins"])
            current_freq = current_hist.astype(float) / max(1, len(current_vals))

            # PSI computation with epsilon smoothing to avoid log(0)
            eps = 1e-6
            ref_h = np.maximum(ref["hist"], eps)
            cur_h = np.maximum(current_freq, eps)
            psi = float(np.sum((cur_h - ref_h) * np.log(cur_h / ref_h)))
            psi_scores[col] = psi

        if not psi_scores:
            return PSIResult(
                shift_detected=False,
                avg_psi=0.0,
                max_psi=0.0,
                threshold=self.psi_threshold,
            )

        max_psi = max(psi_scores.values())
        avg_psi = float(np.mean(list(psi_scores.values())))
        shift_detected = avg_psi > self.psi_threshold

        # Top shifted features sorted by PSI descending
        top_shifted = sorted(psi_scores.items(), key=lambda x: -x[1])[:10]

        return PSIResult(
            shift_detected=shift_detected,
            avg_psi=avg_psi,
            max_psi=max_psi,
            threshold=self.psi_threshold,
            top_shifted_features=top_shifted,
            psi_scores=psi_scores,
        )

    def check_all(
        self,
        prediction_errors: Optional[pd.Series] = None,
        current_features: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Run all available shift detection checks.

        Args:
            prediction_errors: Recent prediction errors for CUSUM.
            current_features: Current features for PSI.

        Returns:
            Dict with combined results and overall shift assessment.
        """
        results: Dict = {
            "any_shift_detected": False,
            "cusum": None,
            "psi": None,
        }

        if prediction_errors is not None and len(prediction_errors) >= 10:
            cusum = self.check_cusum(prediction_errors)
            results["cusum"] = {
                "shift_detected": cusum.shift_detected,
                "cusum_statistic": cusum.cusum_statistic,
                "threshold": cusum.threshold,
                "direction": cusum.direction,
                "changepoint_index": cusum.changepoint_index,
            }
            if cusum.shift_detected:
                results["any_shift_detected"] = True

        if current_features is not None and self._reference_set:
            psi = self.check_psi(current_features)
            results["psi"] = {
                "shift_detected": psi.shift_detected,
                "avg_psi": psi.avg_psi,
                "max_psi": psi.max_psi,
                "threshold": psi.threshold,
                "top_shifted_features": psi.top_shifted_features,
            }
            if psi.shift_detected:
                results["any_shift_detected"] = True

        return results
