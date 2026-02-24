"""
Confidence Calibration --- Platt scaling and isotonic regression.

Raw model confidence scores are often poorly calibrated: a prediction
with "70% confidence" may in reality be correct far more or less than
70% of the time.  This module post-hoc calibrates raw scores so they
approximate true probabilities.

Supported methods:
    - ``platt``: Logistic (sigmoid) regression mapping (Platt 1999).
    - ``isotonic``: Isotonic regression -- a non-parametric, monotone-
      increasing mapping (Zadrozny & Elkan 2002).

If scikit-learn is unavailable, a simple linear rescaling fallback is
used instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Try to import sklearn calibration helpers; set flags for fallback
# ---------------------------------------------------------------------------
_HAS_SKLEARN = False
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    _HAS_SKLEARN = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Simple linear rescaling fallback (no sklearn)
# ---------------------------------------------------------------------------

class _LinearRescaler:
    """Maps raw scores to [0, 1] via min-max linear rescaling.

    Used as a last-resort fallback when sklearn is not installed.
    """

    def __init__(self) -> None:
        """Initialize _LinearRescaler."""
        self._min: float = 0.0
        self._max: float = 1.0

    def fit(self, raw: np.ndarray, _outcomes: np.ndarray) -> None:  # noqa: ARG002
        """Fit the transformer to the provided data."""
        self._min = float(np.min(raw))
        self._max = float(np.max(raw))
        if self._max - self._min < 1e-12:
            self._max = self._min + 1.0

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Transform the provided data using the fitted parameters."""
        scaled = (raw - self._min) / (self._max - self._min)
        return np.clip(scaled, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """Post-hoc confidence calibration via Platt scaling or isotonic regression.

    Parameters
    ----------
    method : str
        ``"isotonic"`` (default) or ``"platt"``.

    Examples
    --------
    >>> cal = ConfidenceCalibrator(method="isotonic")
    >>> cal.fit(raw_confidence, actual_binary_outcomes)
    >>> calibrated = cal.transform(new_raw_confidence)
    """

    def __init__(self, method: str = "isotonic") -> None:
        """Initialize ConfidenceCalibrator."""
        if method not in ("platt", "isotonic"):
            raise ValueError(f"method must be 'platt' or 'isotonic', got '{method}'")
        self.method = method
        self._fitted = False
        self._model: Optional[object] = None
        self._fallback: Optional[_LinearRescaler] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, raw_confidence: np.ndarray, actual_outcomes: np.ndarray) -> None:
        """Fit the calibration mapping.

        Parameters
        ----------
        raw_confidence : np.ndarray
            1-D array of raw (uncalibrated) confidence scores, typically
            in [0, 1] but not required.
        actual_outcomes : np.ndarray
            1-D binary array (0 or 1) indicating whether the model's
            prediction was correct.
        """
        raw = np.asarray(raw_confidence, dtype=float).ravel()
        outcomes = np.asarray(actual_outcomes, dtype=float).ravel()

        if len(raw) != len(outcomes):
            raise ValueError(
                f"raw_confidence length ({len(raw)}) != actual_outcomes length ({len(outcomes)})"
            )

        if len(raw) < 2:
            raise ValueError("Need at least 2 samples to fit calibration")

        if _HAS_SKLEARN:
            try:
                self._fit_sklearn(raw, outcomes)
                self._fitted = True
                return
            except (ValueError, RuntimeError):
                # Fall through to linear fallback
                pass

        # Fallback: simple linear rescaling
        self._fallback = _LinearRescaler()
        self._fallback.fit(raw, outcomes)
        self._fitted = True

    def _fit_sklearn(self, raw: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit using sklearn implementations."""
        if self.method == "isotonic":
            model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            model.fit(raw, outcomes)
            self._model = model
        else:
            # Platt scaling: logistic regression on raw scores
            X = raw.reshape(-1, 1)
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )
            lr.fit(X, (outcomes >= 0.5).astype(int))
            self._model = lr

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, raw_confidence: np.ndarray) -> np.ndarray:
        """Transform raw confidence scores into calibrated probabilities.

        Parameters
        ----------
        raw_confidence : np.ndarray
            1-D array of raw confidence scores to calibrate.

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("ConfidenceCalibrator has not been fitted yet. Call fit() first.")

        raw = np.asarray(raw_confidence, dtype=float).ravel()

        # Fallback path
        if self._fallback is not None:
            return self._fallback.transform(raw)

        # sklearn path
        if self.method == "isotonic":
            result = self._model.transform(raw)  # type: ignore[union-attr]
            return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)
        else:
            # Platt: use predict_proba from LogisticRegression
            X = raw.reshape(-1, 1)
            proba = self._model.predict_proba(X)  # type: ignore[union-attr]
            # predict_proba returns shape (n, 2); column 1 = P(outcome=1)
            return np.clip(proba[:, 1], 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        raw_confidence: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> np.ndarray:
        """Fit on the data and return calibrated scores in one step."""
        self.fit(raw_confidence, actual_outcomes)
        return self.transform(raw_confidence)

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def backend(self) -> str:
        """Return which backend is in use: 'sklearn' or 'linear_fallback'."""
        if self._fallback is not None:
            return "linear_fallback"
        return "sklearn"

    def __repr__(self) -> str:
        """Return a debug-friendly string representation of ConfidenceCalibrator."""
        status = "fitted" if self._fitted else "unfitted"
        backend = self.backend if self._fitted else "n/a"
        return (
            f"ConfidenceCalibrator(method='{self.method}', "
            f"status={status}, backend={backend})"
        )


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE measures how well predicted probabilities match observed frequencies.
    Lower ECE indicates better calibration. A perfectly calibrated model
    has ECE = 0.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities in [0, 1].
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1).
    n_bins : int
        Number of equal-width bins to partition predictions into.

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float).ravel()
    actual_outcomes = np.asarray(actual_outcomes, dtype=float).ravel()

    if len(predicted_probs) == 0:
        return 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predicted_probs)

    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        # Include upper boundary in last bin
        if i == n_bins - 1:
            mask = mask | (predicted_probs == bins[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_pred = predicted_probs[mask].mean()
            avg_actual = actual_outcomes[mask].mean()
            ece += (n_in_bin / total) * abs(avg_pred - avg_actual)

    return float(ece)


def compute_reliability_curve(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability curve data for calibration diagnostics.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities in [0, 1].
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1).
    n_bins : int
        Number of bins.

    Returns
    -------
    dict
        Keys: 'bin_centers', 'observed_freq', 'avg_predicted', 'bin_counts'.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float).ravel()
    actual_outcomes = np.asarray(actual_outcomes, dtype=float).ravel()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_freq = []
    avg_predicted = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted_probs == bins[i + 1])
        n_in_bin = int(mask.sum())
        bin_counts.append(n_in_bin)
        if n_in_bin > 0:
            bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
            observed_freq.append(float(actual_outcomes[mask].mean()))
            avg_predicted.append(float(predicted_probs[mask].mean()))
        else:
            bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
            observed_freq.append(np.nan)
            avg_predicted.append(np.nan)

    return {
        "bin_centers": bin_centers,
        "observed_freq": observed_freq,
        "avg_predicted": avg_predicted,
        "bin_counts": bin_counts,
    }
