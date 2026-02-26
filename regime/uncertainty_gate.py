"""Regime Uncertainty Gate — entropy-based position sizing modifier (SPEC_10 T3).

Computes a sizing multiplier from the entropy of the regime posterior
distribution.  High entropy (regime uncertain) reduces position sizes;
low entropy (regime certain) leaves sizes unchanged.

The gate also provides a binary ``should_assume_stress`` signal for
the promotion gate: when uncertainty is very high, conservatively
assume the stress (high_volatility) regime for risk limits.

Usage::

    gate = UncertaintyGate()
    multiplier = gate.compute_size_multiplier(uncertainty=0.6)
    adjusted_weights = gate.apply_uncertainty_gate(weights, uncertainty=0.6)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UncertaintyGate:
    """Reduce position sizes when regime detection is uncertain.

    Parameters
    ----------
    entropy_threshold : float
        Normalized entropy above which regime is flagged as uncertain.
    stress_threshold : float
        Normalized entropy above which the gate recommends assuming
        stress regime for risk limits.
    sizing_map : dict
        Mapping from entropy level to sizing multiplier.  Values between
        map entries are linearly interpolated.
    min_multiplier : float
        Floor for the sizing multiplier (never reduce below this).
    """

    def __init__(
        self,
        entropy_threshold: Optional[float] = None,
        stress_threshold: Optional[float] = None,
        sizing_map: Optional[Dict[float, float]] = None,
        min_multiplier: Optional[float] = None,
    ):
        from ..config import (
            REGIME_UNCERTAINTY_ENTROPY_THRESHOLD,
            REGIME_UNCERTAINTY_STRESS_THRESHOLD,
            REGIME_UNCERTAINTY_SIZING_MAP,
            REGIME_UNCERTAINTY_MIN_MULTIPLIER,
        )

        self.entropy_threshold = (
            entropy_threshold if entropy_threshold is not None
            else REGIME_UNCERTAINTY_ENTROPY_THRESHOLD
        )
        self.stress_threshold = (
            stress_threshold if stress_threshold is not None
            else REGIME_UNCERTAINTY_STRESS_THRESHOLD
        )
        self.sizing_map = sizing_map or dict(REGIME_UNCERTAINTY_SIZING_MAP)
        self.min_multiplier = (
            min_multiplier if min_multiplier is not None
            else REGIME_UNCERTAINTY_MIN_MULTIPLIER
        )

        # Pre-sort for interpolation
        self._map_keys = sorted(self.sizing_map.keys())
        self._map_vals = [self.sizing_map[k] for k in self._map_keys]

    def compute_size_multiplier(self, uncertainty: float) -> float:
        """Compute position sizing multiplier from regime uncertainty.

        Linearly interpolates between the configured sizing map entries.
        Clamps the result to [min_multiplier, 1.0].

        Parameters
        ----------
        uncertainty : float
            Normalized entropy of regime posterior in [0, 1].
            0 = certain, 1 = maximum uncertainty.

        Returns
        -------
        float
            Sizing multiplier in [min_multiplier, 1.0].
        """
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

        if not self._map_keys:
            return 1.0

        # Interpolate
        multiplier = float(np.interp(uncertainty, self._map_keys, self._map_vals))
        return float(np.clip(multiplier, self.min_multiplier, 1.0))

    def apply_uncertainty_gate(
        self,
        weights: np.ndarray,
        uncertainty: float,
    ) -> np.ndarray:
        """Apply the uncertainty gate to a weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Position weight vector (e.g., from portfolio optimizer).
        uncertainty : float
            Normalized entropy of regime posterior.

        Returns
        -------
        np.ndarray
            Adjusted weights (same shape).
        """
        multiplier = self.compute_size_multiplier(uncertainty)
        return np.asarray(weights, dtype=float) * multiplier

    def should_assume_stress(self, uncertainty: float) -> bool:
        """Whether uncertainty is high enough to assume stress regime.

        When True, downstream systems should use high_volatility regime
        constraints (tighter limits) regardless of the detected regime.

        Parameters
        ----------
        uncertainty : float
            Normalized entropy of regime posterior.

        Returns
        -------
        bool
        """
        return float(uncertainty) > self.stress_threshold

    def is_uncertain(self, uncertainty: float) -> bool:
        """Whether the regime detection is flagged as uncertain.

        Parameters
        ----------
        uncertainty : float
            Normalized entropy of regime posterior.

        Returns
        -------
        bool
        """
        return float(uncertainty) > self.entropy_threshold

    def gate_series(
        self,
        uncertainty_series: pd.Series,
    ) -> pd.DataFrame:
        """Apply the gate to an entire time series of uncertainties.

        Parameters
        ----------
        uncertainty_series : pd.Series
            Time series of normalized entropy values.

        Returns
        -------
        pd.DataFrame
            Columns: ``multiplier``, ``is_uncertain``, ``assume_stress``.
        """
        multipliers = uncertainty_series.apply(self.compute_size_multiplier)
        uncertain = uncertainty_series.apply(self.is_uncertain)
        stress = uncertainty_series.apply(self.should_assume_stress)

        return pd.DataFrame({
            "multiplier": multipliers,
            "is_uncertain": uncertain,
            "assume_stress": stress,
        }, index=uncertainty_series.index)
