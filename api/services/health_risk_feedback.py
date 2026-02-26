"""Health-to-risk feedback loop — Spec 09.

Feeds overall system health score into the risk governor to
proportionally reduce position sizes when health degrades.

The mapping from health score to size multiplier uses linear
interpolation between configurable breakpoints (from health.yaml).

Integration point: called after position sizing but before order
submission in paper_trader.py and backtest/engine.py.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HealthRiskGate:
    """Scales position sizes based on system health score.

    Parameters
    ----------
    health_to_size_table : dict
        Mapping from health_score (0–1 scale, i.e. 0–100 / 100)
        to size_multiplier.  Linear interpolation between points.
    smoothing_alpha : float
        Exponential smoothing factor for health score updates.
        0 = no smoothing (immediate), 1 = full smoothing (never changes).
    halt_threshold : float
        Minimum health score (0–1 scale) below which trading is halted.
    enabled : bool
        If False, all methods return identity (no effect on sizing).
    """

    def __init__(
        self,
        health_to_size_table: Optional[Dict[float, float]] = None,
        smoothing_alpha: float = 0.3,
        halt_threshold: float = 0.05,
        enabled: bool = True,
    ):
        self.enabled = enabled

        # Default table (health score on 0–1 scale → multiplier)
        if health_to_size_table is None:
            health_to_size_table = {
                1.00: 1.00,
                0.80: 0.95,
                0.60: 0.80,
                0.40: 0.50,
                0.20: 0.20,
                0.00: 0.05,
            }

        # Sort by health score ascending for interpolation
        self._table: List[Tuple[float, float]] = sorted(
            health_to_size_table.items(), key=lambda x: x[0],
        )
        self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
        self.halt_threshold = halt_threshold

        # Smoothed health score (None until first update)
        self._smoothed_health: Optional[float] = None

    def update_health(self, raw_health_score: float) -> float:
        """Update the smoothed health score with a new observation.

        Parameters
        ----------
        raw_health_score : float
            Current health score on 0–100 scale.

        Returns
        -------
        Smoothed health score on 0–100 scale.
        """
        normalized = max(0.0, min(100.0, raw_health_score))

        if self._smoothed_health is None:
            self._smoothed_health = normalized
        else:
            alpha = self.smoothing_alpha
            self._smoothed_health = (
                alpha * self._smoothed_health + (1 - alpha) * normalized
            )

        return self._smoothed_health

    def compute_size_multiplier(
        self,
        health_score: Optional[float] = None,
    ) -> float:
        """Compute position size multiplier from health score.

        Uses linear interpolation between table breakpoints.

        Parameters
        ----------
        health_score : float, optional
            Health score on 0–100 scale.  If None, uses the last
            smoothed health score.

        Returns
        -------
        Multiplier in [0, 1].  1.0 = full size, 0.05 = near-halt.
        """
        if not self.enabled:
            return 1.0

        if health_score is None:
            health_score = self._smoothed_health
        if health_score is None:
            return 1.0  # No health data yet → no restriction

        # Convert 0–100 scale to 0–1 for table lookup
        h = max(0.0, min(1.0, health_score / 100.0))

        # Edge cases
        if h <= self._table[0][0]:
            return self._table[0][1]
        if h >= self._table[-1][0]:
            return self._table[-1][1]

        # Linear interpolation between the two surrounding breakpoints
        for i in range(len(self._table) - 1):
            h_low, m_low = self._table[i]
            h_high, m_high = self._table[i + 1]
            if h_low <= h <= h_high:
                t = (h - h_low) / (h_high - h_low) if h_high != h_low else 0.0
                return m_low + t * (m_high - m_low)

        return 1.0  # Fallback

    def apply_health_gate(
        self,
        position_size: float,
        health_score: Optional[float] = None,
    ) -> float:
        """Apply health-based sizing reduction to a single position.

        Parameters
        ----------
        position_size : float
            Original position size (fraction of portfolio, e.g. 0.05).
        health_score : float, optional
            Health score on 0–100 scale.

        Returns
        -------
        Adjusted position size.
        """
        if not self.enabled:
            return position_size

        multiplier = self.compute_size_multiplier(health_score)
        adjusted = position_size * multiplier

        if multiplier < 1.0:
            logger.info(
                "Health gate: size %.4f → %.4f (multiplier=%.3f, health=%.1f)",
                position_size, adjusted, multiplier,
                health_score if health_score is not None else -1.0,
            )

        return adjusted

    def apply_health_gate_weights(
        self,
        weights: np.ndarray,
        health_score: Optional[float] = None,
    ) -> np.ndarray:
        """Apply health-based sizing reduction to a weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Array of position weights (fractions of portfolio).
        health_score : float, optional
            Health score on 0–100 scale.

        Returns
        -------
        Adjusted weight vector.
        """
        if not self.enabled:
            return weights

        multiplier = self.compute_size_multiplier(health_score)
        return weights * multiplier

    def should_halt_trading(
        self,
        health_score: Optional[float] = None,
    ) -> bool:
        """Check if health is so low that trading should be halted.

        Parameters
        ----------
        health_score : float, optional
            Health score on 0–100 scale.

        Returns
        -------
        True if trading should be halted.
        """
        if not self.enabled:
            return False

        if health_score is None:
            health_score = self._smoothed_health
        if health_score is None:
            return False

        # Convert 0–100 to 0–1 for comparison
        h = health_score / 100.0
        if h < self.halt_threshold:
            logger.critical(
                "TRADING HALT: health score %.1f below halt threshold %.2f",
                health_score, self.halt_threshold * 100,
            )
            return True
        return False

    def get_status(self) -> Dict:
        """Return current gate status for diagnostics."""
        health = self._smoothed_health
        return {
            "enabled": self.enabled,
            "smoothed_health": round(health, 2) if health is not None else None,
            "current_multiplier": round(self.compute_size_multiplier(), 4),
            "halt_threshold": self.halt_threshold * 100,
            "would_halt": self.should_halt_trading(),
        }


def load_health_risk_config() -> Dict:
    """Load risk feedback configuration from health.yaml."""
    try:
        import yaml

        config_path = (
            __import__("pathlib").Path(__file__).parent.parent.parent
            / "config" / "health.yaml"
        )
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("risk_feedback", {})
    except ImportError:
        logger.debug("PyYAML not available; using default risk feedback config")
    except Exception as e:
        logger.warning("Failed to load risk feedback config: %s", e)
    return {}


def create_health_risk_gate() -> HealthRiskGate:
    """Factory: create a HealthRiskGate from health.yaml config."""
    config = load_health_risk_config()

    table = config.get("health_to_size_multiplier")
    if table is not None:
        # YAML keys may be strings; convert to float
        table = {float(k): float(v) for k, v in table.items()}

    return HealthRiskGate(
        health_to_size_table=table,
        smoothing_alpha=config.get("smoothing_alpha", 0.3),
        halt_threshold=config.get("halt_threshold", 0.05),
        enabled=config.get("enabled", True),
    )
