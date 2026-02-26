"""
Cost model calibrator for per-market-cap-segment impact coefficients.

Spec 06 T4: Adjusts EXEC_IMPACT_COEFF_BPS per market cap segment
(micro, small, mid, large) based on realized impact from historical
trades.  Calibrated coefficients are stored to disk (joblib) and loaded
for production use.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CostCalibrator:
    """Per-market-cap-segment impact coefficient calibrator.

    Accumulates historical trade observations (expected vs realized cost),
    then fits a median-robust impact coefficient per segment with EMA
    smoothing to avoid wild swings.

    Parameters
    ----------
    default_coefficients : dict
        Initial impact coefficients by segment, e.g.
        ``{"micro": 40.0, "small": 30.0, "mid": 20.0, "large": 15.0}``.
    marketcap_thresholds : dict
        Segment boundary thresholds in USD, e.g.
        ``{"micro": 300e6, "small": 2e9, "mid": 10e9}``.
    spread_bps : float
        Base spread in bps (used to decompose realized cost into
        spread + impact components).
    min_total_trades : int
        Minimum total trades before any calibration is allowed.
    min_segment_trades : int
        Minimum trades per segment for that segment to be recalibrated.
    smoothing : float
        Weight on the *new* coefficient in EMA update (0-1).
        ``new = smoothing * observed + (1-smoothing) * old``.
    model_dir : Path or str
        Directory to persist calibrated coefficients.
    """

    def __init__(
        self,
        default_coefficients: Optional[Dict[str, float]] = None,
        marketcap_thresholds: Optional[Dict[str, float]] = None,
        spread_bps: float = 3.0,
        min_total_trades: int = 100,
        min_segment_trades: int = 20,
        smoothing: float = 0.30,
        model_dir: Optional[Path] = None,
    ):
        self._default = dict(default_coefficients or {
            "micro": 40.0, "small": 30.0, "mid": 20.0, "large": 15.0,
        })
        self._thresholds = dict(marketcap_thresholds or {
            "micro": 300e6, "small": 2e9, "mid": 10e9,
        })
        self._spread_bps = float(max(0.0, spread_bps))
        self._min_total = int(max(10, min_total_trades))
        self._min_segment = int(max(5, min_segment_trades))
        self._smoothing = float(np.clip(smoothing, 0.05, 0.95))
        self._model_dir = Path(model_dir) if model_dir else None

        # Current coefficients (initialized from defaults)
        self._coefficients: Dict[str, float] = dict(self._default)

        # Trade history per segment
        self._trade_history: Dict[str, List[Dict]] = defaultdict(list)
        self._total_trades = 0

        # Try to load persisted coefficients
        self._load_calibration()

    # ── Market Cap Segmentation ───────────────────────────────────────

    def get_marketcap_segment(self, market_cap: float) -> str:
        """Classify market cap into segment.

        Parameters
        ----------
        market_cap : float
            Market capitalization in USD.

        Returns
        -------
        str
            One of "micro", "small", "mid", "large".
        """
        if not np.isfinite(market_cap) or market_cap <= 0:
            return "mid"  # safe default for unknown
        if market_cap < self._thresholds.get("micro", 300e6):
            return "micro"
        elif market_cap < self._thresholds.get("small", 2e9):
            return "small"
        elif market_cap < self._thresholds.get("mid", 10e9):
            return "mid"
        else:
            return "large"

    # ── Coefficient Access ────────────────────────────────────────────

    def get_impact_coeff(self, market_cap: float) -> float:
        """Get calibrated impact coefficient for a given market cap.

        Parameters
        ----------
        market_cap : float
            Market capitalization in USD.

        Returns
        -------
        float
            Impact coefficient in bps per sqrt(participation).
        """
        segment = self.get_marketcap_segment(market_cap)
        return self._coefficients.get(
            segment, self._default.get(segment, 25.0)
        )

    def get_impact_coeff_by_segment(self, segment: str) -> float:
        """Get calibrated impact coefficient for a named segment."""
        return self._coefficients.get(
            segment, self._default.get(segment, 25.0)
        )

    @property
    def coefficients(self) -> Dict[str, float]:
        """Current calibrated coefficients (copy)."""
        return dict(self._coefficients)

    # ── Trade Recording ───────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        market_cap: float,
        participation_rate: float,
        realized_cost_bps: float,
    ) -> None:
        """Record a historical trade for future calibration.

        Parameters
        ----------
        symbol : str
            Ticker or PERMNO.
        market_cap : float
            Market cap in USD.
        participation_rate : float
            Fraction of daily volume executed.
        realized_cost_bps : float
            Total realized slippage in basis points (spread + impact).
        """
        if participation_rate <= 0 or not np.isfinite(participation_rate):
            return
        if not np.isfinite(realized_cost_bps):
            return

        segment = self.get_marketcap_segment(market_cap)
        self._trade_history[segment].append({
            "symbol": symbol,
            "participation_rate": float(participation_rate),
            "realized_cost_bps": float(realized_cost_bps),
        })
        self._total_trades += 1

    # ── Calibration ───────────────────────────────────────────────────

    def calibrate(self) -> Dict[str, Dict]:
        """Calibrate impact coefficients per segment from recorded trades.

        Uses the sqrt(participation) model:
            realized_cost = spread_bps + impact_coeff * sqrt(participation)
        Solves for impact_coeff per trade, then takes the median (robust
        to outliers) and applies EMA smoothing against the prior coefficient.

        Returns
        -------
        dict
            Per-segment calibration results with keys: trades_count,
            old_coeff, observed_coeff, smoothed_coeff.  Empty if
            insufficient data.
        """
        if self._total_trades < self._min_total:
            logger.info(
                "CostCalibrator: insufficient trades (%d < %d) for calibration",
                self._total_trades,
                self._min_total,
            )
            return {}

        results: Dict[str, Dict] = {}

        for segment in ("micro", "small", "mid", "large"):
            trades = self._trade_history.get(segment, [])
            if len(trades) < self._min_segment:
                logger.debug(
                    "CostCalibrator: skipping %s segment (%d < %d trades)",
                    segment,
                    len(trades),
                    self._min_segment,
                )
                continue

            # Solve for impact coefficient per trade
            coefficients: List[float] = []
            for t in trades:
                part = t["participation_rate"]
                cost = t["realized_cost_bps"]
                sqrt_part = np.sqrt(part)
                if sqrt_part < 1e-9:
                    continue
                # impact = realized_cost - half_spread
                net_impact = max(0.0, cost - 0.5 * self._spread_bps)
                coeff = net_impact / sqrt_part
                if np.isfinite(coeff) and coeff > 0:
                    coefficients.append(coeff)

            if not coefficients:
                continue

            observed = float(np.median(coefficients))
            old_coeff = self._coefficients.get(
                segment, self._default.get(segment, 25.0)
            )
            smoothed = (
                self._smoothing * observed
                + (1.0 - self._smoothing) * old_coeff
            )
            # Enforce reasonable bounds
            smoothed = float(np.clip(smoothed, 5.0, 100.0))
            self._coefficients[segment] = smoothed

            results[segment] = {
                "trades_count": len(trades),
                "old_coeff": old_coeff,
                "observed_coeff": observed,
                "smoothed_coeff": smoothed,
            }
            logger.info(
                "CostCalibrator: %s segment calibrated: %.1f → %.1f "
                "(observed=%.1f, %d trades)",
                segment,
                old_coeff,
                smoothed,
                observed,
                len(trades),
            )

        # Persist calibrated coefficients
        self._save_calibration()
        return results

    # ── Persistence ───────────────────────────────────────────────────

    def _save_calibration(self) -> None:
        """Persist calibrated coefficients to disk."""
        if self._model_dir is None:
            return
        try:
            import joblib

            self._model_dir.mkdir(parents=True, exist_ok=True)
            path = self._model_dir / "cost_calibration_current.joblib"
            joblib.dump(self._coefficients, path)
            logger.info("CostCalibrator: saved to %s", path)
        except ImportError:
            logger.debug("joblib not available; skipping calibration save")
        except Exception as e:
            logger.warning("CostCalibrator: save failed: %s", e)

    def _load_calibration(self) -> None:
        """Load previously calibrated coefficients from disk."""
        if self._model_dir is None:
            return
        path = self._model_dir / "cost_calibration_current.joblib"
        if not path.exists():
            return
        try:
            import joblib

            loaded = joblib.load(path)
            if isinstance(loaded, dict):
                for seg in ("micro", "small", "mid", "large"):
                    if seg in loaded and np.isfinite(loaded[seg]):
                        self._coefficients[seg] = float(loaded[seg])
                logger.info(
                    "CostCalibrator: loaded calibration from %s: %s",
                    path,
                    self._coefficients,
                )
        except ImportError:
            logger.debug("joblib not available; using default coefficients")
        except Exception as e:
            logger.warning(
                "CostCalibrator: load failed, using defaults: %s", e
            )

    def reset_history(self) -> None:
        """Clear all recorded trade history (coefficients are preserved)."""
        self._trade_history.clear()
        self._total_trades = 0

    def __repr__(self) -> str:
        return (
            f"CostCalibrator(coeffs={self._coefficients}, "
            f"trades={self._total_trades})"
        )
