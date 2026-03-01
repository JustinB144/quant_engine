"""
Cost model calibrator for per-market-cap-segment impact coefficients.

Spec 06 T4: Adjusts EXEC_IMPACT_COEFF_BPS per market cap segment
(micro, small, mid, large) based on realized impact from historical
trades.  Calibrated coefficients are stored to disk (joblib) and loaded
for production use.

SPEC-E04: Calibration feedback loop — compares predicted execution costs
against actual paper-trade fills, computes cost surprise by regime bucket,
and applies EMA-smoothed coefficient updates on a configurable cadence.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Maximum history sizes to prevent unbounded memory growth.
MAX_CALIBRATION_HISTORY = 5000
MAX_FEEDBACK_HISTORY = 5000


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
        feedback_path: Optional[Path] = None,
        feedback_interval_days: int = 30,
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

        # SPEC-E04: Feedback loop state
        self._feedback_path = Path(feedback_path) if feedback_path else None
        self._feedback_interval_days = int(max(1, feedback_interval_days))
        self._actual_fills: List[Dict] = []
        self._last_feedback_recalibration: Optional[str] = None

        # Try to load persisted coefficients
        self._load_calibration()

        # SPEC-E04: Load persisted feedback history
        self._load_feedback_history()

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
        effective_spread_bps: Optional[float] = None,
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
            Total realized slippage in basis points (spread + market impact).
        effective_spread_bps : float, optional
            The actual spread including all multipliers (structural, event, etc.).
            If None, uses base spread (backward-compatible).
        """
        if participation_rate <= 0 or not np.isfinite(participation_rate):
            return
        if not np.isfinite(realized_cost_bps):
            return
        assert realized_cost_bps >= 0, (
            f"Expected non-negative total slippage, got {realized_cost_bps:.2f} bps"
        )

        segment = self.get_marketcap_segment(market_cap)
        self._trade_history[segment].append({
            "symbol": symbol,
            "participation_rate": float(participation_rate),
            "realized_cost_bps": float(realized_cost_bps),
            "effective_spread_bps": float(effective_spread_bps) if effective_spread_bps is not None else None,
        })
        self._total_trades += 1

        # Prune per-segment history to prevent unbounded growth
        if len(self._trade_history[segment]) > MAX_CALIBRATION_HISTORY:
            self._trade_history[segment] = self._trade_history[segment][-MAX_CALIBRATION_HISTORY:]

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
                # Use effective spread (including multipliers) when available,
                # otherwise fall back to base spread. This prevents
                # over-attributing to impact during stressed periods.
                eff_spread = t.get("effective_spread_bps")
                spread = eff_spread if eff_spread is not None else 0.5 * self._spread_bps
                net_impact = max(0.0, cost - spread)
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

    # ── SPEC-E04: Calibration Feedback Loop ─────────────────────────

    def record_actual_fill(
        self,
        symbol: str,
        market_cap: float,
        predicted_cost_bps: float,
        actual_cost_bps: float,
        participation_rate: float,
        regime: int = 2,
        fill_timestamp: Optional[str] = None,
    ) -> None:
        """Record an actual paper-trade fill for predicted-vs-actual comparison.

        Parameters
        ----------
        symbol : str
            Ticker or PERMNO identifier.
        market_cap : float
            Market capitalization in USD at time of fill.
        predicted_cost_bps : float
            The execution cost (spread + impact) that the model predicted
            at order time, in basis points.
        actual_cost_bps : float
            The realized execution cost observed from the paper-trade fill,
            in basis points.
        participation_rate : float
            Fraction of daily volume that was executed.
        regime : int
            Regime code at time of fill (0=trending_bull, 1=trending_bear,
            2=mean_reverting, 3=high_volatility).
        fill_timestamp : str, optional
            ISO-format timestamp of the fill.  Defaults to current UTC time.
        """
        if participation_rate <= 0 or not np.isfinite(participation_rate):
            return
        if not np.isfinite(predicted_cost_bps) or not np.isfinite(actual_cost_bps):
            return

        segment = self.get_marketcap_segment(market_cap)
        ts = fill_timestamp or datetime.now(timezone.utc).isoformat()

        self._actual_fills.append({
            "symbol": str(symbol),
            "segment": segment,
            "market_cap": float(market_cap),
            "predicted_cost_bps": float(predicted_cost_bps),
            "actual_cost_bps": float(actual_cost_bps),
            "participation_rate": float(participation_rate),
            "regime": int(regime),
            "timestamp": ts,
        })

        # Prune feedback history to prevent unbounded growth
        if len(self._actual_fills) > MAX_FEEDBACK_HISTORY:
            self._actual_fills = self._actual_fills[-MAX_FEEDBACK_HISTORY:]

    def compute_cost_surprise(self) -> Dict[str, Dict]:
        """Compute cost surprise distribution: predicted minus actual, by regime.

        Cost surprise is defined as ``predicted - actual`` so that positive
        values indicate the model over-estimated costs (conservative) and
        negative values indicate under-estimation (optimistic).

        Results are bucketed by regime code, with each bucket containing:
        - ``count``: number of fills in the bucket
        - ``mean_surprise_bps``: average surprise
        - ``median_surprise_bps``: median surprise (robust to outliers)
        - ``std_surprise_bps``: standard deviation
        - ``pct_overestimated``: fraction of fills where model was conservative

        Returns
        -------
        dict
            Keyed by regime code (as string), plus an ``"_all"`` aggregate.
            Empty dict if no actual fills recorded.
        """
        if not self._actual_fills:
            return {}

        results: Dict[str, Dict] = {}

        # Group fills by regime
        regime_buckets: Dict[int, List[float]] = defaultdict(list)
        all_surprises: List[float] = []

        for fill in self._actual_fills:
            surprise = fill["predicted_cost_bps"] - fill["actual_cost_bps"]
            regime_buckets[fill["regime"]].append(surprise)
            all_surprises.append(surprise)

        # Per-regime statistics
        for regime_code, surprises in sorted(regime_buckets.items()):
            arr = np.array(surprises, dtype=float)
            results[str(regime_code)] = {
                "count": len(surprises),
                "mean_surprise_bps": float(np.mean(arr)),
                "median_surprise_bps": float(np.median(arr)),
                "std_surprise_bps": float(np.std(arr)) if len(arr) > 1 else 0.0,
                "pct_overestimated": float(np.mean(arr > 0)),
            }

        # Aggregate across all regimes
        all_arr = np.array(all_surprises, dtype=float)
        results["_all"] = {
            "count": len(all_surprises),
            "mean_surprise_bps": float(np.mean(all_arr)),
            "median_surprise_bps": float(np.median(all_arr)),
            "std_surprise_bps": float(np.std(all_arr)) if len(all_arr) > 1 else 0.0,
            "pct_overestimated": float(np.mean(all_arr > 0)),
        }

        return results

    def compute_cost_surprise_by_segment(self) -> Dict[str, Dict]:
        """Compute cost surprise distribution by market-cap segment.

        Same structure as :meth:`compute_cost_surprise` but bucketed by
        segment (micro, small, mid, large) instead of regime.

        Returns
        -------
        dict
            Keyed by segment name, plus ``"_all"`` aggregate.
        """
        if not self._actual_fills:
            return {}

        results: Dict[str, Dict] = {}
        segment_buckets: Dict[str, List[float]] = defaultdict(list)

        for fill in self._actual_fills:
            surprise = fill["predicted_cost_bps"] - fill["actual_cost_bps"]
            segment_buckets[fill["segment"]].append(surprise)

        for seg_name in ("micro", "small", "mid", "large"):
            surprises = segment_buckets.get(seg_name, [])
            if not surprises:
                continue
            arr = np.array(surprises, dtype=float)
            results[seg_name] = {
                "count": len(surprises),
                "mean_surprise_bps": float(np.mean(arr)),
                "median_surprise_bps": float(np.median(arr)),
                "std_surprise_bps": float(np.std(arr)) if len(arr) > 1 else 0.0,
                "pct_overestimated": float(np.mean(arr > 0)),
            }

        # Aggregate across all segments
        all_surprises = [s for bucket in segment_buckets.values() for s in bucket]
        if all_surprises:
            all_arr = np.array(all_surprises, dtype=float)
            results["_all"] = {
                "count": len(all_surprises),
                "mean_surprise_bps": float(np.mean(all_arr)),
                "median_surprise_bps": float(np.median(all_arr)),
                "std_surprise_bps": float(np.std(all_arr)) if len(all_arr) > 1 else 0.0,
                "pct_overestimated": float(np.mean(all_arr > 0)),
            }

        return results

    def run_feedback_recalibration(
        self,
        force: bool = False,
    ) -> Dict[str, Dict]:
        """Run the feedback-driven recalibration loop (SPEC-E04).

        Compares predicted vs actual costs from recorded paper-trade fills,
        computes the cost surprise distribution per market-cap segment, and
        updates impact coefficients using EMA smoothing.

        The recalibration is skipped if fewer than ``_min_total`` fills
        have been recorded or if the configured interval has not elapsed
        since the last recalibration (unless ``force=True``).

        Parameters
        ----------
        force : bool
            If True, skip the interval check and recalibrate immediately.

        Returns
        -------
        dict
            Per-segment recalibration results.  Empty if skipped.
        """
        if not self._actual_fills:
            logger.info(
                "CostCalibrator feedback: no actual fills recorded, skipping"
            )
            return {}

        # Check interval
        if not force and self._last_feedback_recalibration is not None:
            try:
                last_dt = datetime.fromisoformat(
                    self._last_feedback_recalibration
                )
                # Ensure timezone-aware comparison (Python 3.12+ requires it)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_dt).days
                if elapsed < self._feedback_interval_days:
                    logger.debug(
                        "CostCalibrator feedback: %d days since last "
                        "recalibration (interval=%d), skipping",
                        elapsed,
                        self._feedback_interval_days,
                    )
                    return {}
            except (ValueError, TypeError):
                pass  # Invalid timestamp, proceed with recalibration

        # Check minimum fill count
        if len(self._actual_fills) < self._min_total:
            logger.info(
                "CostCalibrator feedback: insufficient fills (%d < %d)",
                len(self._actual_fills),
                self._min_total,
            )
            return {}

        # Group fills by segment and compute adjustment
        segment_fills: Dict[str, List[Dict]] = defaultdict(list)
        for fill in self._actual_fills:
            segment_fills[fill["segment"]].append(fill)

        results: Dict[str, Dict] = {}

        for segment in ("micro", "small", "mid", "large"):
            fills = segment_fills.get(segment, [])
            if len(fills) < self._min_segment:
                logger.debug(
                    "CostCalibrator feedback: skipping %s (%d < %d fills)",
                    segment,
                    len(fills),
                    self._min_segment,
                )
                continue

            # Compute implied coefficient from actual costs
            implied_coefficients: List[float] = []
            for fill in fills:
                part = fill["participation_rate"]
                actual_cost = fill["actual_cost_bps"]
                sqrt_part = np.sqrt(part)
                if sqrt_part < 1e-9:
                    continue
                net_impact = max(0.0, actual_cost - 0.5 * self._spread_bps)
                implied = net_impact / sqrt_part
                if np.isfinite(implied) and implied > 0:
                    implied_coefficients.append(implied)

            if not implied_coefficients:
                continue

            observed = float(np.median(implied_coefficients))
            old_coeff = self._coefficients.get(
                segment, self._default.get(segment, 25.0)
            )
            smoothed = (
                self._smoothing * observed
                + (1.0 - self._smoothing) * old_coeff
            )
            smoothed = float(np.clip(smoothed, 5.0, 100.0))
            self._coefficients[segment] = smoothed

            # Compute surprise stats for this segment
            surprises = [
                f["predicted_cost_bps"] - f["actual_cost_bps"]
                for f in fills
            ]
            surprise_arr = np.array(surprises, dtype=float)

            results[segment] = {
                "fills_count": len(fills),
                "old_coeff": old_coeff,
                "observed_coeff": observed,
                "smoothed_coeff": smoothed,
                "mean_surprise_bps": float(np.mean(surprise_arr)),
                "median_surprise_bps": float(np.median(surprise_arr)),
            }
            logger.info(
                "CostCalibrator feedback: %s calibrated: %.1f → %.1f "
                "(observed=%.1f, mean_surprise=%.2f bps, %d fills)",
                segment,
                old_coeff,
                smoothed,
                observed,
                float(np.mean(surprise_arr)),
                len(fills),
            )

        # Update recalibration timestamp and persist
        self._last_feedback_recalibration = datetime.now(
            timezone.utc
        ).isoformat()

        self._save_calibration()
        self._save_feedback_history()

        return results

    @property
    def actual_fills(self) -> List[Dict]:
        """Recorded actual fill history (copy)."""
        return list(self._actual_fills)

    @property
    def feedback_fill_count(self) -> int:
        """Number of actual fills recorded for feedback."""
        return len(self._actual_fills)

    @property
    def last_feedback_recalibration(self) -> Optional[str]:
        """ISO timestamp of the last feedback recalibration, or None."""
        return self._last_feedback_recalibration

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

    # ── SPEC-E04: Feedback History Persistence ─────────────────────

    def _save_feedback_history(self) -> None:
        """Persist actual fill history and recalibration timestamp to disk."""
        if self._feedback_path is None:
            return
        try:
            self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "actual_fills": self._actual_fills,
                "last_feedback_recalibration": self._last_feedback_recalibration,
            }
            with open(self._feedback_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(
                "CostCalibrator: saved %d feedback fills to %s",
                len(self._actual_fills),
                self._feedback_path,
            )
        except Exception as e:
            logger.warning("CostCalibrator: feedback save failed: %s", e)

    def _load_feedback_history(self) -> None:
        """Load previously persisted feedback fill history from disk."""
        if self._feedback_path is None:
            return
        if not self._feedback_path.exists():
            return
        try:
            with open(self._feedback_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                fills = data.get("actual_fills", [])
                if isinstance(fills, list):
                    self._actual_fills = fills
                last_ts = data.get("last_feedback_recalibration")
                if isinstance(last_ts, str) and last_ts:
                    self._last_feedback_recalibration = last_ts
                logger.info(
                    "CostCalibrator: loaded %d feedback fills from %s",
                    len(self._actual_fills),
                    self._feedback_path,
                )
        except Exception as e:
            logger.warning(
                "CostCalibrator: feedback load failed: %s", e
            )

    def reset_history(self) -> None:
        """Clear all recorded trade history (coefficients are preserved)."""
        self._trade_history.clear()
        self._total_trades = 0

    def reset_feedback_history(self) -> None:
        """Clear all recorded feedback fills (coefficients are preserved)."""
        self._actual_fills.clear()
        self._last_feedback_recalibration = None

    def __repr__(self) -> str:
        return (
            f"CostCalibrator(coeffs={self._coefficients}, "
            f"trades={self._total_trades}, "
            f"feedback_fills={len(self._actual_fills)})"
        )
