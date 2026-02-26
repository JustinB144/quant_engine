"""
Backtester — converts model predictions into simulated trades.

Two modes:
    1. Simple mode (default): Fixed holding period, fixed position size
    2. Risk-managed mode: Dynamic sizing, stops, drawdown controls, portfolio limits

Strategy:
    - Go LONG when predicted_return > threshold AND confidence > min_confidence
    - Enter at NEXT BAR OPEN (no look-ahead on signal bar close)
    - Exit at holding period end OR stop-loss trigger (risk-managed mode)
    - Enforce max concurrent positions ACROSS all tickers
    - Slippage modeled as fraction of intraday range
    - Flat transaction cost per round-trip
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

import logging

from ..config import (
    TRANSACTION_COST_BPS, ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD,
    MAX_POSITIONS, POSITION_SIZE_PCT,
    BACKTEST_ASSUMED_CAPITAL_USD, EXEC_SPREAD_BPS, EXEC_MAX_PARTICIPATION,
    EXEC_IMPACT_COEFF_BPS, EXEC_MIN_FILL_RATIO, REGIME_RISK_MULTIPLIER,
    EXEC_DYNAMIC_COSTS, EXEC_DOLLAR_VOLUME_REF_USD, EXEC_VOL_REF,
    EXEC_VOL_SPREAD_BETA, EXEC_GAP_SPREAD_BETA, EXEC_RANGE_SPREAD_BETA,
    EXEC_VOL_IMPACT_BETA,
    REQUIRE_PERMNO,
    ALMGREN_CHRISS_ENABLED, ALMGREN_CHRISS_ADV_THRESHOLD,
    ALMGREN_CHRISS_RISK_AVERSION,
    MAX_ANNUALIZED_TURNOVER,
    ALMGREN_CHRISS_FALLBACK_VOL, REGIME_TRADE_POLICY,
    # Spec 06: structural state-aware costs
    EXEC_STRUCTURAL_STRESS_ENABLED,
    EXEC_BREAK_PROB_COST_MULT,
    EXEC_STRUCTURE_UNCERTAINTY_COST_MULT,
    EXEC_DRIFT_SCORE_COST_REDUCTION,
    EXEC_SYSTEMIC_STRESS_COST_MULT,
    EXEC_EXIT_URGENCY_COST_LIMIT_MULT,
    EXEC_ENTRY_URGENCY_COST_LIMIT_MULT,
    EXEC_STRESS_PULLBACK_MIN_SIZE,
    EXEC_NO_TRADE_STRESS_THRESHOLD,
    EXEC_VOLUME_TREND_ENABLED,
    ADV_LOOKBACK_DAYS,
    ADV_EMA_SPAN,
    EXEC_LOW_VOLUME_COST_MULT,
    # Spec 06: cost calibration per market cap segment
    EXEC_CALIBRATION_ENABLED,
    EXEC_CALIBRATION_MIN_TRADES,
    EXEC_CALIBRATION_MIN_SEGMENT_TRADES,
    EXEC_CALIBRATION_SMOOTHING,
    EXEC_COST_IMPACT_COEFF_BY_MARKETCAP,
    EXEC_MARKETCAP_THRESHOLDS,
    # SPEC-E01: edge-after-costs trade gating
    EDGE_COST_GATE_ENABLED,
    EDGE_COST_BUFFER_BASE_BPS,
)

logger = logging.getLogger(__name__)
from .execution import ExecutionModel
from .cost_calibrator import CostCalibrator
from ..regime.shock_vector import compute_shock_vectors, ShockVector
from ..regime.uncertainty_gate import UncertaintyGate

_PERMNO_RE = re.compile(r"^\d{1,10}$")


@dataclass
class Trade:
    """Trade record produced by the backtester for one simulated position lifecycle."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    predicted_return: float
    actual_return: float
    net_return: float
    regime: int
    confidence: float
    holding_days: int
    position_size: float = 0.05     # fraction of capital
    exit_reason: str = "holding"    # holding, atr_stop, trailing_stop, time_stop, regime_change, hard_stop, target_hit, circuit_breaker, end_of_data
    fill_ratio: float = 1.0
    entry_impact_bps: float = 0.0
    exit_impact_bps: float = 0.0
    entry_reference_price: float = 0.0   # pre-slippage Open on entry bar
    exit_reference_price: float = 0.0    # pre-slippage Close on exit bar
    market_cap_segment: str = ""         # SPEC-W02: micro/small/mid/large


@dataclass
class BacktestResult:
    """Aggregate backtest outputs including metrics, curves, and trade history."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_holding_days: float
    trades_per_year: float
    returns_series: pd.Series = field(repr=False, default=None)
    equity_curve: pd.Series = field(repr=False, default=None)
    daily_equity: pd.Series = field(repr=False, default=None)
    trades: List[Trade] = field(repr=False, default_factory=list)
    regime_breakdown: Dict = field(default_factory=dict)
    regime_performance: Dict = field(default_factory=dict)
    tca_report: Dict = field(default_factory=dict)
    risk_report: Optional[object] = field(repr=False, default=None)
    drawdown_summary: Optional[Dict] = field(repr=False, default=None)
    exit_reason_breakdown: Dict = field(default_factory=dict)
    # Turnover tracking
    total_turnover: float = 0.0
    annualized_turnover: float = 0.0
    avg_daily_turnover: float = 0.0
    turnover_history: List[float] = field(repr=False, default_factory=list)
    # Truth Layer: null model baselines and cost stress results
    null_baselines: Optional[object] = field(repr=False, default=None)
    cost_stress_result: Optional[object] = field(repr=False, default=None)

    def summarize_vs_null(self) -> Dict[str, float]:
        """Compare strategy Sharpe/return vs null baselines.

        Returns an empty dict if null baselines have not been computed.
        """
        if self.null_baselines is None:
            return {}

        nb = self.null_baselines
        return {
            "sharpe_vs_random": self.sharpe_ratio - nb.random.sharpe_ratio,
            "sharpe_vs_zero": self.sharpe_ratio - nb.zero.sharpe_ratio,
            "sharpe_vs_momentum": self.sharpe_ratio - nb.momentum.sharpe_ratio,
            "return_vs_random": self.total_return - nb.random.total_return,
            "return_vs_zero": self.total_return - nb.zero.total_return,
            "return_vs_momentum": self.total_return - nb.momentum.total_return,
        }


class Backtester:
    """
    Simulates trading from model predictions.

    Key design principles (anti-leakage):
        - Entry at next-bar OPEN after signal, not signal-bar close
        - Exit at exit-bar CLOSE (can be observed in real time)
        - Max positions enforced across all tickers simultaneously
        - Slippage modeled proportional to intraday range

    Risk-managed mode adds:
        - Dynamic position sizing (Kelly/vol-scaled/ATR blend)
        - Drawdown circuit breakers (warning/caution/critical)
        - Stop-loss management (ATR, trailing, time, regime-change)
        - Portfolio risk checks (sector, correlation, exposure)
    """

    def __init__(
        self,
        entry_threshold: float = ENTRY_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        transaction_cost_bps: float = TRANSACTION_COST_BPS,
        holding_days: int = 10,
        max_positions: int = MAX_POSITIONS,
        position_size_pct: float = POSITION_SIZE_PCT,
        slippage_pct: float = 0.0005,  # 5 bps slippage per side
        use_risk_management: bool = False,
        assumed_capital_usd: float = BACKTEST_ASSUMED_CAPITAL_USD,
        spread_bps: float = EXEC_SPREAD_BPS,
        max_participation_rate: float = EXEC_MAX_PARTICIPATION,
        impact_coefficient_bps: float = EXEC_IMPACT_COEFF_BPS,
        min_fill_ratio: float = EXEC_MIN_FILL_RATIO,
    ):
        """Initialize Backtester."""
        # Truth Layer: validate execution contract preconditions
        from ..validation.preconditions import enforce_preconditions
        enforce_preconditions()

        self.entry_threshold = entry_threshold
        self.confidence_threshold = confidence_threshold
        self.tx_cost = transaction_cost_bps / 10000
        self.holding_days = holding_days
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.slippage_pct = slippage_pct
        self.use_risk = use_risk_management
        self.assumed_capital_usd = float(assumed_capital_usd)
        self.execution_model = ExecutionModel(
            spread_bps=spread_bps,
            max_participation_rate=max_participation_rate,
            impact_coefficient_bps=impact_coefficient_bps,
            min_fill_ratio=min_fill_ratio,
            dynamic_costs=EXEC_DYNAMIC_COSTS,
            dollar_volume_ref_usd=EXEC_DOLLAR_VOLUME_REF_USD,
            vol_ref=EXEC_VOL_REF,
            vol_spread_beta=EXEC_VOL_SPREAD_BETA,
            gap_spread_beta=EXEC_GAP_SPREAD_BETA,
            range_spread_beta=EXEC_RANGE_SPREAD_BETA,
            vol_impact_beta=EXEC_VOL_IMPACT_BETA,
            # Spec 06: structural state-aware costs
            structural_stress_enabled=EXEC_STRUCTURAL_STRESS_ENABLED,
            break_prob_cost_mult=EXEC_BREAK_PROB_COST_MULT,
            structure_uncertainty_cost_mult=EXEC_STRUCTURE_UNCERTAINTY_COST_MULT,
            drift_score_cost_reduction=EXEC_DRIFT_SCORE_COST_REDUCTION,
            systemic_stress_cost_mult=EXEC_SYSTEMIC_STRESS_COST_MULT,
            exit_urgency_cost_limit_mult=EXEC_EXIT_URGENCY_COST_LIMIT_MULT,
            entry_urgency_cost_limit_mult=EXEC_ENTRY_URGENCY_COST_LIMIT_MULT,
            stress_pullback_min_size=EXEC_STRESS_PULLBACK_MIN_SIZE,
            no_trade_stress_threshold=EXEC_NO_TRADE_STRESS_THRESHOLD,
            volume_trend_enabled=EXEC_VOLUME_TREND_ENABLED,
        )

        # Set base transaction cost for urgency limit comparison
        self.execution_model.set_base_transaction_cost_bps(
            float(transaction_cost_bps)
        )

        # Spec 06: ADV tracker for volume trend analysis
        from .adv_tracker import ADVTracker
        self._adv_tracker = ADVTracker(
            lookback_days=ADV_LOOKBACK_DAYS,
            ema_span=ADV_EMA_SPAN,
            low_volume_cost_mult=EXEC_LOW_VOLUME_COST_MULT,
        )

        # SPEC-W02: Per-market-cap-segment cost calibrator.
        # Provides differentiated impact coefficients (micro ~40 bps,
        # large ~15 bps) instead of the flat 25 bps default.
        from pathlib import Path
        if EXEC_CALIBRATION_ENABLED:
            self._cost_calibrator = CostCalibrator(
                default_coefficients=EXEC_COST_IMPACT_COEFF_BY_MARKETCAP,
                marketcap_thresholds=EXEC_MARKETCAP_THRESHOLDS,
                min_total_trades=EXEC_CALIBRATION_MIN_TRADES,
                min_segment_trades=EXEC_CALIBRATION_MIN_SEGMENT_TRADES,
                smoothing=EXEC_CALIBRATION_SMOOTHING,
                model_dir=Path("trained_models"),
            )
        else:
            self._cost_calibrator = None

        # When dynamic execution model is active, costs are already embedded in
        # fill prices from _simulate_entry/_simulate_exit.  Setting tx_cost to
        # zero avoids double-counting.  TRANSACTION_COST_BPS is for legacy
        # flat-cost mode only (EXEC_DYNAMIC_COSTS=False).
        if EXEC_DYNAMIC_COSTS:
            self.tx_cost = 0.0

        # SPEC-W01: Pre-computed shock vectors for structural state
        # conditioning.  Populated in run() before trade processing.
        # Keyed by (ticker_str, bar_date) → ShockVector.
        self._shock_vectors: Dict[tuple, ShockVector] = {}

        # SPEC-W03: Uncertainty gate scales position sizes down when
        # regime detection entropy is high (regime transitions).
        self._uncertainty_gate = UncertaintyGate()

        # Risk components (lazy-initialized if needed)
        self._position_sizer = None
        self._drawdown_ctrl = None
        self._stop_loss_mgr = None
        self._portfolio_risk = None
        self._risk_metrics = None

    def _init_risk_components(self):
        """Initialize risk management components."""
        from ..risk.position_sizer import PositionSizer
        from ..risk.drawdown import DrawdownController
        from ..risk.stop_loss import StopLossManager
        from ..risk.portfolio_risk import PortfolioRiskManager
        from ..risk.metrics import RiskMetrics

        self._position_sizer = PositionSizer(
            max_position_pct=self.position_size_pct * 2,  # Allow up to 2x base
            min_position_pct=self.position_size_pct * 0.25,
        )
        self._drawdown_ctrl = DrawdownController()
        self._stop_loss_mgr = StopLossManager(
            max_holding_days=self.holding_days * 3,  # Allow longer holds with stops
            atr_stop_multiplier=2.0,
            trailing_atr_multiplier=1.5,
        )
        self._portfolio_risk = PortfolioRiskManager()
        self._risk_metrics = RiskMetrics()

    def _almgren_chriss_cost_bps(
        self,
        shares: float,
        reference_price: float,
        daily_volume: float,
        daily_volatility: float,
        n_intervals: int = 10,
    ) -> Optional[float]:
        """Compute Almgren-Chriss execution cost in basis points.

        Returns ``None`` when the AC module is unavailable or raises an error,
        allowing callers to fall back to the simple impact model.
        """
        try:
            from .optimal_execution import (
                almgren_chriss_trajectory,
                estimate_execution_cost,
            )

            trajectory = almgren_chriss_trajectory(
                total_shares=int(abs(shares)),
                n_intervals=n_intervals,
                daily_volume=daily_volume,
                daily_volatility=daily_volatility,
            )
            cost_info = estimate_execution_cost(
                trajectory=trajectory,
                reference_price=reference_price,
                daily_volume=daily_volume,
                daily_volatility=daily_volatility,
            )
            cost_bps = float(cost_info["cost_bps"])
            if np.isfinite(cost_bps) and cost_bps >= 0:
                return cost_bps
            return None
        except (ImportError, ValueError, RuntimeError):
            return None

    def _simulate_entry(
        self,
        ohlcv: pd.DataFrame,
        entry_idx: int,
        position_size: float,
    ) -> Optional[dict]:
        """
        Simulate entry execution with participation and impact constraints.

        For positions exceeding ``ALMGREN_CHRISS_ADV_THRESHOLD`` of daily
        volume the Almgren-Chriss optimal-execution cost model replaces the
        simple square-root impact model.  Small positions keep the fast path.
        """
        ref_price = float(ohlcv["Open"].iloc[entry_idx])
        vol = float(ohlcv["Volume"].iloc[entry_idx]) if "Volume" in ohlcv.columns else 0.0
        context = self._execution_context(ohlcv=ohlcv, bar_idx=entry_idx)
        desired_notional = self.assumed_capital_usd * max(0.0, float(position_size))

        # Spec 06: update ADV tracker and get volume trend
        ticker_id = ohlcv.attrs.get("ticker", ohlcv.attrs.get("permno", ""))
        if vol > 0 and ticker_id:
            self._adv_tracker.update(ticker_id, vol)
        volume_trend = (
            self._adv_tracker.get_volume_trend(ticker_id)
            if ticker_id and vol > 0 else None
        )

        # SPEC-W01: look up pre-computed shock vector for this bar
        bar_date = ohlcv.index[entry_idx]
        shock = self._shock_vectors.get((str(ticker_id), bar_date))

        # SPEC-W02: get per-segment calibrated impact coefficient
        impact_coeff_override = None
        estimated_market_cap = 0.0
        if self._cost_calibrator is not None:
            estimated_market_cap = self._estimate_market_cap(ohlcv, entry_idx)
            if estimated_market_cap > 0:
                impact_coeff_override = self._cost_calibrator.get_impact_coeff(
                    estimated_market_cap,
                )

        fill = self.execution_model.simulate(
            side="buy",
            reference_price=ref_price,
            daily_volume=vol,
            desired_notional_usd=desired_notional,
            force_full=False,
            realized_vol=context["realized_vol"],
            overnight_gap=context["overnight_gap"],
            intraday_range=context["intraday_range"],
            urgency_type="entry",
            volume_trend=volume_trend,
            # SPEC-W01: structural state from shock vector
            break_probability=shock.bocpd_changepoint_prob if shock else None,
            structure_uncertainty=shock.hmm_uncertainty if shock else None,
            drift_score=shock.structural_features.get("drift_score") if shock else None,
            systemic_stress=shock.structural_features.get("systemic_stress") if shock else None,
            # SPEC-W02: per-segment impact coefficient
            impact_coeff_override=impact_coeff_override,
        )
        if fill.fill_ratio <= 0:
            return None
        actual_notional = desired_notional * fill.fill_ratio
        shares = actual_notional / max(1e-9, fill.fill_price)

        # ── Almgren-Chriss upgrade for large positions ──
        impact_bps = float(fill.impact_bps)
        fill_price = float(fill.fill_price)

        if ALMGREN_CHRISS_ENABLED and vol > 0:
            participation_rate = shares / vol
            if participation_rate > ALMGREN_CHRISS_ADV_THRESHOLD:
                daily_vol = context["realized_vol"]
                if daily_vol is None or not np.isfinite(daily_vol):
                    daily_vol = ALMGREN_CHRISS_FALLBACK_VOL  # fallback annualised vol
                # Convert annualised vol to daily
                daily_volatility = daily_vol / np.sqrt(252.0)
                ac_cost_bps = self._almgren_chriss_cost_bps(
                    shares=shares,
                    reference_price=ref_price,
                    daily_volume=vol,
                    daily_volatility=daily_volatility,
                )
                if ac_cost_bps is not None:
                    impact_bps = ac_cost_bps
                    # Recompute fill price using AC cost (half-spread already
                    # included in the simple model; keep spread, replace impact)
                    half_spread_bps = 0.5 * float(fill.spread_bps)
                    total_bps = half_spread_bps + ac_cost_bps
                    fill_price = ref_price * (1.0 + total_bps / 10_000.0)
                    shares = actual_notional / max(1e-9, fill_price)

        # SPEC-W02: record realized cost for online recalibration
        if self._cost_calibrator is not None and fill.participation_rate > 0:
            self._cost_calibrator.record_trade(
                symbol=str(ticker_id),
                market_cap=estimated_market_cap,
                participation_rate=fill.participation_rate,
                realized_cost_bps=float(impact_bps),
            )

        # SPEC-W02: include market cap segment in return dict for TCA
        segment = ""
        if self._cost_calibrator is not None and estimated_market_cap > 0:
            segment = self._cost_calibrator.get_marketcap_segment(
                estimated_market_cap,
            )

        return {
            "entry_price": float(fill_price),
            "entry_impact_bps": float(impact_bps),
            "fill_ratio": float(fill.fill_ratio),
            "shares": float(shares),
            "position_size": float(position_size * fill.fill_ratio),
            "market_cap_segment": segment,
        }

    def _simulate_exit(
        self,
        ohlcv: pd.DataFrame,
        exit_idx: int,
        shares: float,
        force_full: bool = True,
    ) -> dict:
        """Simulate exit execution, upgrading to Almgren-Chriss for large positions."""
        ref_price = float(ohlcv["Close"].iloc[exit_idx])
        vol = float(ohlcv["Volume"].iloc[exit_idx]) if "Volume" in ohlcv.columns else 0.0
        context = self._execution_context(ohlcv=ohlcv, bar_idx=exit_idx)
        desired_notional = max(0.0, float(shares) * ref_price)

        # Spec 06: update ADV tracker and get volume trend
        ticker_id = ohlcv.attrs.get("ticker", ohlcv.attrs.get("permno", ""))
        if vol > 0 and ticker_id:
            self._adv_tracker.update(ticker_id, vol)
        volume_trend = (
            self._adv_tracker.get_volume_trend(ticker_id)
            if ticker_id and vol > 0 else None
        )

        # SPEC-W01: look up pre-computed shock vector for this bar
        bar_date = ohlcv.index[exit_idx]
        shock = self._shock_vectors.get((str(ticker_id), bar_date))

        # SPEC-W02: get per-segment calibrated impact coefficient
        impact_coeff_override = None
        estimated_market_cap = 0.0
        if self._cost_calibrator is not None:
            estimated_market_cap = self._estimate_market_cap(ohlcv, exit_idx)
            if estimated_market_cap > 0:
                impact_coeff_override = self._cost_calibrator.get_impact_coeff(
                    estimated_market_cap,
                )

        fill = self.execution_model.simulate(
            side="sell",
            reference_price=ref_price,
            daily_volume=vol,
            desired_notional_usd=desired_notional,
            force_full=force_full,
            realized_vol=context["realized_vol"],
            overnight_gap=context["overnight_gap"],
            intraday_range=context["intraday_range"],
            urgency_type="exit",
            volume_trend=volume_trend,
            # SPEC-W01: structural state from shock vector
            break_probability=shock.bocpd_changepoint_prob if shock else None,
            structure_uncertainty=shock.hmm_uncertainty if shock else None,
            drift_score=shock.structural_features.get("drift_score") if shock else None,
            systemic_stress=shock.structural_features.get("systemic_stress") if shock else None,
            # SPEC-W02: per-segment impact coefficient
            impact_coeff_override=impact_coeff_override,
        )
        if fill.fill_ratio <= 0:
            # Emergency fallback: force full exit with legacy slippage.
            fallback_price = ref_price * (1 - self.slippage_pct)
            return {
                "exit_price": float(fallback_price),
                "exit_impact_bps": 0.0,
                "exit_fill_ratio": 1.0,
            }

        # ── Almgren-Chriss upgrade for large positions ──
        impact_bps = float(fill.impact_bps)
        fill_price = float(fill.fill_price)

        if ALMGREN_CHRISS_ENABLED and vol > 0:
            participation_rate = abs(float(shares)) / vol
            if participation_rate > ALMGREN_CHRISS_ADV_THRESHOLD:
                daily_vol = context["realized_vol"]
                if daily_vol is None or not np.isfinite(daily_vol):
                    daily_vol = ALMGREN_CHRISS_FALLBACK_VOL  # fallback annualised vol
                daily_volatility = daily_vol / np.sqrt(252.0)
                ac_cost_bps = self._almgren_chriss_cost_bps(
                    shares=shares,
                    reference_price=ref_price,
                    daily_volume=vol,
                    daily_volatility=daily_volatility,
                )
                if ac_cost_bps is not None:
                    impact_bps = ac_cost_bps
                    half_spread_bps = 0.5 * float(fill.spread_bps)
                    total_bps = half_spread_bps + ac_cost_bps
                    fill_price = ref_price * (1.0 - total_bps / 10_000.0)

        # SPEC-W02: record realized cost for online recalibration
        if self._cost_calibrator is not None and fill.participation_rate > 0:
            self._cost_calibrator.record_trade(
                symbol=str(ticker_id),
                market_cap=estimated_market_cap,
                participation_rate=fill.participation_rate,
                realized_cost_bps=float(impact_bps),
            )

        return {
            "exit_price": float(fill_price),
            "exit_impact_bps": float(impact_bps),
            "exit_fill_ratio": float(fill.fill_ratio),
        }

    @staticmethod
    def _execution_context(ohlcv: pd.DataFrame, bar_idx: int) -> dict:
        """
        Build local microstructure context for conditional execution costs.
        """
        close = ohlcv["Close"]
        open_ = ohlcv["Open"]
        high = ohlcv["High"]
        low = ohlcv["Low"]

        if bar_idx > 0:
            prev_close = float(close.iloc[bar_idx - 1])
            overnight_gap = float(open_.iloc[bar_idx] / max(1e-12, prev_close) - 1.0)
        else:
            overnight_gap = 0.0

        intraday_range = float(
            (high.iloc[bar_idx] - low.iloc[bar_idx]) / max(1e-12, close.iloc[bar_idx]),
        )

        start = max(0, bar_idx - 20)
        rets = close.iloc[start : bar_idx + 1].pct_change().dropna()
        if len(rets) >= 5:
            realized_vol = float(rets.std() * np.sqrt(252.0))
        else:
            realized_vol = np.nan

        return {
            "realized_vol": realized_vol,
            "overnight_gap": overnight_gap,
            "intraday_range": intraday_range,
        }

    @staticmethod
    def _estimate_market_cap(ohlcv: pd.DataFrame, bar_idx: int) -> float:
        """Estimate market cap from trailing average daily dollar volume.

        Uses the empirical relationship:
            market_cap ≈ ADV / assumed_daily_turnover

        where assumed_daily_turnover = 0.5% is the median turnover rate
        across US equities.  This is a robust proxy when shares outstanding
        data is unavailable.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            OHLCV price data with ``Close`` and ``Volume`` columns.
        bar_idx : int
            Index of the current bar.

        Returns
        -------
        float
            Estimated market capitalization in USD.  Returns 0.0 if the
            estimate cannot be computed (no volume data).
        """
        ASSUMED_DAILY_TURNOVER = 0.005  # 0.5% — median for US equities
        LOOKBACK = 20  # trailing window for ADV

        start = max(0, bar_idx - LOOKBACK)
        close_slice = ohlcv["Close"].iloc[start : bar_idx + 1]
        if "Volume" not in ohlcv.columns:
            return 0.0
        vol_slice = ohlcv["Volume"].iloc[start : bar_idx + 1]
        dollar_volumes = close_slice * vol_slice
        valid = dollar_volumes.dropna()
        if len(valid) == 0:
            return 0.0
        adv = float(valid.mean())
        if adv <= 0 or not np.isfinite(adv):
            return 0.0
        return adv / ASSUMED_DAILY_TURNOVER

    @staticmethod
    def _effective_return_series(ohlcv: pd.DataFrame) -> pd.Series:
        """
        Return the best available close-to-close return stream.

        Preference:
            1) total_ret (includes CRSP delisting returns when available)
            2) Return
            3) Close.pct_change()
        """
        if "total_ret" in ohlcv.columns:
            series = pd.to_numeric(ohlcv["total_ret"], errors="coerce")
            if "Return" in ohlcv.columns:
                fallback = pd.to_numeric(ohlcv["Return"], errors="coerce")
                series = series.where(series.notna(), fallback)
        elif "Return" in ohlcv.columns:
            series = pd.to_numeric(ohlcv["Return"], errors="coerce")
        else:
            series = pd.to_numeric(ohlcv["Close"], errors="coerce").pct_change()
        return series.replace([np.inf, -np.inf], np.nan)

    def _delisting_adjustment_multiplier(
        self,
        ohlcv: pd.DataFrame,
        entry_idx: int,
        exit_idx: int,
    ) -> float:
        """
        Convert ret-based price-path return into total-return-path return.
        """
        if "total_ret" not in ohlcv.columns:
            return 1.0
        if exit_idx < entry_idx:
            return 1.0

        total = pd.to_numeric(
            ohlcv["total_ret"].iloc[entry_idx : exit_idx + 1],
            errors="coerce",
        )
        if "Return" in ohlcv.columns:
            base = pd.to_numeric(
                ohlcv["Return"].iloc[entry_idx : exit_idx + 1],
                errors="coerce",
            )
        else:
            base = (
                pd.to_numeric(ohlcv["Close"], errors="coerce")
                .pct_change()
                .iloc[entry_idx : exit_idx + 1]
            )

        denom = 1.0 + base
        numer = 1.0 + total
        valid = total.notna() & base.notna() & (denom.abs() > 1e-12)
        if not valid.any():
            return 1.0

        ratio = (numer[valid] / denom[valid]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) == 0:
            return 1.0
        return float(np.prod(ratio.to_numpy(dtype=float)))

    def _trade_realized_return(
        self,
        ohlcv: pd.DataFrame,
        entry_idx: int,
        exit_idx: int,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """
        Trade-level return including any total-return delisting adjustments.
        """
        price_return = (float(exit_price) - float(entry_price)) / max(1e-9, float(entry_price))
        multiplier = self._delisting_adjustment_multiplier(
            ohlcv=ohlcv,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
        )
        return float((1.0 + price_return) * multiplier - 1.0)

    @staticmethod
    def _is_permno_key(value: object) -> bool:
        """Return whether permno key satisfies the expected condition."""
        return _PERMNO_RE.match(str(value).strip()) is not None

    def _assert_permno_inputs(
        self,
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Enforce PERMNO-keyed panels at runtime when strict identity mode is on.
        """
        if not REQUIRE_PERMNO:
            return
        if not isinstance(predictions.index, pd.MultiIndex) or predictions.index.nlevels < 2:
            raise ValueError(
                "Backtester expects MultiIndex predictions with first level = PERMNO.",
            )

        pred_ids = predictions.index.get_level_values(0)
        bad_pred = sorted({str(x) for x in pred_ids if not self._is_permno_key(x)})
        bad_price = sorted({str(k) for k in price_data.keys() if not self._is_permno_key(k)})
        if bad_pred:
            preview = ", ".join(bad_pred[:5])
            raise ValueError(
                f"Non-PERMNO prediction keys detected: {preview}",
            )
        if bad_price:
            preview = ", ".join(bad_price[:5])
            raise ValueError(
                f"Non-PERMNO price_data keys detected: {preview}",
            )

    def run(
        self,
        predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run backtest across all tickers with cross-ticker position limits.

        Processes signals chronologically across ALL tickers to enforce
        the max_positions constraint globally.
        """
        if self.use_risk:
            self._init_risk_components()
        self._assert_permno_inputs(predictions=predictions, price_data=price_data)

        # Build regime lookup from predictions so the backtester uses
        # the same HMM regime as the model, not the legacy SMA50 proxy.
        self._regime_lookup = {}
        if "regime" in predictions.columns:
            for idx, row in predictions.iterrows():
                if isinstance(idx, tuple) and len(idx) == 2:
                    self._regime_lookup[idx] = int(row["regime"])
                else:
                    self._regime_lookup[idx] = int(row["regime"])

        # SPEC-W01: Pre-compute shock vectors for structural state conditioning.
        # This runs BOCPD + jump detection + drift/stress estimation per ticker
        # once, so the execution simulator can condition costs on structural
        # state without per-bar regime re-detection overhead.
        self._shock_vectors = {}
        tickers = predictions.index.get_level_values(0).unique()

        if EXEC_STRUCTURAL_STRESS_ENABLED:
            for ticker in tickers:
                if ticker not in price_data:
                    continue
                ohlcv = price_data[ticker]
                try:
                    ticker_preds = predictions.loc[ticker]
                    regime_s = (
                        ticker_preds["regime"].astype(int)
                        if "regime" in ticker_preds.columns else None
                    )
                    conf_s = (
                        ticker_preds["confidence"]
                        if "confidence" in ticker_preds.columns else None
                    )
                except KeyError:
                    regime_s = None
                    conf_s = None

                try:
                    shock_vecs = compute_shock_vectors(
                        ohlcv=ohlcv,
                        regime_series=regime_s,
                        confidence_series=conf_s,
                        ticker=str(ticker),
                    )
                    for dt, sv in shock_vecs.items():
                        self._shock_vectors[(str(ticker), dt)] = sv
                except Exception as e:
                    logger.warning(
                        "Shock vector computation failed for %s: %s", ticker, e,
                    )

        # Collect all candidate signals across tickers
        all_signals = []

        for ticker in tickers:
            if ticker not in price_data:
                continue

            try:
                ticker_preds = predictions.loc[ticker]
            except KeyError:
                continue

            # Filter to signal bars
            signals = ticker_preds[
                (ticker_preds["predicted_return"] > self.entry_threshold)
                & (ticker_preds["confidence"] > self.confidence_threshold)
            ]

            for dt, row in signals.iterrows():
                regime = int(row.get("regime", -1))
                confidence = float(row.get("confidence", 0))

                # SPEC-E02: Per-regime trade gating.  Each regime has an
                # ``enabled`` flag and a ``min_confidence`` threshold.
                #   • enabled=False → regime disabled; suppress ALL signals
                #     unless confidence >= min_confidence (high-conf override).
                #   • enabled=True  → regime allowed, but still enforce
                #     min_confidence as an additional floor.
                _policy = REGIME_TRADE_POLICY.get(
                    regime, {"enabled": True, "min_confidence": 0.0},
                )
                if not _policy["enabled"]:
                    if _policy["min_confidence"] <= 0 or confidence < _policy["min_confidence"]:
                        continue
                elif confidence < _policy["min_confidence"]:
                    continue

                all_signals.append({
                    "permno": str(ticker),
                    "date": dt,
                    "predicted_return": row["predicted_return"],
                    "confidence": confidence,
                    "regime": regime,
                })

        if not all_signals:
            if verbose:
                print("  No signals generated.")
            return self._empty_result()

        # Sort all signals chronologically
        signals_df = pd.DataFrame(all_signals).sort_values("date")

        # Process signals chronologically with position limits
        if self.use_risk:
            all_trades = self._process_signals_risk_managed(signals_df, price_data)
        else:
            all_trades = self._process_signals(signals_df, price_data)

        if not all_trades:
            if verbose:
                print("  No trades generated.")
            return self._empty_result()

        # SPEC-W02: trigger online recalibration after backtest completes.
        # This uses the accumulated trade cost observations to update
        # per-segment impact coefficients via EMA smoothing.
        if self._cost_calibrator is not None:
            calibration_result = self._cost_calibrator.calibrate()
            if calibration_result:
                logger.info(
                    "CostCalibrator: recalibrated %d segments after backtest",
                    len(calibration_result),
                )

        result = self._compute_metrics(all_trades, price_data, verbose=verbose)
        return result

    def _process_signals(
        self,
        signals_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[Trade]:
        """
        Process signals by day with capacity-aware ranking and realistic execution.
        """
        trades = []
        # Track open positions: list of (ticker, exit_date)
        open_positions: List[Tuple[str, pd.Timestamp]] = []
        # Track per-ticker last exit date to prevent overlapping positions
        ticker_last_exit: Dict[str, pd.Timestamp] = {}

        grouped = signals_df.groupby("date", sort=True)
        for dt, day_signals in grouped:
            # Expire finished positions before processing today's candidates
            open_positions = [(t, ed) for t, ed in open_positions if ed > dt]
            available_slots = self.max_positions - len(open_positions)
            if available_slots <= 0:
                continue

            ranked = day_signals.copy()
            ranked.loc[:, "score"] = ranked["predicted_return"] * (0.5 + ranked["confidence"])
            ranked = ranked.sort_values("score", ascending=False)

            entered_today = 0
            for _, signal in ranked.iterrows():
                if entered_today >= available_slots:
                    break

                permno = str(signal["permno"])
                if permno not in price_data:
                    continue
                ohlcv = price_data[permno]
                close = ohlcv["Close"]

                if permno in ticker_last_exit and dt <= ticker_last_exit[permno]:
                    continue
                if dt not in close.index:
                    continue

                signal_idx = close.index.get_loc(dt)
                entry_idx = signal_idx + 1
                if entry_idx >= len(ohlcv):
                    continue

                exit_idx = entry_idx + self.holding_days
                if exit_idx >= len(close):
                    continue

                # SPEC-E01: Edge-after-costs trade gate.
                # Skip trades where predicted edge does not exceed expected
                # round-trip cost plus an uncertainty-scaled buffer.
                shock = self._shock_vectors.get((permno, dt))
                if EDGE_COST_GATE_ENABLED:
                    entry_vol = (
                        float(ohlcv["Volume"].iloc[entry_idx])
                        if "Volume" in ohlcv.columns else 0.0
                    )
                    entry_ref_price = float(ohlcv["Open"].iloc[entry_idx])
                    context = self._execution_context(ohlcv=ohlcv, bar_idx=entry_idx)
                    desired_notional = self.assumed_capital_usd * self.position_size_pct
                    uncertainty = (
                        shock.hmm_uncertainty
                        if shock is not None and shock.hmm_uncertainty is not None
                        else 0.0
                    )

                    expected_cost_bps = self.execution_model.estimate_cost(
                        daily_volume=entry_vol,
                        desired_notional=desired_notional,
                        realized_vol=context["realized_vol"],
                        structure_uncertainty=uncertainty,
                        overnight_gap=context["overnight_gap"],
                        intraday_range=context["intraday_range"],
                        reference_price=entry_ref_price,
                        break_probability=(
                            shock.bocpd_changepoint_prob if shock else None
                        ),
                        drift_score=(
                            shock.structural_features.get("drift_score")
                            if shock else None
                        ),
                        systemic_stress=(
                            shock.structural_features.get("systemic_stress")
                            if shock else None
                        ),
                    )

                    predicted_edge_bps = abs(float(signal["predicted_return"])) * 10000
                    cost_buffer_bps = EDGE_COST_BUFFER_BASE_BPS * (1.0 + uncertainty)

                    if predicted_edge_bps <= expected_cost_bps + cost_buffer_bps:
                        logger.debug(
                            "Edge-cost gate: skipping %s on %s — edge %.1f bps "
                            "<= cost %.1f + buffer %.1f bps",
                            permno, dt, predicted_edge_bps,
                            expected_cost_bps, cost_buffer_bps,
                        )
                        continue

                # SPEC-W03: Apply uncertainty gate to position size.
                # When regime entropy is high, reduce position size to limit
                # exposure during uncertain regime transitions.
                adjusted_size = self.position_size_pct
                if shock is not None and shock.hmm_uncertainty is not None:
                    size_mult = self._uncertainty_gate.compute_size_multiplier(
                        shock.hmm_uncertainty,
                    )
                    adjusted_size *= size_mult

                entry_fill = self._simulate_entry(
                    ohlcv=ohlcv,
                    entry_idx=entry_idx,
                    position_size=adjusted_size,
                )
                if entry_fill is None:
                    continue

                entry_date = close.index[entry_idx]
                exit_date = close.index[exit_idx]
                exit_fill = self._simulate_exit(
                    ohlcv=ohlcv,
                    exit_idx=exit_idx,
                    shares=entry_fill["shares"],
                    force_full=True,
                )

                entry_price = float(entry_fill["entry_price"])
                exit_price = float(exit_fill["exit_price"])
                entry_ref = float(ohlcv["Open"].iloc[entry_idx])
                exit_ref = float(ohlcv["Close"].iloc[exit_idx])
                actual_return = self._trade_realized_return(
                    ohlcv=ohlcv,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    entry_price=entry_price,
                    exit_price=exit_price,
                )
                net_return = actual_return - self.tx_cost

                trades.append(Trade(
                    ticker=permno,
                    entry_date=str(entry_date.date()) if hasattr(entry_date, "date") else str(entry_date),
                    exit_date=str(exit_date.date()) if hasattr(exit_date, "date") else str(exit_date),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    predicted_return=float(signal["predicted_return"]),
                    actual_return=float(actual_return),
                    net_return=float(net_return),
                    regime=int(signal["regime"]),
                    confidence=float(signal["confidence"]),
                    holding_days=self.holding_days,
                    position_size=float(entry_fill["position_size"]),
                    exit_reason="holding",
                    fill_ratio=float(entry_fill["fill_ratio"]),
                    entry_impact_bps=float(entry_fill["entry_impact_bps"]),
                    exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                    entry_reference_price=entry_ref,
                    exit_reference_price=exit_ref,
                    market_cap_segment=entry_fill.get("market_cap_segment", ""),
                ))

                open_positions.append((permno, exit_date))
                ticker_last_exit[permno] = exit_date
                entered_today += 1

        return trades

    def _process_signals_risk_managed(
        self,
        signals_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[Trade]:
        """
        Process signals with full risk management.

        Adds dynamic position sizing, stop-loss evaluation,
        drawdown circuit breakers, and portfolio risk checks.
        """
        trades = []
        # Open positions: {ticker: {entry_price, entry_date, entry_idx, regime, highest_price, size, ...}}
        open_positions: Dict[str, dict] = {}
        ticker_last_exit: Dict[str, pd.Timestamp] = {}

        # Residual positions being unwound over multiple bars due to volume
        # constraints.  Keyed by ticker; each value tracks remaining shares,
        # cumulative fill records, and original entry data for final trade
        # recording once fully exited.
        residual_positions: Dict[str, dict] = {}

        # Also add intermediate dates from price data for daily stop evaluation
        all_trading_dates = set()
        for ticker in price_data:
            all_trading_dates.update(price_data[ticker].index.tolist())
        all_trading_dates = sorted(all_trading_dates)

        # Build signal lookup: date -> list of signals
        signal_lookup: Dict[pd.Timestamp, List[dict]] = {}
        for _, signal in signals_df.iterrows():
            dt = signal["date"]
            if dt not in signal_lookup:
                signal_lookup[dt] = []
            signal_lookup[dt].append(signal.to_dict())

        # Historical win rate tracking (for Kelly)
        completed_returns = []

        for dt in all_trading_dates:
            # ── 0. Process residual exits (volume-constrained multi-bar unwinding) ──
            residual_closed = []
            for ticker, res in residual_positions.items():
                if ticker not in price_data:
                    continue
                ohlcv = price_data[ticker]
                if dt not in ohlcv.index:
                    continue
                dt_idx = ohlcv.index.get_loc(dt)

                remaining = res["remaining_shares"]
                ref_price = float(ohlcv["Close"].iloc[dt_idx])
                vol = float(ohlcv["Volume"].iloc[dt_idx]) if "Volume" in ohlcv.columns else 0.0
                daily_dollar_vol = max(1e-9, ref_price * vol)
                max_exit_notional = daily_dollar_vol * self.execution_model.max_participation
                remaining_notional = remaining * ref_price

                if remaining_notional <= max_exit_notional:
                    shares_this_bar = remaining
                else:
                    shares_this_bar = max_exit_notional / max(1e-9, ref_price)

                if shares_this_bar <= 0:
                    continue

                exit_fill = self._simulate_exit(
                    ohlcv=ohlcv, exit_idx=dt_idx,
                    shares=shares_this_bar, force_full=True,
                )

                res["exit_fills"].append((float(exit_fill["exit_price"]), float(shares_this_bar)))
                res["remaining_shares"] -= shares_this_bar
                res["bars_since_exit_start"] += 1

                # Fully exited or max 20 bars of unwinding
                if res["remaining_shares"] <= 1e-6 or res["bars_since_exit_start"] > 20:
                    total_fill_shares = sum(s for _, s in res["exit_fills"])
                    vwap_exit = (
                        sum(p * s for p, s in res["exit_fills"])
                        / max(1e-9, total_fill_shares)
                    )
                    exit_ref = float(ohlcv["Close"].iloc[dt_idx])
                    actual_return = self._trade_realized_return(
                        ohlcv=ohlcv,
                        entry_idx=res["entry_idx"],
                        exit_idx=dt_idx,
                        entry_price=res["entry_price"],
                        exit_price=vwap_exit,
                    )
                    net_return = actual_return - self.tx_cost
                    total_holding = res["bars_held_at_stop"] + res["bars_since_exit_start"]

                    trades.append(Trade(
                        ticker=ticker,
                        entry_date=res["entry_date_str"],
                        exit_date=str(dt.date()) if hasattr(dt, "date") else str(dt),
                        entry_price=res["entry_price"],
                        exit_price=float(vwap_exit),
                        predicted_return=res["predicted_return"],
                        actual_return=float(actual_return),
                        net_return=float(net_return),
                        regime=res["regime"],
                        confidence=res["confidence"],
                        holding_days=total_holding,
                        position_size=res["size"],
                        exit_reason=res["exit_reason"],
                        fill_ratio=total_fill_shares / max(1e-9, res["total_shares"]),
                        entry_impact_bps=res.get("entry_impact_bps", 0.0),
                        exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                        entry_reference_price=res.get("entry_reference_price", 0.0),
                        exit_reference_price=exit_ref,
                        market_cap_segment=res.get("market_cap_segment", ""),
                    ))
                    completed_returns.append(net_return)
                    ticker_last_exit[ticker] = dt
                    residual_closed.append(ticker)

            for ticker in residual_closed:
                del residual_positions[ticker]

            # ── 1. Evaluate stops on all open positions ──
            closed_tickers = []
            for ticker, pos in list(open_positions.items()):
                if ticker not in price_data:
                    continue
                ohlcv = price_data[ticker]
                if dt not in ohlcv.index:
                    continue

                dt_idx = ohlcv.index.get_loc(dt)
                current_price = float(ohlcv["Close"].iloc[dt_idx])
                current_high = float(ohlcv["High"].iloc[dt_idx])
                pos["highest_price"] = max(pos["highest_price"], current_high)
                pos["bars_held"] += 1

                # Compute current ATR
                lookback_start = max(0, dt_idx - 14)
                h = ohlcv["High"].iloc[lookback_start:dt_idx + 1]
                l = ohlcv["Low"].iloc[lookback_start:dt_idx + 1]
                c = ohlcv["Close"].iloc[max(0, lookback_start - 1):dt_idx + 1]
                if len(h) > 1:
                    tr = pd.concat([
                        h - l,
                        (h - c.shift(1).iloc[-len(h):]).abs(),
                        (l - c.shift(1).iloc[-len(h):]).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.mean())
                else:
                    atr = float((ohlcv["High"].iloc[dt_idx] - ohlcv["Low"].iloc[dt_idx]))

                # Use HMM-derived regime from predictions if available,
                # falling back to entry regime. The backtester must use the
                # same regime definition as the model that generated signals.
                current_regime = pos["regime"]
                if hasattr(self, '_regime_lookup') and self._regime_lookup is not None:
                    regime_key = (pos["permno"], dt) if "permno" in pos else dt
                    if regime_key in self._regime_lookup:
                        current_regime = self._regime_lookup[regime_key]

                stop_result = self._stop_loss_mgr.evaluate(
                    entry_price=pos["entry_price"],
                    current_price=current_price,
                    highest_price=pos["highest_price"],
                    atr=atr,
                    bars_held=pos["bars_held"],
                    entry_regime=pos["regime"],
                    current_regime=current_regime,
                )

                if stop_result.should_exit:
                    # Volume-aware exit: check if position can fully exit
                    # within daily participation limits.
                    shares_to_exit = pos.get("shares", 0.0)
                    exit_ref_price = float(ohlcv["Close"].iloc[dt_idx])
                    exit_vol = float(ohlcv["Volume"].iloc[dt_idx]) if "Volume" in ohlcv.columns else 0.0
                    daily_dollar_vol = max(1e-9, exit_ref_price * exit_vol)
                    max_exit_notional = daily_dollar_vol * self.execution_model.max_participation
                    position_notional = shares_to_exit * exit_ref_price

                    if position_notional > max_exit_notional and max_exit_notional > 0:
                        # Partial fill — volume constraint binds
                        exit_fill_pct = max_exit_notional / position_notional
                        shares_this_bar = shares_to_exit * exit_fill_pct
                        remaining_shares = shares_to_exit - shares_this_bar
                        logger.debug(
                            "Partial exit for %s: %.0f%% of position (volume constraint)",
                            ticker, exit_fill_pct * 100,
                        )
                    else:
                        shares_this_bar = shares_to_exit
                        remaining_shares = 0.0

                    exit_fill = self._simulate_exit(
                        ohlcv=ohlcv,
                        exit_idx=dt_idx,
                        shares=shares_this_bar,
                        force_full=True,
                    )

                    entry_idx_val = int(pos.get("entry_idx", max(0, dt_idx - max(1, pos.get("bars_held", 1)))))

                    if remaining_shares > 1e-6:
                        # Multi-bar exit: track residual, record trade when fully unwound
                        residual_positions[ticker] = {
                            "remaining_shares": remaining_shares,
                            "total_shares": shares_to_exit,
                            "exit_fills": [(float(exit_fill["exit_price"]), float(shares_this_bar))],
                            "entry_price": pos["entry_price"],
                            "entry_date_str": pos["entry_date_str"],
                            "entry_idx": entry_idx_val,
                            "predicted_return": pos["predicted_return"],
                            "confidence": pos["confidence"],
                            "regime": pos["regime"],
                            "size": pos["size"],
                            "fill_ratio": pos.get("fill_ratio", 1.0),
                            "entry_impact_bps": pos.get("entry_impact_bps", 0.0),
                            "entry_reference_price": pos.get("entry_reference_price", 0.0),
                            "market_cap_segment": pos.get("market_cap_segment", ""),
                            "exit_reason": stop_result.reason.value,
                            "bars_held_at_stop": pos["bars_held"],
                            "bars_since_exit_start": 0,
                        }
                    else:
                        # Full exit — record trade immediately
                        exit_price = float(exit_fill["exit_price"])
                        exit_ref = float(ohlcv["Close"].iloc[dt_idx])
                        actual_return = self._trade_realized_return(
                            ohlcv=ohlcv,
                            entry_idx=entry_idx_val,
                            exit_idx=dt_idx,
                            entry_price=float(pos["entry_price"]),
                            exit_price=exit_price,
                        )
                        net_return = actual_return - self.tx_cost

                        trades.append(Trade(
                            ticker=ticker,
                            entry_date=pos["entry_date_str"],
                            exit_date=str(dt.date()) if hasattr(dt, "date") else str(dt),
                            entry_price=pos["entry_price"],
                            exit_price=float(exit_price),
                            predicted_return=pos["predicted_return"],
                            actual_return=float(actual_return),
                            net_return=float(net_return),
                            regime=pos["regime"],
                            confidence=pos["confidence"],
                            holding_days=pos["bars_held"],
                            position_size=pos["size"],
                            exit_reason=stop_result.reason.value,
                            fill_ratio=pos.get("fill_ratio", 1.0),
                            entry_impact_bps=pos.get("entry_impact_bps", 0.0),
                            exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                            entry_reference_price=pos.get("entry_reference_price", 0.0),
                            exit_reference_price=exit_ref,
                            market_cap_segment=pos.get("market_cap_segment", ""),
                        ))
                        completed_returns.append(net_return)
                        ticker_last_exit[ticker] = dt

                    closed_tickers.append(ticker)

            for ticker in closed_tickers:
                del open_positions[ticker]

            # ── 2. Update drawdown controller with daily PnL ──
            daily_pnl = 0.0
            for ticker, pos in open_positions.items():
                if ticker in price_data and dt in price_data[ticker].index:
                    ret_series = self._effective_return_series(price_data[ticker])
                    bar_ret = ret_series.loc[dt]
                    if pd.notna(bar_ret):
                        daily_pnl += float(bar_ret) * pos["size"]

            # Include residual positions in drawdown tracking — they still
            # have market exposure until fully unwound.
            for ticker, res in residual_positions.items():
                if ticker in price_data and dt in price_data[ticker].index:
                    ret_series = self._effective_return_series(price_data[ticker])
                    bar_ret = ret_series.loc[dt]
                    if pd.notna(bar_ret):
                        # Residual exposure is proportional to remaining shares
                        residual_frac = res["remaining_shares"] / max(1e-9, res["total_shares"])
                        daily_pnl += float(bar_ret) * res["size"] * residual_frac

            dd_status = self._drawdown_ctrl.update(daily_pnl)

            # CRITICAL state: force-close all open positions at the earliest
            # available mark so circuit-breakers are actually enforced.
            # Exits respect volume constraints — positions too large for a
            # single bar are moved to residual tracking for multi-bar unwinding.
            if dd_status.force_liquidate and open_positions:
                for ticker, pos in list(open_positions.items()):
                    ohlcv = price_data.get(ticker)
                    if ohlcv is None or len(ohlcv) == 0:
                        continue

                    if dt in ohlcv.index:
                        exit_date = dt
                        exit_idx = int(ohlcv.index.get_loc(dt))
                    else:
                        next_idx = int(ohlcv.index.searchsorted(dt, side="left"))
                        if next_idx >= len(ohlcv):
                            next_idx = len(ohlcv) - 1
                        exit_date = ohlcv.index[next_idx]
                        exit_idx = int(next_idx)

                    # Volume-aware circuit breaker exit
                    shares_to_exit = pos.get("shares", 0.0)
                    cb_ref_price = float(ohlcv["Close"].iloc[exit_idx])
                    cb_vol = float(ohlcv["Volume"].iloc[exit_idx]) if "Volume" in ohlcv.columns else 0.0
                    cb_daily_dollar_vol = max(1e-9, cb_ref_price * cb_vol)
                    cb_max_exit = cb_daily_dollar_vol * self.execution_model.max_participation
                    cb_position_notional = shares_to_exit * cb_ref_price

                    if cb_position_notional > cb_max_exit and cb_max_exit > 0:
                        cb_fill_pct = cb_max_exit / cb_position_notional
                        shares_this_bar = shares_to_exit * cb_fill_pct
                        cb_remaining = shares_to_exit - shares_this_bar
                    else:
                        shares_this_bar = shares_to_exit
                        cb_remaining = 0.0

                    exit_fill = self._simulate_exit(
                        ohlcv=ohlcv,
                        exit_idx=exit_idx,
                        shares=shares_this_bar,
                        force_full=True,
                    )

                    cb_entry_idx = int(pos.get("entry_idx", max(0, exit_idx - max(1, pos.get("bars_held", 1)))))

                    if cb_remaining > 1e-6:
                        # Multi-bar circuit breaker exit
                        residual_positions[ticker] = {
                            "remaining_shares": cb_remaining,
                            "total_shares": shares_to_exit,
                            "exit_fills": [(float(exit_fill["exit_price"]), float(shares_this_bar))],
                            "entry_price": pos["entry_price"],
                            "entry_date_str": pos["entry_date_str"],
                            "entry_idx": cb_entry_idx,
                            "predicted_return": pos["predicted_return"],
                            "confidence": pos["confidence"],
                            "regime": pos["regime"],
                            "size": pos["size"],
                            "fill_ratio": pos.get("fill_ratio", 1.0),
                            "entry_impact_bps": pos.get("entry_impact_bps", 0.0),
                            "entry_reference_price": pos.get("entry_reference_price", 0.0),
                            "market_cap_segment": pos.get("market_cap_segment", ""),
                            "exit_reason": "circuit_breaker",
                            "bars_held_at_stop": max(1, pos.get("bars_held", 0)),
                            "bars_since_exit_start": 0,
                        }
                    else:
                        exit_price = float(exit_fill["exit_price"])
                        exit_ref = float(ohlcv["Close"].iloc[exit_idx])
                        actual_return = self._trade_realized_return(
                            ohlcv=ohlcv,
                            entry_idx=cb_entry_idx,
                            exit_idx=exit_idx,
                            entry_price=float(pos["entry_price"]),
                            exit_price=exit_price,
                        )
                        net_return = actual_return - self.tx_cost

                        trades.append(Trade(
                            ticker=ticker,
                            entry_date=pos["entry_date_str"],
                            exit_date=str(exit_date.date()) if hasattr(exit_date, "date") else str(exit_date),
                            entry_price=pos["entry_price"],
                            exit_price=float(exit_price),
                            predicted_return=pos["predicted_return"],
                            actual_return=float(actual_return),
                            net_return=float(net_return),
                            regime=pos["regime"],
                            confidence=pos["confidence"],
                            holding_days=max(1, pos.get("bars_held", 0)),
                            position_size=pos["size"],
                            exit_reason="circuit_breaker",
                            fill_ratio=pos.get("fill_ratio", 1.0),
                            entry_impact_bps=pos.get("entry_impact_bps", 0.0),
                            exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                            entry_reference_price=pos.get("entry_reference_price", 0.0),
                            exit_reference_price=exit_ref,
                            market_cap_segment=pos.get("market_cap_segment", ""),
                        ))
                        completed_returns.append(net_return)
                        ticker_last_exit[ticker] = pd.Timestamp(exit_date)

                    del open_positions[ticker]

            # ── 3. Process new signals for this date ──
            if dt not in signal_lookup:
                continue

            if not dd_status.allow_new_entries:
                continue

            for signal in signal_lookup[dt]:
                ticker = str(signal["permno"])
                ohlcv = price_data.get(ticker)
                if ohlcv is None:
                    continue
                close = ohlcv["Close"]

                # Check per-ticker overlap (including positions being unwound)
                if ticker in open_positions:
                    continue
                if ticker in residual_positions:
                    continue
                if ticker in ticker_last_exit and dt <= ticker_last_exit[ticker]:
                    continue

                # Check global position limit
                if len(open_positions) >= self.max_positions:
                    continue

                # Entry at NEXT BAR OPEN
                if dt not in close.index:
                    continue
                signal_idx = close.index.get_loc(dt)
                entry_idx = signal_idx + 1
                if entry_idx >= len(ohlcv):
                    continue

                entry_price = float(ohlcv["Open"].iloc[entry_idx])
                entry_date = close.index[entry_idx]

                # SPEC-E01: Edge-after-costs trade gate (risk-managed mode).
                shock = self._shock_vectors.get((ticker, dt))
                if EDGE_COST_GATE_ENABLED:
                    entry_vol = (
                        float(ohlcv["Volume"].iloc[entry_idx])
                        if "Volume" in ohlcv.columns else 0.0
                    )
                    context = self._execution_context(ohlcv=ohlcv, bar_idx=entry_idx)
                    desired_notional = self.assumed_capital_usd * self.position_size_pct
                    uncertainty = (
                        shock.hmm_uncertainty
                        if shock is not None and shock.hmm_uncertainty is not None
                        else 0.0
                    )

                    expected_cost_bps = self.execution_model.estimate_cost(
                        daily_volume=entry_vol,
                        desired_notional=desired_notional,
                        realized_vol=context["realized_vol"],
                        structure_uncertainty=uncertainty,
                        overnight_gap=context["overnight_gap"],
                        intraday_range=context["intraday_range"],
                        reference_price=entry_price,
                        break_probability=(
                            shock.bocpd_changepoint_prob if shock else None
                        ),
                        drift_score=(
                            shock.structural_features.get("drift_score")
                            if shock else None
                        ),
                        systemic_stress=(
                            shock.structural_features.get("systemic_stress")
                            if shock else None
                        ),
                    )

                    predicted_edge_bps = abs(float(signal["predicted_return"])) * 10000
                    cost_buffer_bps = EDGE_COST_BUFFER_BASE_BPS * (1.0 + uncertainty)

                    if predicted_edge_bps <= expected_cost_bps + cost_buffer_bps:
                        logger.debug(
                            "Edge-cost gate (risk): skipping %s on %s — edge %.1f bps "
                            "<= cost %.1f + buffer %.1f bps",
                            ticker, dt, predicted_edge_bps,
                            expected_cost_bps, cost_buffer_bps,
                        )
                        continue

                # ── Dynamic position sizing ──
                # Compute recent volatility and ATR
                lookback_start = max(0, entry_idx - 20)
                recent_returns = close.iloc[lookback_start:entry_idx].pct_change().dropna()
                realized_vol = float(recent_returns.std() * np.sqrt(252)) if len(recent_returns) > 5 else 0.25

                lookback_atr = max(0, entry_idx - 14)
                h = ohlcv["High"].iloc[lookback_atr:entry_idx]
                l = ohlcv["Low"].iloc[lookback_atr:entry_idx]
                c = ohlcv["Close"].iloc[max(0, lookback_atr - 1):entry_idx]
                if len(h) > 1:
                    tr = pd.concat([
                        h - l,
                        (h - c.shift(1).iloc[-len(h):]).abs(),
                        (l - c.shift(1).iloc[-len(h):]).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.mean())
                else:
                    atr = entry_price * 0.02  # fallback 2%

                # Compute win rate from completed trades
                if len(completed_returns) > 20:
                    cr = np.array(completed_returns)
                    win_rate = (cr > 0).mean()
                    avg_win = cr[cr > 0].mean() if (cr > 0).any() else 0.02
                    avg_loss = cr[cr < 0].mean() if (cr < 0).any() else -0.02
                else:
                    win_rate, avg_win, avg_loss = 0.50, 0.02, -0.02

                ps = self._position_sizer.size_position(
                    ticker=ticker,
                    win_rate=float(win_rate),
                    avg_win=float(avg_win),
                    avg_loss=float(avg_loss),
                    realized_vol=realized_vol,
                    atr=atr,
                    price=entry_price,
                    holding_days=self.holding_days,
                    confidence=float(signal["confidence"]),
                    n_current_positions=len(open_positions),
                    max_positions=self.max_positions,
                )

                # Apply drawdown size multiplier
                regime_mult = float(REGIME_RISK_MULTIPLIER.get(int(signal["regime"]), 1.0))
                position_size = ps.final_size * dd_status.size_multiplier * regime_mult

                # SPEC-W03: Apply uncertainty gate to position size.
                # When regime entropy is high, reduce position size to limit
                # exposure during uncertain regime transitions.
                # (shock already looked up by SPEC-E01 gate above)
                if shock is not None and shock.hmm_uncertainty is not None:
                    size_mult = self._uncertainty_gate.compute_size_multiplier(
                        shock.hmm_uncertainty,
                    )
                    position_size *= size_mult

                if position_size <= 0:
                    continue

                # ── Portfolio risk check ──
                current_pos_dict = {t: p["size"] for t, p in open_positions.items()}
                risk_check = self._portfolio_risk.check_new_position(
                    ticker=ticker,
                    position_size=position_size,
                    current_positions=current_pos_dict,
                    price_data=price_data,
                )
                if not risk_check.passed:
                    continue

                entry_fill = self._simulate_entry(
                    ohlcv=ohlcv,
                    entry_idx=entry_idx,
                    position_size=position_size,
                )
                if entry_fill is None:
                    continue

                # Open position
                entry_ref = float(ohlcv["Open"].iloc[entry_idx])
                open_positions[ticker] = {
                    "entry_price": float(entry_fill["entry_price"]),
                    "entry_date": entry_date,
                    "entry_date_str": str(entry_date.date()) if hasattr(entry_date, "date") else str(entry_date),
                    "entry_idx": entry_idx,
                    "regime": int(signal["regime"]),
                    "predicted_return": float(signal["predicted_return"]),
                    "confidence": float(signal["confidence"]),
                    "highest_price": float(entry_fill["entry_price"]),
                    "size": float(entry_fill["position_size"]),
                    "shares": float(entry_fill["shares"]),
                    "fill_ratio": float(entry_fill["fill_ratio"]),
                    "entry_impact_bps": float(entry_fill["entry_impact_bps"]),
                    "entry_reference_price": entry_ref,
                    "bars_held": 0,
                    "permno": ticker,
                    "market_cap_segment": entry_fill.get("market_cap_segment", ""),
                }

        # ── Close any remaining open positions at last available price ──
        for ticker, pos in open_positions.items():
            if ticker not in price_data:
                continue
            ohlcv = price_data[ticker]
            last_idx = len(ohlcv) - 1
            exit_fill = self._simulate_exit(
                ohlcv=ohlcv,
                exit_idx=last_idx,
                shares=pos.get("shares", 0.0),
                force_full=True,
            )
            last_price = float(exit_fill["exit_price"])
            exit_ref = float(ohlcv["Close"].iloc[last_idx])
            actual_return = self._trade_realized_return(
                ohlcv=ohlcv,
                entry_idx=int(pos.get("entry_idx", 0)),
                exit_idx=last_idx,
                entry_price=float(pos["entry_price"]),
                exit_price=last_price,
            )
            net_return = actual_return - self.tx_cost

            trades.append(Trade(
                ticker=ticker,
                entry_date=pos["entry_date_str"],
                exit_date=str(ohlcv.index[-1].date()) if hasattr(ohlcv.index[-1], "date") else str(ohlcv.index[-1]),
                entry_price=pos["entry_price"],
                exit_price=float(last_price),
                predicted_return=pos["predicted_return"],
                actual_return=float(actual_return),
                net_return=float(net_return),
                regime=pos["regime"],
                confidence=pos["confidence"],
                holding_days=pos["bars_held"],
                position_size=pos["size"],
                exit_reason="end_of_data",
                fill_ratio=pos.get("fill_ratio", 1.0),
                entry_impact_bps=pos.get("entry_impact_bps", 0.0),
                exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                entry_reference_price=pos.get("entry_reference_price", 0.0),
                exit_reference_price=exit_ref,
                market_cap_segment=pos.get("market_cap_segment", ""),
            ))

        # ── Close any remaining residual positions (force full at end of data) ──
        for ticker, res in residual_positions.items():
            if ticker not in price_data:
                continue
            ohlcv = price_data[ticker]
            last_idx = len(ohlcv) - 1
            exit_fill = self._simulate_exit(
                ohlcv=ohlcv,
                exit_idx=last_idx,
                shares=res["remaining_shares"],
                force_full=True,
            )
            # Add final fill to the existing fill list and compute VWAP
            res["exit_fills"].append((float(exit_fill["exit_price"]), float(res["remaining_shares"])))
            total_fill_shares = sum(s for _, s in res["exit_fills"])
            vwap_exit = sum(p * s for p, s in res["exit_fills"]) / max(1e-9, total_fill_shares)

            exit_ref = float(ohlcv["Close"].iloc[last_idx])
            actual_return = self._trade_realized_return(
                ohlcv=ohlcv,
                entry_idx=res["entry_idx"],
                exit_idx=last_idx,
                entry_price=res["entry_price"],
                exit_price=vwap_exit,
            )
            net_return = actual_return - self.tx_cost
            total_holding = res["bars_held_at_stop"] + res.get("bars_since_exit_start", 0) + 1

            trades.append(Trade(
                ticker=ticker,
                entry_date=res["entry_date_str"],
                exit_date=str(ohlcv.index[-1].date()) if hasattr(ohlcv.index[-1], "date") else str(ohlcv.index[-1]),
                entry_price=res["entry_price"],
                exit_price=float(vwap_exit),
                predicted_return=res["predicted_return"],
                actual_return=float(actual_return),
                net_return=float(net_return),
                regime=res["regime"],
                confidence=res["confidence"],
                holding_days=total_holding,
                position_size=res["size"],
                exit_reason=res["exit_reason"],
                fill_ratio=total_fill_shares / max(1e-9, res["total_shares"]),
                entry_impact_bps=res.get("entry_impact_bps", 0.0),
                exit_impact_bps=float(exit_fill["exit_impact_bps"]),
                entry_reference_price=res.get("entry_reference_price", 0.0),
                exit_reference_price=exit_ref,
                market_cap_segment=res.get("market_cap_segment", ""),
            ))

        return trades

    def _compute_metrics(
        self,
        trades: List[Trade],
        price_data: Dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> BacktestResult:
        """Compute performance metrics from trade list."""
        # Use position-weighted returns if risk management is on
        if self.use_risk:
            returns = np.array([t.net_return * t.position_size / self.position_size_pct for t in trades])
        else:
            returns = np.array([t.net_return for t in trades])
        n = len(returns)

        winners = returns > 0
        losers = returns < 0

        win_rate = winners.sum() / n if n > 0 else 0
        avg_return = returns.mean()
        avg_win = returns[winners].mean() if winners.any() else 0
        avg_loss = returns[losers].mean() if losers.any() else 0
        total_return = np.prod(1 + returns) - 1

        # Time span for annualization
        dates = sorted(set(t.entry_date for t in trades))
        if len(dates) >= 2:
            first = datetime.strptime(dates[0], "%Y-%m-%d")
            last = datetime.strptime(dates[-1], "%Y-%m-%d")
            years = max((last - first).days / 365.25, 0.5)
        else:
            years = 1.0

        trades_per_year = n / years
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Annualization factor
        ann_factor = 252.0 / self.holding_days

        # Sharpe ratio (excess return over risk-free)
        rf_annual = 0.04
        rf_per_trade = rf_annual * self.holding_days / 252.0
        excess_returns = returns - rf_per_trade

        if returns.std() > 0:
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(ann_factor)
        else:
            sharpe = 0

        # Sortino ratio
        target = 0.0
        downside_diff = np.minimum(returns - target, 0)
        downside_deviation = np.sqrt(np.mean(downside_diff**2))
        if downside_deviation > 0:
            sortino = (returns.mean() / downside_deviation) * np.sqrt(ann_factor)
        else:
            sortino = 0

        # Build proper daily equity curve for drawdown
        daily_equity = self._build_daily_equity(trades, price_data)

        # Max drawdown from daily equity curve (or fallback to trade-level)
        if daily_equity is not None and len(daily_equity) > 1:
            eq_values = daily_equity.values
            running_max = np.maximum.accumulate(eq_values)
            drawdowns = (eq_values - running_max) / running_max
            max_dd = drawdowns.min()
        else:
            cum_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_dd = drawdowns.min() if len(drawdowns) > 0 else 0

        # Profit factor
        gross_profit = returns[winners].sum() if winners.any() else 0
        gross_loss = abs(returns[losers].sum()) if losers.any() else 0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0

        # Returns series
        returns_series = pd.Series(returns, name="trade_returns")
        equity_curve = pd.Series(np.cumprod(1 + returns), name="equity")

        # Regime breakdown
        regime_breakdown = {}
        for regime in set(t.regime for t in trades):
            r_trades = [t for t in trades if t.regime == regime]
            r_returns = np.array([t.net_return for t in r_trades])
            r_excess = r_returns - rf_per_trade
            regime_breakdown[regime] = {
                "n_trades": len(r_trades),
                "avg_return": float(r_returns.mean()),
                "win_rate": float((r_returns > 0).sum() / len(r_returns)),
                "sharpe": float(
                    (r_excess.mean() / r_returns.std()) * np.sqrt(ann_factor)
                    if r_returns.std() > 0 else 0
                ),
            }

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Risk report (if risk management enabled)
        risk_report = None
        dd_summary = None
        if self.use_risk and self._risk_metrics is not None:
            trade_returns_raw = np.array([t.net_return for t in trades])
            risk_report = self._risk_metrics.compute_full_report(
                trade_returns_raw, daily_equity, holding_days=self.holding_days,
            )
            dd_summary = self._drawdown_ctrl.get_summary()

        # ── Turnover tracking ──
        turnover_history, total_turnover, ann_turnover, avg_daily_turnover = (
            self._compute_turnover(trades)
        )

        # Warn if annualized turnover exceeds threshold
        if ann_turnover > MAX_ANNUALIZED_TURNOVER:
            msg = (
                f"Annualized turnover {ann_turnover:.0f}% exceeds threshold "
                f"{MAX_ANNUALIZED_TURNOVER:.0f}%. High turnover erodes returns "
                f"via transaction costs and market impact."
            )
            logger.warning(msg)
            if verbose:
                print(f"  WARNING: {msg}")

        # ── Per-regime performance (NEW 9) ──
        regime_perf = self._compute_regime_performance(trades)

        # ── Transaction Cost Analysis (NEW 10) ──
        tca_report = self._compute_tca(trades)

        result = BacktestResult(
            total_trades=n,
            winning_trades=int(winners.sum()),
            losing_trades=int(losers.sum()),
            win_rate=float(win_rate),
            avg_return=float(avg_return),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_dd),
            profit_factor=float(profit_factor),
            avg_holding_days=float(np.mean([t.holding_days for t in trades])),
            trades_per_year=float(trades_per_year),
            returns_series=returns_series,
            equity_curve=equity_curve,
            daily_equity=daily_equity,
            trades=trades,
            regime_breakdown=regime_breakdown,
            regime_performance=regime_perf,
            tca_report=tca_report,
            risk_report=risk_report,
            drawdown_summary=dd_summary,
            exit_reason_breakdown=exit_reasons,
            total_turnover=float(total_turnover),
            annualized_turnover=float(ann_turnover),
            avg_daily_turnover=float(avg_daily_turnover),
            turnover_history=turnover_history,
        )

        if verbose:
            self._print_result(result)

        return result

    def _build_daily_equity(
        self,
        trades: List[Trade],
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[pd.Series]:
        """
        Build a proper time-weighted daily equity curve.

        For each calendar day, computes mark-to-market PnL across all
        open positions. This gives an accurate drawdown measurement.
        """
        if not trades:
            return None

        # Collect all trade date ranges
        trade_entries = []
        for t in trades:
            try:
                entry_dt = pd.Timestamp(t.entry_date)
                exit_dt = pd.Timestamp(t.exit_date)
                trade_entries.append((t.ticker, entry_dt, exit_dt, t.entry_price, t.position_size))
            except (ValueError, TypeError):
                continue

        if not trade_entries:
            return None

        # Get full date range
        all_dates = set()
        for ticker, entry_dt, exit_dt, _, _ in trade_entries:
            if ticker in price_data:
                close = price_data[ticker]["Close"]
                mask = (close.index >= entry_dt) & (close.index <= exit_dt)
                all_dates.update(close.index[mask].tolist())

        if not all_dates:
            return None

        dates = sorted(all_dates)
        daily_pnl = pd.Series(0.0, index=dates)

        # For each trade, compute daily mark-to-market return
        for ticker, entry_dt, exit_dt, entry_price, pos_size in trade_entries:
            if ticker not in price_data:
                continue
            total_ret = self._effective_return_series(price_data[ticker])
            trade_returns = total_ret[(total_ret.index >= entry_dt) & (total_ret.index <= exit_dt)]
            if len(trade_returns) == 0:
                continue

            # Use total-return path; neutralize first bar to avoid counting
            # pre-entry close-to-close movement.
            position_returns = trade_returns.fillna(0.0).astype(float) * pos_size
            if len(position_returns) > 0:
                position_returns.iloc[0] = 0.0
            for dt, ret in position_returns.items():
                if dt in daily_pnl.index:
                    daily_pnl[dt] += ret

        # Build equity curve
        equity = (1 + daily_pnl).cumprod()
        return equity

    def _compute_turnover(
        self,
        trades: List[Trade],
    ) -> Tuple[List[float], float, float, float]:
        """
        Compute realized portfolio turnover from the trade list.

        Turnover is measured as the sum of absolute weight changes across all
        positions on each rebalance day.  An entry from 0% to 5% contributes
        5% of one-sided turnover; an exit from 5% to 0% contributes another
        5%.  A full round-trip therefore contributes 10%.

        Returns:
            turnover_history: list of (daily) turnover values per rebalance day
            total_turnover: cumulative sum of all daily turnovers (fraction, e.g. 5.0 = 500%)
            annualized_turnover: total_turnover scaled to 252 trading days (percentage)
            avg_daily_turnover: total_turnover / n_trading_days (percentage)
        """
        if not trades:
            return [], 0.0, 0.0, 0.0

        # Build a timeline of weight changes per day.
        # Each entry adds +position_size, each exit adds +position_size
        # (both are absolute weight changes).
        daily_turnover: Dict[str, float] = {}  # date_str -> turnover

        for t in trades:
            entry_date = str(t.entry_date)
            exit_date = str(t.exit_date)
            pos_size = abs(t.position_size)

            # Entry: weight goes from 0 to position_size
            daily_turnover[entry_date] = daily_turnover.get(entry_date, 0.0) + pos_size

            # Exit: weight goes from position_size to 0
            daily_turnover[exit_date] = daily_turnover.get(exit_date, 0.0) + pos_size

        # Sort by date and build history
        sorted_dates = sorted(daily_turnover.keys())
        turnover_history = [daily_turnover[d] for d in sorted_dates]

        total_turnover = sum(turnover_history)

        # Compute trading day span
        if len(sorted_dates) >= 2:
            try:
                first = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
                last = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
                n_calendar_days = max(1, (last - first).days)
                # Approximate trading days from calendar days
                n_trading_days = max(1, int(n_calendar_days * 252 / 365.25))
            except (ValueError, TypeError):
                n_trading_days = max(1, len(sorted_dates))
        else:
            n_trading_days = max(1, len(sorted_dates))

        # Annualized turnover as a percentage
        annualized_turnover = total_turnover * (252.0 / n_trading_days) * 100.0

        # Average daily turnover as a percentage
        avg_daily_turnover = (total_turnover / n_trading_days) * 100.0

        return turnover_history, total_turnover, annualized_turnover, avg_daily_turnover

    def _compute_regime_performance(
        self,
        trades: List[Trade],
    ) -> Dict:
        """Compute detailed per-regime performance metrics.

        Groups all trades by the regime active when the trade was opened and
        computes a comprehensive set of metrics for each regime including
        cumulative return, win rate, average return, Sharpe ratio, and trade
        count.

        Returns a dict keyed by regime integer with per-regime metric dicts.
        """
        if not trades:
            return {}

        ann_factor = 252.0 / max(1, self.holding_days)
        rf_annual = 0.04
        rf_per_trade = rf_annual * self.holding_days / 252.0

        regime_performance: Dict = {}
        regime_groups: Dict[int, List[Trade]] = {}
        for t in trades:
            regime_groups.setdefault(t.regime, []).append(t)

        for regime, r_trades in regime_groups.items():
            r_returns = np.array([t.net_return for t in r_trades])
            n = len(r_returns)
            winners = r_returns > 0
            losers = r_returns < 0

            cumulative_return = float(np.prod(1 + r_returns) - 1)
            win_rate = float(winners.sum() / n) if n > 0 else 0.0
            avg_return = float(r_returns.mean())
            avg_win = float(r_returns[winners].mean()) if winners.any() else 0.0
            avg_loss = float(r_returns[losers].mean()) if losers.any() else 0.0

            # Sharpe ratio
            excess = r_returns - rf_per_trade
            if r_returns.std() > 0 and n >= 2:
                sharpe = float((excess.mean() / r_returns.std()) * np.sqrt(ann_factor))
            else:
                sharpe = 0.0

            # Sortino ratio
            downside = np.minimum(r_returns, 0.0)
            downside_dev = float(np.sqrt(np.mean(downside ** 2)))
            if downside_dev > 0:
                sortino = float((r_returns.mean() / downside_dev) * np.sqrt(ann_factor))
            else:
                sortino = 0.0

            # Max drawdown within regime trades (sequential)
            cum = np.cumprod(1 + r_returns)
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / np.where(running_max > 0, running_max, 1.0)
            max_dd = float(dd.min()) if len(dd) > 0 else 0.0

            # Profit factor
            gross_profit = float(r_returns[winners].sum()) if winners.any() else 0.0
            gross_loss = float(abs(r_returns[losers].sum())) if losers.any() else 0.0
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf') if gross_profit > 0 else 0.0

            # Average holding days
            avg_holding = float(np.mean([t.holding_days for t in r_trades]))

            regime_performance[regime] = {
                "n_trades": n,
                "cumulative_return": cumulative_return,
                "avg_return": avg_return,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "profit_factor": float(profit_factor),
                "avg_holding_days": avg_holding,
            }

        return regime_performance

    def _compute_tca(self, trades: List[Trade]) -> Dict:
        """Compute Transaction Cost Analysis (TCA) report.

        For each trade, compares the realized slippage (fill_price vs
        reference_price) to the cost model's predicted impact.  Produces
        aggregate statistics broken down by side (entry=buy, exit=sell)
        and by market cap segment (SPEC-W02).

        Returns a TCA summary dict with average slippage, slippage std,
        slippage by side, predicted-vs-realized correlation, and
        per-segment cost breakdowns.
        """
        if not trades:
            return {}

        entry_slippages_bps: list = []
        exit_slippages_bps: list = []
        entry_predicted_bps: list = []
        exit_predicted_bps: list = []

        # SPEC-W02: per-segment cost accumulation
        from collections import defaultdict
        segment_costs: Dict[str, list] = defaultdict(list)

        for t in trades:
            # Entry side (buy): realized slippage = (fill - ref) / ref * 10000
            if t.entry_reference_price > 0:
                entry_slip = (
                    (t.entry_price - t.entry_reference_price)
                    / t.entry_reference_price
                    * 10_000
                )
                entry_slippages_bps.append(float(entry_slip))
                entry_predicted_bps.append(float(t.entry_impact_bps))

                # SPEC-W02: accumulate per-segment
                seg = t.market_cap_segment or "unknown"
                segment_costs[seg].append(float(entry_slip))

            # Exit side (sell): realized slippage = (ref - fill) / ref * 10000
            if t.exit_reference_price > 0:
                exit_slip = (
                    (t.exit_reference_price - t.exit_price)
                    / t.exit_reference_price
                    * 10_000
                )
                exit_slippages_bps.append(float(exit_slip))
                exit_predicted_bps.append(float(t.exit_impact_bps))

                # SPEC-W02: accumulate per-segment (exit uses same segment)
                seg = t.market_cap_segment or "unknown"
                segment_costs[seg].append(float(exit_slip))

        all_realized = np.array(entry_slippages_bps + exit_slippages_bps)
        all_predicted = np.array(entry_predicted_bps + exit_predicted_bps)

        if len(all_realized) == 0:
            return {}

        entry_arr = np.array(entry_slippages_bps) if entry_slippages_bps else np.array([0.0])
        exit_arr = np.array(exit_slippages_bps) if exit_slippages_bps else np.array([0.0])

        # Predicted vs realized correlation
        if len(all_realized) >= 3 and np.std(all_predicted) > 0 and np.std(all_realized) > 0:
            corr = float(np.corrcoef(all_predicted, all_realized)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Predicted vs realized RMSE
        resid = all_realized - all_predicted
        rmse_bps = float(np.sqrt(np.mean(resid ** 2)))

        tca: Dict = {
            "n_observations": len(all_realized),
            "avg_slippage_bps": float(all_realized.mean()),
            "slippage_std_bps": float(all_realized.std()),
            "slippage_median_bps": float(np.median(all_realized)),
            "slippage_buy_avg_bps": float(entry_arr.mean()),
            "slippage_sell_avg_bps": float(exit_arr.mean()),
            "predicted_impact_avg_bps": float(all_predicted.mean()),
            "predicted_vs_realized_corr": corr,
            "predicted_vs_realized_rmse_bps": rmse_bps,
            "total_cost_bps": float(all_realized.sum() / max(1, len(trades))),
        }

        # SPEC-W02: per-market-cap-segment cost breakdown
        segment_breakdown: Dict[str, Dict] = {}
        for seg, costs in sorted(segment_costs.items()):
            arr = np.array(costs)
            segment_breakdown[seg] = {
                "n_trades": len(costs),
                "avg_slippage_bps": float(arr.mean()),
                "median_slippage_bps": float(np.median(arr)),
                "std_slippage_bps": float(arr.std()) if len(arr) > 1 else 0.0,
            }
        tca["segment_breakdown"] = segment_breakdown

        # SPEC-W02: include calibrated coefficients if available
        if self._cost_calibrator is not None:
            tca["calibrated_coefficients"] = self._cost_calibrator.coefficients

        return tca

    def _print_result(self, r: BacktestResult):
        """Print backtest results."""
        from ..config import REGIME_NAMES

        print(f"\n{'='*60}")
        mode = "RISK-MANAGED" if self.use_risk else "SIMPLE"
        print(f"BACKTEST RESULTS [{mode}] — {self.holding_days}d holding")
        print(f"{'='*60}")
        print(f"  Trades: {r.total_trades} ({r.trades_per_year:.0f}/year)")
        print(f"  Win rate: {r.win_rate:.1%}")
        print(f"  Avg return: {r.avg_return:.4f} (win: {r.avg_win:.4f}, loss: {r.avg_loss:.4f})")
        print(f"  Sharpe: {r.sharpe_ratio:.2f}")
        print(f"  Sortino: {r.sortino_ratio:.2f}")
        print(f"  Max DD: {r.max_drawdown:.1%}")
        print(f"  Profit factor: {r.profit_factor:.2f}")
        print(f"  Annualized return: {r.annualized_return:.1%}")

        # Turnover reporting
        if r.total_turnover > 0:
            print(f"\n  Turnover:")
            print(f"    Total turnover: {r.total_turnover:.2f}x ({r.total_turnover * 100:.0f}%)")
            print(f"    Annualized turnover: {r.annualized_turnover:.0f}%")
            print(f"    Avg daily turnover: {r.avg_daily_turnover:.2f}%")
            if r.annualized_turnover > MAX_ANNUALIZED_TURNOVER:
                print(f"    ** WARNING: exceeds {MAX_ANNUALIZED_TURNOVER:.0f}% threshold **")

        if r.regime_breakdown:
            print(f"\n  By regime:")
            for regime, stats in sorted(r.regime_breakdown.items()):
                name = REGIME_NAMES.get(regime, f"regime_{regime}")
                print(f"    {name}: {stats['n_trades']} trades, "
                      f"avg={stats['avg_return']:.4f}, "
                      f"wr={stats['win_rate']:.0%}, "
                      f"sharpe={stats['sharpe']:.2f}")

        # ── Per-Regime Performance (NEW 9) ──
        if r.regime_performance:
            print(f"\n  Per-Regime Performance:")
            for regime, rp in sorted(r.regime_performance.items()):
                name = REGIME_NAMES.get(regime, f"regime_{regime}")
                pf_str = (
                    f"{rp['profit_factor']:.2f}"
                    if np.isfinite(rp["profit_factor"])
                    else "inf"
                )
                print(f"    {name} ({rp['n_trades']} trades):")
                print(f"      Cumulative return: {rp['cumulative_return']:.2%}")
                print(f"      Avg return: {rp['avg_return']:.4f} "
                      f"(win: {rp['avg_win']:.4f}, loss: {rp['avg_loss']:.4f})")
                print(f"      Win rate: {rp['win_rate']:.1%}")
                print(f"      Sharpe: {rp['sharpe']:.2f}  |  "
                      f"Sortino: {rp['sortino']:.2f}")
                print(f"      Max DD: {rp['max_drawdown']:.1%}  |  "
                      f"Profit factor: {pf_str}")
                print(f"      Avg holding: {rp['avg_holding_days']:.1f} days")

        # ── Transaction Cost Analysis (NEW 10) ──
        if r.tca_report:
            tca = r.tca_report
            print(f"\n  Transaction Cost Analysis ({tca['n_observations']} observations):")
            print(f"    Avg slippage: {tca['avg_slippage_bps']:.2f} bps "
                  f"(std: {tca['slippage_std_bps']:.2f}, "
                  f"median: {tca['slippage_median_bps']:.2f})")
            print(f"    Buy slippage:  {tca['slippage_buy_avg_bps']:.2f} bps  |  "
                  f"Sell slippage: {tca['slippage_sell_avg_bps']:.2f} bps")
            print(f"    Predicted impact avg: {tca['predicted_impact_avg_bps']:.2f} bps")
            print(f"    Predicted vs realized corr: {tca['predicted_vs_realized_corr']:.3f}")
            print(f"    Predicted vs realized RMSE: {tca['predicted_vs_realized_rmse_bps']:.2f} bps")
            print(f"    Round-trip cost per trade: {tca['total_cost_bps']:.2f} bps")

            # SPEC-W02: per-segment cost breakdown
            seg_breakdown = tca.get("segment_breakdown", {})
            if seg_breakdown:
                print(f"    By market cap segment:")
                for seg, stats in seg_breakdown.items():
                    print(f"      {seg:>6s}: {stats['avg_slippage_bps']:6.2f} bps avg "
                          f"({stats['n_trades']} trades, "
                          f"median: {stats['median_slippage_bps']:.2f})")

            cal_coeffs = tca.get("calibrated_coefficients", {})
            if cal_coeffs:
                print(f"    Calibrated impact coefficients:")
                for seg in ("micro", "small", "mid", "large"):
                    if seg in cal_coeffs:
                        print(f"      {seg:>6s}: {cal_coeffs[seg]:6.1f} bps/sqrt(part)")

        if r.exit_reason_breakdown:
            print(f"\n  Exit reasons:")
            for reason, count in sorted(r.exit_reason_breakdown.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count} ({count/r.total_trades:.0%})")

        if r.risk_report is not None:
            rr = r.risk_report
            print(f"\n  Risk Metrics:")
            print(f"    VaR 95%: {rr.var_95:.4f}")
            print(f"    CVaR 95%: {rr.cvar_95:.4f}")
            print(f"    Tail ratio: {rr.tail_ratio:.2f}")
            print(f"    Ulcer Index: {rr.ulcer_index:.4f}")
            print(f"    Calmar: {rr.calmar_ratio:.2f}")

        if r.drawdown_summary:
            ds = r.drawdown_summary
            print(f"\n  Drawdown Control:")
            print(f"    Circuit breakers triggered: {ds.get('n_circuit_breakers', 0)}")
            print(f"    Days in warning: {ds.get('days_in_warning', 0)}")
            print(f"    Days in caution: {ds.get('days_in_caution', 0)}")

    def _empty_result(self) -> BacktestResult:
        """Internal helper for empty result."""
        return BacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_return=0, avg_win=0, avg_loss=0,
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, profit_factor=0,
            avg_holding_days=0, trades_per_year=0,
        )
