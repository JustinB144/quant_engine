"""
Stateful paper-trading engine for promoted strategies.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    PAPER_STATE_PATH,
    PAPER_INITIAL_CAPITAL,
    PAPER_MAX_TOTAL_POSITIONS,
    TRANSACTION_COST_BPS,
    PAPER_USE_KELLY_SIZING,
    PAPER_KELLY_FRACTION,
    PAPER_KELLY_LOOKBACK_TRADES,
    PAPER_KELLY_MIN_SIZE_MULTIPLIER,
    PAPER_KELLY_MAX_SIZE_MULTIPLIER,
    REGIME_RISK_MULTIPLIER,
    EXEC_SPREAD_BPS,
    EXEC_MAX_PARTICIPATION,
    EXEC_IMPACT_COEFF_BPS,
    EXEC_MIN_FILL_RATIO,
    EXEC_DYNAMIC_COSTS,
    EXEC_DOLLAR_VOLUME_REF_USD,
    EXEC_VOL_REF,
    EXEC_VOL_SPREAD_BETA,
    EXEC_GAP_SPREAD_BETA,
    EXEC_RANGE_SPREAD_BETA,
    EXEC_VOL_IMPACT_BETA,
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
    EXEC_CALIBRATION_FEEDBACK_ENABLED,
    EXEC_CALIBRATION_FEEDBACK_INTERVAL_DAYS,
    EXEC_CALIBRATION_FEEDBACK_PATH,
    EXEC_CALIBRATION_MIN_TRADES,
    EXEC_CALIBRATION_MIN_SEGMENT_TRADES,
    EXEC_CALIBRATION_SMOOTHING,
)
from ..backtest.execution import ExecutionModel
from ..backtest.adv_tracker import ADVTracker
from ..backtest.cost_calibrator import CostCalibrator
from ..risk.position_sizer import PositionSizer
from ..risk.stop_loss import StopLossManager
from ..risk.portfolio_risk import PortfolioRiskManager
from .registry import ActiveStrategy

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Executes paper entries/exits from promoted strategy definitions.
    """

    def __init__(
        self,
        state_path: Path = PAPER_STATE_PATH,
        initial_capital: float = PAPER_INITIAL_CAPITAL,
        max_total_positions: int = PAPER_MAX_TOTAL_POSITIONS,
        transaction_cost_bps: float = TRANSACTION_COST_BPS,
        use_kelly_sizing: bool = PAPER_USE_KELLY_SIZING,
        kelly_fraction: float = PAPER_KELLY_FRACTION,
        kelly_lookback_trades: int = PAPER_KELLY_LOOKBACK_TRADES,
        kelly_min_size_multiplier: float = PAPER_KELLY_MIN_SIZE_MULTIPLIER,
        kelly_max_size_multiplier: float = PAPER_KELLY_MAX_SIZE_MULTIPLIER,
    ):
        """Initialize PaperTrader."""
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = float(initial_capital)
        self.max_total_positions = int(max_total_positions)
        self.tx_cost = float(transaction_cost_bps) / 10000.0
        self.use_kelly_sizing = bool(use_kelly_sizing)
        self.kelly_fraction = float(np.clip(kelly_fraction, 0.05, 1.0))
        self.kelly_lookback_trades = int(max(20, kelly_lookback_trades))
        self.kelly_min_mult = float(max(0.05, kelly_min_size_multiplier))
        self.kelly_max_mult = float(max(self.kelly_min_mult, kelly_max_size_multiplier))

        # Persistent sizer for regime stats and Bayesian updates across cycles
        self._persistent_sizer = PositionSizer(kelly_fraction=self.kelly_fraction)

        # Drawdown controller — tracks equity and enforces circuit breakers
        from ..risk.drawdown import DrawdownController
        self._dd_controller = DrawdownController(initial_equity=self.initial_capital)
        self._prev_equity: Optional[float] = None

        # Stop loss manager — ATR stops, trailing stops, hard stops, regime-change stops
        self._stop_mgr = StopLossManager()

        # Portfolio risk manager — sector, correlation, single-name, volatility limits
        self._risk_mgr = PortfolioRiskManager()

        # Spec 09: Health-to-risk feedback gate
        self._health_risk_gate = self._init_health_gate()

        # Spec 06: Execution model with structural state-aware costs
        self._execution_model = ExecutionModel(
            spread_bps=EXEC_SPREAD_BPS,
            max_participation_rate=EXEC_MAX_PARTICIPATION,
            impact_coefficient_bps=EXEC_IMPACT_COEFF_BPS,
            min_fill_ratio=EXEC_MIN_FILL_RATIO,
            dynamic_costs=EXEC_DYNAMIC_COSTS,
            dollar_volume_ref_usd=EXEC_DOLLAR_VOLUME_REF_USD,
            vol_ref=EXEC_VOL_REF,
            vol_spread_beta=EXEC_VOL_SPREAD_BETA,
            gap_spread_beta=EXEC_GAP_SPREAD_BETA,
            range_spread_beta=EXEC_RANGE_SPREAD_BETA,
            vol_impact_beta=EXEC_VOL_IMPACT_BETA,
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
        self._execution_model.set_base_transaction_cost_bps(
            float(transaction_cost_bps)
        )

        # Spec 06: ADV tracker for volume trend analysis
        self._adv_tracker = ADVTracker(
            lookback_days=ADV_LOOKBACK_DAYS,
            ema_span=ADV_EMA_SPAN,
            low_volume_cost_mult=EXEC_LOW_VOLUME_COST_MULT,
        )

        # SPEC-E04: Cost calibrator with feedback loop
        if EXEC_CALIBRATION_FEEDBACK_ENABLED:
            self._cost_calibrator = CostCalibrator(
                min_total_trades=EXEC_CALIBRATION_MIN_TRADES,
                min_segment_trades=EXEC_CALIBRATION_MIN_SEGMENT_TRADES,
                smoothing=EXEC_CALIBRATION_SMOOTHING,
                model_dir=Path("trained_models"),
                feedback_path=EXEC_CALIBRATION_FEEDBACK_PATH,
                feedback_interval_days=EXEC_CALIBRATION_FEEDBACK_INTERVAL_DAYS,
            )
        else:
            self._cost_calibrator = None

        # A/B test registry — lazily loaded to avoid circular imports
        self._ab_registry = None

    @staticmethod
    def _init_health_gate():
        """Initialize the health-to-risk feedback gate (Spec 09).

        Loads configuration from health.yaml; returns None if unavailable
        (health gate is optional and non-blocking).
        """
        try:
            from ..api.services.health_risk_feedback import create_health_risk_gate
            return create_health_risk_gate()
        except Exception as e:
            logger.debug("Health risk gate unavailable: %s", e)
            return None

    def _apply_health_gate(self, position_size_pct: float) -> float:
        """Apply health-based position size scaling (Spec 09).

        Reads the latest overall health score from the health history
        database and reduces position sizes proportionally if health
        is degraded.  No-op if the health gate is disabled or unavailable.
        """
        if self._health_risk_gate is None:
            return position_size_pct
        try:
            from ..api.services.health_service import HealthService
            svc = HealthService()
            history = svc.get_health_history(limit=1)
            if history:
                latest_score = history[-1].get("overall_score")
                if latest_score is not None:
                    self._health_risk_gate.update_health(latest_score)
                    return self._health_risk_gate.apply_health_gate(
                        position_size_pct, latest_score,
                    )
        except Exception as e:
            logger.debug("Health gate lookup failed: %s", e)
        return position_size_pct

    def _get_active_ab_test(self):
        """Return the active A/B test, if any.

        Lazily initializes the ABTestRegistry on first call to avoid
        circular imports at module load time.
        """
        try:
            if self._ab_registry is None:
                from ..api.ab_testing import ABTestRegistry
                self._ab_registry = ABTestRegistry()
            return self._ab_registry.get_active_test()
        except Exception as e:
            logger.debug("A/B test registry unavailable: %s", e)
            return None

    def _record_trade_to_ab_test(
        self, ab_test, variant: str, trade_record: Dict,
    ) -> None:
        """Record a closed trade to the active A/B test variant."""
        if ab_test is None or not variant:
            return
        try:
            ab_test.record_trade(variant, trade_record)
            # Persist after recording
            if self._ab_registry is not None:
                self._ab_registry._save()
        except Exception as e:
            logger.warning("Failed to record trade to A/B test: %s", e)

    def _load_state(self) -> Dict:
        """Internal helper to load state."""
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {
            "cash": self.initial_capital,
            "realized_pnl": 0.0,
            "positions": [],
            "trades": [],
            "last_update": None,
        }

    def _save_state(self, state: Dict):
        # Keep trade history bounded to prevent unbounded state growth.
        """Internal helper to persist state."""
        state["trades"] = state.get("trades", [])[-5000:]
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @staticmethod
    def _resolve_as_of(price_data: Dict[str, pd.DataFrame]) -> pd.Timestamp:
        """Internal helper for resolve as of."""
        dates = []
        for df in price_data.values():
            if len(df) > 0:
                dates.append(pd.Timestamp(df.index.max()))
        if not dates:
            raise ValueError("No price data available for paper trading")
        return max(dates)

    @staticmethod
    def _latest_predictions_by_id(latest_predictions: pd.DataFrame) -> Dict[str, Dict]:
        """Internal helper for latest predictions by id."""
        if latest_predictions is None or len(latest_predictions) == 0:
            return {}
        df = latest_predictions.copy()
        if "permno" in df.columns:
            key_col = "permno"
        elif "ticker" in df.columns:
            key_col = "ticker"
        else:
            raise ValueError("latest_predictions must contain a 'permno' (or legacy 'ticker') column")
        return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}

    @staticmethod
    def _latest_predictions_by_ticker(latest_predictions: pd.DataFrame) -> Dict[str, Dict]:
        # Backward-compatible alias.
        """Internal helper for latest predictions by ticker."""
        return PaperTrader._latest_predictions_by_id(latest_predictions)

    @staticmethod
    def _current_price(
        ticker: str,
        as_of: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[float]:
        """Internal helper for current price."""
        df = price_data.get(ticker)
        if df is None or len(df) == 0:
            return None
        series = df["Close"]
        prior = series[series.index <= as_of]
        if len(prior) == 0:
            return None
        px = float(prior.iloc[-1])
        if px <= 0:
            return None
        return px

    @staticmethod
    def _position_id(position: Dict) -> str:
        """
        Canonical position key (PERMNO in PERMNO-first mode), with legacy fallback.
        """
        if "permno" in position:
            return str(position["permno"])
        return str(position.get("ticker", ""))

    def _mark_to_market(self, state: Dict, as_of: pd.Timestamp, price_data: Dict[str, pd.DataFrame]) -> float:
        """Internal helper for mark to market."""
        eq = float(state.get("cash", 0.0))
        for pos in state.get("positions", []):
            px = self._current_price(self._position_id(pos), as_of, price_data)
            if px is not None:
                eq += float(pos["shares"]) * px
        return eq

    @staticmethod
    def _trade_return(trade: Dict) -> Optional[float]:
        """Internal helper for trade return."""
        entry_price = float(trade.get("entry_price", 0.0))
        shares = float(trade.get("shares", 0.0))
        pnl = float(trade.get("pnl", 0.0))
        notional = entry_price * shares
        if notional <= 0:
            return None
        return float(pnl / notional)

    def _historical_trade_stats(
        self,
        state: Dict,
        strategy_id: str,
    ) -> Tuple[float, float, float]:
        """
        Estimate win/loss stats from recent closed paper trades.
        """
        all_trades = state.get("trades", [])
        strat_trades = [t for t in all_trades if str(t.get("strategy_id")) == str(strategy_id)]
        trades = strat_trades if len(strat_trades) >= 20 else all_trades
        trades = trades[-self.kelly_lookback_trades:]

        returns = []
        for t in trades:
            r = self._trade_return(t)
            if r is not None and np.isfinite(r):
                returns.append(float(r))

        if len(returns) < 20:
            return 0.50, 0.02, -0.02

        arr = np.array(returns, dtype=float)
        win_mask = arr > 0
        lose_mask = arr < 0
        win_rate = float(win_mask.mean())
        avg_win = float(arr[win_mask].mean()) if win_mask.any() else 0.02
        avg_loss = float(arr[lose_mask].mean()) if lose_mask.any() else -0.02

        if avg_loss >= 0:
            avg_loss = -0.02
        if avg_win <= 0:
            avg_win = 0.02
        return win_rate, avg_win, avg_loss

    @staticmethod
    def _market_risk_stats(
        ticker: str,
        as_of: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[Dict[str, float]]:
        """Internal helper for market risk stats."""
        df = price_data.get(ticker)
        if df is None or len(df) < 20:
            return None

        hist = df[df.index <= as_of]
        if len(hist) < 20:
            return None

        close = hist["Close"].astype(float)
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)

        returns = close.pct_change().dropna().iloc[-20:]
        realized_vol = float(returns.std() * np.sqrt(252)) if len(returns) >= 5 else 0.25

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1).dropna()
        atr = float(tr.iloc[-14:].mean()) if len(tr) > 0 else float(close.iloc[-1] * 0.02)
        price = float(close.iloc[-1])

        return {
            "realized_vol": realized_vol if np.isfinite(realized_vol) else 0.25,
            "atr": atr if np.isfinite(atr) and atr > 0 else price * 0.02,
            "price": price,
        }

    @staticmethod
    def _extract_structural_state(
        pred_row: Optional[Dict],
    ) -> Dict[str, Optional[float]]:
        """Extract structural state inputs from a prediction row.

        Returns a dict with keys: break_probability, structure_uncertainty,
        drift_score, systemic_stress.  Values are None if not available.
        """
        if pred_row is None:
            return {
                "break_probability": None,
                "structure_uncertainty": None,
                "drift_score": None,
                "systemic_stress": None,
            }

        def _safe_float(val) -> Optional[float]:
            if val is None:
                return None
            try:
                v = float(val)
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None

        return {
            "break_probability": _safe_float(
                pred_row.get("break_probability")
            ),
            "structure_uncertainty": _safe_float(
                pred_row.get("regime_entropy",
                             pred_row.get("structure_uncertainty"))
            ),
            "drift_score": _safe_float(pred_row.get("drift_score")),
            "systemic_stress": _safe_float(
                pred_row.get("systemic_stress",
                             pred_row.get("vix_percentile"))
            ),
        }

    def _record_fill_feedback(
        self,
        fill,
        reference_price: float,
        side: str,
        symbol: str,
        daily_volume: float,
        price: float,
        regime: int,
        as_of: pd.Timestamp,
    ) -> None:
        """Record predicted-vs-actual cost for calibration and health monitoring.

        SPEC-E04: Computes the actual execution cost from the fill slippage
        and records it alongside the model's predicted cost for later
        recalibration.

        SPEC-H03: Also saves each fill's predicted-vs-actual cost to the
        execution quality health monitoring database for rolling cost
        surprise tracking.
        """
        if fill.fill_ratio <= 0 or fill.participation_rate <= 0:
            return

        predicted_cost = fill.cost_details.get("total_bps", 0.0)
        if not np.isfinite(predicted_cost) or predicted_cost <= 0:
            return

        # Compute actual cost from fill slippage
        if reference_price <= 0:
            return
        if side == "buy":
            actual_cost = (
                (fill.fill_price - reference_price) / reference_price * 10000.0
            )
        else:
            actual_cost = (
                (reference_price - fill.fill_price) / reference_price * 10000.0
            )

        if not np.isfinite(actual_cost):
            return

        # SPEC-E04: Record to cost calibrator for coefficient recalibration
        if self._cost_calibrator is not None:
            # Estimate market cap from price * volume (rough proxy)
            estimated_mcap = float(price * daily_volume * 200.0)

            self._cost_calibrator.record_actual_fill(
                symbol=symbol,
                market_cap=estimated_mcap,
                predicted_cost_bps=float(predicted_cost),
                actual_cost_bps=float(max(0.0, actual_cost)),
                participation_rate=float(fill.participation_rate),
                regime=int(regime),
                fill_timestamp=str(as_of.isoformat()),
            )

        # SPEC-H03: Record to execution quality health monitoring DB
        self._record_execution_quality(
            symbol=symbol,
            side=side,
            predicted_cost_bps=float(predicted_cost),
            actual_cost_bps=float(max(0.0, actual_cost)),
            fill_ratio=float(fill.fill_ratio),
            participation_rate=float(fill.participation_rate),
            regime=int(regime),
        )

    def _record_execution_quality(
        self,
        symbol: str,
        side: str,
        predicted_cost_bps: float,
        actual_cost_bps: float,
        fill_ratio: float,
        participation_rate: float,
        regime: int,
    ) -> None:
        """Persist a fill's execution quality to the health monitoring DB.

        SPEC-H03: Saves predicted-vs-actual cost data for each paper-trade
        fill to SQLite so the health service can track rolling cost surprise.
        Non-blocking — silently drops records on failure.
        """
        try:
            from ..api.services.health_service import HealthService
            svc = HealthService()
            svc.save_execution_quality_fill(
                symbol=symbol,
                side=side,
                predicted_cost_bps=predicted_cost_bps,
                actual_cost_bps=actual_cost_bps,
                fill_ratio=fill_ratio,
                participation_rate=participation_rate,
                regime=regime,
            )
        except Exception as e:
            logger.debug("SPEC-H03: execution quality recording failed: %s", e)

    def _get_volume(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Get daily volume for a ticker and update ADV tracker."""
        df = price_data.get(ticker)
        if df is None or len(df) == 0:
            return 0.0
        hist = df[df.index <= as_of]
        if len(hist) == 0:
            return 0.0
        vol = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0
        if vol > 0:
            self._adv_tracker.update(ticker, vol)
        return vol

    @staticmethod
    def _compute_atr(
        ticker: str,
        as_of: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
        lookback: int = 14,
    ) -> float:
        """Compute Average True Range for stop loss evaluation."""
        df = price_data.get(ticker)
        if df is None or len(df) < 2:
            return 0.0

        hist = df[df.index <= as_of]
        if len(hist) < 2:
            return 0.0

        close = hist["Close"].astype(float)
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1).dropna()

        if len(tr) == 0:
            return float(close.iloc[-1] * 0.02)

        return float(tr.iloc[-lookback:].mean())

    def _update_equity_curve(
        self,
        state: Dict,
        as_of_str: str,
        equity: float,
        daily_pnl: float,
        dd_status,
    ) -> Dict:
        """Append to equity curve and compute risk metrics if enough data.

        Returns a dict of risk metrics (empty if fewer than 20 data points).
        """
        if "equity_curve" not in state:
            state["equity_curve"] = []

        state["equity_curve"].append({
            "date": as_of_str,
            "equity": float(equity),
            "cash": float(state.get("cash", 0.0)),
            "positions_value": float(equity) - float(state.get("cash", 0.0)),
            "n_positions": len(state.get("positions", [])),
            "daily_pnl": float(daily_pnl),
            "drawdown": float(dd_status.current_drawdown),
            "dd_state": dd_status.state.name,
        })

        # Cap equity curve to last 2520 entries (10 years of daily data)
        if len(state["equity_curve"]) > 2520:
            state["equity_curve"] = state["equity_curve"][-2520:]

        # Compute risk metrics if enough data
        risk_metrics: Dict = {}
        if len(state["equity_curve"]) >= 20:
            equity_series = pd.Series(
                [e["equity"] for e in state["equity_curve"]]
            )
            returns_series = equity_series.pct_change().dropna()
            if len(returns_series) > 0 and returns_series.std() > 0:
                ann_return = float(returns_series.mean() * 252)
                ann_vol = float(returns_series.std() * np.sqrt(252))
                current_dd = float(dd_status.current_drawdown)
                risk_metrics = {
                    "sharpe": float(ann_return / ann_vol) if ann_vol > 0 else 0.0,
                    "max_drawdown": current_dd,
                    "volatility": ann_vol,
                    "calmar": float(ann_return / abs(current_dd)) if current_dd < 0 else 0.0,
                    "win_rate": float(
                        sum(1 for r in returns_series if r > 0) / len(returns_series)
                    ),
                }

        return risk_metrics

    def _position_size_pct(
        self,
        state: Dict,
        strategy_id: str,
        base_position_size_pct: float,
        max_holding_days: int,
        confidence: float,
        regime: int,
        ticker: str,
        as_of: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
        signal_uncertainty: Optional[float] = None,
        regime_entropy: Optional[float] = None,
        drift_score: Optional[float] = None,
    ) -> float:
        """
        Compute entry size percentage using the unified PositionSizer.

        Delegates sizing to the persistent PositionSizer instance for
        consistency with all budget constraints, uncertainty scaling, and
        regime-conditional blend weights.
        """
        base = float(max(0.001, base_position_size_pct))
        if not self.use_kelly_sizing:
            return base

        min_size = max(0.001, base * self.kelly_min_mult)
        max_size = max(min_size, min(0.25, base * self.kelly_max_mult))

        market = self._market_risk_stats(ticker, as_of, price_data)
        if market is None:
            return base

        # Build kelly_history from recent trade returns
        all_trades = state.get("trades", [])
        strat_trades = [t for t in all_trades if str(t.get("strategy_id")) == str(strategy_id)]
        trades = strat_trades if len(strat_trades) >= 20 else all_trades
        trades = trades[-self.kelly_lookback_trades:]
        kelly_history = []
        for t in trades:
            r = self._trade_return(t)
            if r is not None and np.isfinite(r):
                kelly_history.append(float(r))

        # Compute current equity and existing positions
        current_equity = self._mark_to_market(state, as_of, price_data)
        current_positions: Dict[str, float] = {}
        for existing_pos in state.get("positions", []):
            pos_id = self._position_id(existing_pos)
            pos_px = self._current_price(pos_id, as_of, price_data)
            if pos_px is not None:
                current_positions[pos_id] = float(existing_pos["shares"]) * pos_px

        # Configure persistent sizer bounds for this call
        self._persistent_sizer.max_position = max_size
        self._persistent_sizer.min_position = min_size
        self._persistent_sizer.kelly_frac = self.kelly_fraction

        dd_ratio = float(self._dd_controller.update(0.0).current_drawdown) if hasattr(self, '_dd_controller') else 0.0
        # Use the last actual dd status rather than re-updating
        if hasattr(self._dd_controller, 'current_equity') and hasattr(self._dd_controller, 'peak_equity'):
            peak = self._dd_controller.peak_equity
            curr = self._dd_controller.current_equity
            if peak > 1e-12:
                dd_ratio = (curr - peak) / peak

        sized = self._persistent_sizer.size_position_paper_trader(
            ticker=ticker,
            kelly_history=kelly_history,
            atr=float(market["atr"]),
            realized_vol=float(market["realized_vol"]),
            price=float(market["price"]),
            regime_state=self._dd_controller.state.value.upper() if hasattr(self._dd_controller, 'state') else "NORMAL",
            dd_ratio=dd_ratio,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            holding_days=max(1, int(max_holding_days)),
            n_current_positions=len(state.get("positions", [])),
            max_positions=self.max_total_positions,
            portfolio_equity=max(current_equity, 1.0),
            current_positions=current_positions,
            signal_uncertainty=signal_uncertainty,
            regime_entropy=regime_entropy,
            drift_score=drift_score,
        )

        # Apply regime risk multiplier
        regime_mult = float(REGIME_RISK_MULTIPLIER.get(int(regime), 1.0))
        sized = float(sized) * regime_mult
        return float(np.clip(sized, 0.001, max_size))

    def run_cycle(
        self,
        active_strategies: List[ActiveStrategy],
        latest_predictions: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict:
        """Run cycle."""
        as_of = pd.Timestamp(as_of) if as_of is not None else self._resolve_as_of(price_data)
        pred_map = self._latest_predictions_by_id(latest_predictions)
        state = self._load_state()

        # ── A/B test routing ──
        active_ab_test = self._get_active_ab_test()

        # ── Drawdown controller update ──
        current_equity = self._mark_to_market(state, as_of, price_data)
        if self._prev_equity is not None and self._prev_equity > 0:
            daily_pnl = (current_equity - self._prev_equity) / self._prev_equity
        else:
            daily_pnl = 0.0
        dd_status = self._dd_controller.update(daily_pnl)
        self._prev_equity = current_equity

        exits = 0
        entries = 0
        remaining_positions = []

        # Force liquidate all positions if drawdown controller says critical
        if dd_status.force_liquidate:
            for pos in state.get("positions", []):
                permno = self._position_id(pos)
                px = self._current_price(permno, as_of, price_data)
                if px is None:
                    remaining_positions.append(pos)
                    continue
                gross = float(pos["shares"]) * px
                tx = gross * self.tx_cost
                proceeds = gross - tx
                entry_notional = float(pos["shares"]) * float(pos["entry_price"])
                pnl = proceeds - entry_notional
                net_return = float(pnl / entry_notional) if entry_notional > 0 else 0.0
                entry_regime = int(pos.get("entry_regime", pos.get("regime", 2)))
                pred_row = pred_map.get(permno)
                current_regime = int(pred_row.get("regime", entry_regime)) if pred_row else entry_regime
                state["cash"] = float(state.get("cash", 0.0)) + proceeds
                state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
                trade_record = {
                    "strategy_id": pos["strategy_id"],
                    "permno": permno,
                    "ticker": pos.get("ticker"),
                    "entry_date": pos["entry_date"],
                    "exit_date": str(as_of.date()),
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(px),
                    "shares": float(pos["shares"]),
                    "pnl": float(pnl),
                    "net_return": net_return,
                    "holding_days": int(pos.get("holding_days", 0)) + 1,
                    "reason": "drawdown_liquidation",
                    "position_size_pct": float(pos.get("position_size_pct", 0.0)),
                    "entry_regime": entry_regime,
                    "exit_regime": current_regime,
                    "regime_changed": entry_regime != current_regime,
                    "transaction_cost": float(tx),
                }
                state.setdefault("trades", []).append(trade_record)
                # Record to A/B test if position was part of one
                ab_variant = pos.get("ab_variant", "")
                self._record_trade_to_ab_test(
                    active_ab_test, ab_variant, trade_record,
                )
                exits += 1
            state["positions"] = remaining_positions
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            equity = self._mark_to_market(state, as_of, price_data)
            self._prev_equity = equity
            risk_metrics = self._update_equity_curve(
                state, str(as_of.date()), equity, daily_pnl, dd_status,
            )
            self._save_state(state)
            result = {
                "as_of": str(as_of.date()),
                "entries": 0,
                "exits": exits,
                "cash": float(state["cash"]),
                "equity": float(equity),
                "realized_pnl": float(state.get("realized_pnl", 0.0)),
                "open_positions": len(state.get("positions", [])),
                "active_strategies": len(active_strategies),
                "drawdown_state": dd_status.state.value,
            }
            if risk_metrics:
                result["risk_metrics"] = risk_metrics
            return result

        # 1) Evaluate exits — stop losses first, then signal decay / timeout.
        for pos in state.get("positions", []):
            permno = self._position_id(pos)
            px = self._current_price(permno, as_of, price_data)
            if px is None:
                remaining_positions.append(pos)
                continue

            pos["holding_days"] = int(pos.get("holding_days", 0)) + 1
            entry_price = float(pos["entry_price"])

            # ── Track highest price for trailing stop ──
            highest_price = max(
                float(pos.get("highest_price", entry_price)), px
            )
            pos["highest_price"] = highest_price

            # ── Resolve regimes ──
            entry_regime = int(pos.get("entry_regime", pos.get("regime", 2)))
            pred_row = pred_map.get(permno)
            current_regime = (
                int(pred_row.get("regime", entry_regime))
                if pred_row else entry_regime
            )

            # ── Evaluate stop losses FIRST ──
            atr = self._compute_atr(permno, as_of, price_data)
            stop_result = self._stop_mgr.evaluate(
                entry_price=entry_price,
                current_price=px,
                highest_price=highest_price,
                atr=atr,
                bars_held=pos["holding_days"],
                entry_regime=entry_regime,
                current_regime=current_regime,
            )

            exit_reason = None
            if stop_result.should_exit:
                exit_reason = stop_result.reason.value
            else:
                # Secondary: check signal decay and timeout (unchanged logic)
                weak_signal = (
                    pred_row is None
                    or float(pred_row.get("predicted_return", 0.0))
                    < float(pos["entry_threshold"]) * 0.25
                    or float(pred_row.get("confidence", 0.0))
                    < float(pos["confidence_threshold"]) * 0.75
                )
                timed_out = pos["holding_days"] >= int(pos["max_holding_days"])
                if weak_signal:
                    exit_reason = "signal_decay"
                elif timed_out:
                    exit_reason = "time_exit"

            if exit_reason is None:
                remaining_positions.append(pos)
                continue

            # ── Execute exit with Spec 06 execution model ──
            daily_vol = self._get_volume(permno, as_of, price_data)
            market = self._market_risk_stats(permno, as_of, price_data)
            struct_state = self._extract_structural_state(pred_row)
            vol_trend = self._adv_tracker.get_volume_trend(permno) if daily_vol > 0 else None

            exit_fill = self._execution_model.simulate(
                side="sell",
                reference_price=px,
                daily_volume=daily_vol,
                desired_notional_usd=float(pos["shares"]) * px,
                force_full=True,
                realized_vol=market["realized_vol"] if market else None,
                urgency_type="exit",
                break_probability=struct_state["break_probability"],
                structure_uncertainty=struct_state["structure_uncertainty"],
                drift_score=struct_state["drift_score"],
                systemic_stress=struct_state["systemic_stress"],
                volume_trend=vol_trend,
            )

            # SPEC-E04: Record fill for calibration feedback
            self._record_fill_feedback(
                fill=exit_fill,
                reference_price=px,
                side="sell",
                symbol=permno,
                daily_volume=daily_vol,
                price=px,
                regime=current_regime,
                as_of=as_of,
            )

            # Use execution model cost or fall back to flat cost
            if exit_fill.fill_ratio > 0 and EXEC_DYNAMIC_COSTS:
                effective_exit_price = exit_fill.fill_price
                gross = float(pos["shares"]) * effective_exit_price
                tx = 0.0  # costs embedded in fill price
            else:
                gross = float(pos["shares"]) * px
                tx = gross * self.tx_cost
            proceeds = gross - tx
            entry_notional = float(pos["shares"]) * entry_price
            pnl = proceeds - entry_notional
            state["cash"] = float(state.get("cash", 0.0)) + proceeds
            state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
            net_return = float(pnl / entry_notional) if entry_notional > 0 else 0.0
            trade_record = {
                "strategy_id": pos["strategy_id"],
                "permno": permno,
                "ticker": pos.get("ticker"),
                "entry_date": pos["entry_date"],
                "exit_date": str(as_of.date()),
                "entry_price": entry_price,
                "exit_price": float(px),
                "shares": float(pos["shares"]),
                "pnl": float(pnl),
                "net_return": net_return,
                "holding_days": int(pos["holding_days"]),
                "reason": exit_reason,
                "position_size_pct": float(pos.get("position_size_pct", 0.0)),
                "entry_regime": entry_regime,
                "exit_regime": current_regime,
                "regime_changed": entry_regime != current_regime,
                "transaction_cost": float(tx),
            }
            state.setdefault("trades", []).append(trade_record)
            # Record to A/B test if position was part of one
            ab_variant = pos.get("ab_variant", "")
            self._record_trade_to_ab_test(
                active_ab_test, ab_variant, trade_record,
            )
            exits += 1

        state["positions"] = remaining_positions

        # Update regime stats and Bayesian priors on the persistent sizer
        all_trades = state.get("trades", [])
        if len(all_trades) >= 20:
            recent_trades = all_trades[-self.kelly_lookback_trades:]
            trade_df = pd.DataFrame(recent_trades)
            if "net_return" in trade_df.columns:
                # Support both old "regime" and new "entry_regime" trade formats
                regime_col = "entry_regime" if "entry_regime" in trade_df.columns else "regime"
                if regime_col in trade_df.columns:
                    self._persistent_sizer.update_regime_stats(trade_df, regime_col=regime_col)
                    self._persistent_sizer.update_kelly_bayesian(trade_df, regime_col=regime_col)

        # 2) Evaluate entries — skip entirely if drawdown controller blocks new entries.
        if active_strategies and dd_status.allow_new_entries:
            current_equity = self._mark_to_market(state, as_of, price_data)
            held_permnos = {self._position_id(pos) for pos in state.get("positions", [])}
            held_keys = {
                (pos["strategy_id"], self._position_id(pos))
                for pos in state.get("positions", [])
            }

            for strategy in active_strategies:
                params = dict(strategy.params)
                sid = strategy.strategy_id
                entry_threshold = float(params.get("entry_threshold", 0.0))
                confidence_threshold = float(params.get("confidence_threshold", 0.0))
                max_holding_days = int(params.get("horizon", 10))
                base_position_size_pct = float(params.get("position_size_pct", 0.02))

                if len(state["positions"]) >= self.max_total_positions:
                    break

                eligible = latest_predictions[
                    (latest_predictions["predicted_return"] > entry_threshold)
                    & (latest_predictions["confidence"] > confidence_threshold)
                ].sort_values("predicted_return", ascending=False)
                id_col = "permno" if "permno" in eligible.columns else "ticker"

                for _, row in eligible.iterrows():
                    if len(state["positions"]) >= self.max_total_positions:
                        break
                    permno = str(row[id_col])
                    key = (sid, permno)
                    if key in held_keys or permno in held_permnos:
                        continue

                    px = self._current_price(permno, as_of, price_data)
                    if px is None:
                        continue

                    regime = int(row.get("regime", 2))
                    confidence = float(row.get("confidence", 0.5))

                    # ── A/B test variant routing ──
                    ab_variant = ""
                    eff_entry_threshold = entry_threshold
                    eff_confidence_threshold = confidence_threshold
                    eff_max_holding_days = max_holding_days
                    eff_base_size = base_position_size_pct
                    eff_kelly_fraction = self.kelly_fraction

                    if active_ab_test is not None:
                        ticker_label = (
                            str(row.get("ticker", permno))
                            if "ticker" in row.index else permno
                        )
                        ab_variant = active_ab_test.assign_variant(ticker_label)
                        variant_config = active_ab_test.get_variant_config(ab_variant)
                        eff_entry_threshold = float(
                            variant_config.get("entry_threshold", entry_threshold)
                        )
                        eff_confidence_threshold = float(
                            variant_config.get("confidence_threshold", confidence_threshold)
                        )
                        eff_max_holding_days = int(
                            variant_config.get("max_holding_days", max_holding_days)
                        )
                        eff_base_size = float(
                            variant_config.get("position_size_pct", base_position_size_pct)
                        )
                        eff_kelly_fraction = float(
                            variant_config.get("kelly_fraction", self.kelly_fraction)
                        )

                        # Re-check eligibility with variant-specific thresholds
                        pred_return = float(row.get("predicted_return", 0.0))
                        pred_confidence = float(row.get("confidence", 0.0))
                        if (pred_return <= eff_entry_threshold
                                or pred_confidence <= eff_confidence_threshold):
                            continue

                    # Extract uncertainty inputs if present in prediction row
                    sig_unc = float(row.get("signal_uncertainty", np.nan))
                    reg_ent = float(row.get("regime_entropy", np.nan))
                    drft_sc = float(row.get("drift_score", np.nan))

                    position_size_pct = self._position_size_pct(
                        state=state,
                        strategy_id=sid,
                        base_position_size_pct=eff_base_size,
                        max_holding_days=eff_max_holding_days,
                        confidence=confidence,
                        regime=regime,
                        ticker=permno,
                        as_of=as_of,
                        price_data=price_data,
                        signal_uncertainty=sig_unc if np.isfinite(sig_unc) else None,
                        regime_entropy=reg_ent if np.isfinite(reg_ent) else None,
                        drift_score=drft_sc if np.isfinite(drft_sc) else None,
                    )

                    # ── Confidence-weighted position sizing ──
                    confidence_weight = float(row.get("confidence_weight", 1.0))
                    if not np.isfinite(confidence_weight) or confidence_weight < 0.0:
                        confidence_weight = 1.0
                    confidence_weight = min(confidence_weight, 1.0)
                    position_size_pct *= confidence_weight

                    # Apply drawdown controller size multiplier
                    position_size_pct *= dd_status.size_multiplier

                    # Spec 09: Apply health-to-risk feedback gate
                    position_size_pct = self._apply_health_gate(position_size_pct)

                    # ── Portfolio risk check before entry ──
                    portfolio_weights: Dict[str, float] = {}
                    for existing_pos in state.get("positions", []):
                        pos_id = self._position_id(existing_pos)
                        pos_px = self._current_price(pos_id, as_of, price_data)
                        if pos_px is not None and current_equity > 0:
                            pos_value = float(existing_pos["shares"]) * pos_px
                            portfolio_weights[pos_id] = pos_value / current_equity

                    risk_check = self._risk_mgr.check_new_position(
                        ticker=permno,
                        position_size=position_size_pct,
                        current_positions=portfolio_weights,
                        price_data=price_data,
                    )
                    if not risk_check.passed:
                        continue  # Risk check blocked entry

                    notional = current_equity * position_size_pct

                    # Spec 06: simulate entry execution with structural state
                    entry_daily_vol = self._get_volume(permno, as_of, price_data)
                    entry_market = self._market_risk_stats(permno, as_of, price_data)
                    entry_struct = self._extract_structural_state(row.to_dict())
                    entry_vol_trend = (
                        self._adv_tracker.get_volume_trend(permno)
                        if entry_daily_vol > 0 else None
                    )

                    entry_fill = self._execution_model.simulate(
                        side="buy",
                        reference_price=px,
                        daily_volume=entry_daily_vol,
                        desired_notional_usd=notional,
                        force_full=False,
                        realized_vol=entry_market["realized_vol"] if entry_market else None,
                        urgency_type="entry",
                        break_probability=entry_struct["break_probability"],
                        structure_uncertainty=entry_struct["structure_uncertainty"],
                        drift_score=entry_struct["drift_score"],
                        systemic_stress=entry_struct["systemic_stress"],
                        volume_trend=entry_vol_trend,
                    )

                    # No-trade gate or fill rejection
                    if entry_fill.no_trade_blocked or entry_fill.fill_ratio <= 0:
                        continue

                    # SPEC-E04: Record fill for calibration feedback
                    self._record_fill_feedback(
                        fill=entry_fill,
                        reference_price=px,
                        side="buy",
                        symbol=permno,
                        daily_volume=entry_daily_vol,
                        price=px,
                        regime=regime,
                        as_of=as_of,
                    )

                    if EXEC_DYNAMIC_COSTS:
                        effective_entry_price = entry_fill.fill_price
                        filled_notional = notional * entry_fill.fill_ratio
                        tx = 0.0  # costs embedded in fill price
                    else:
                        effective_entry_price = px
                        filled_notional = notional
                        tx = filled_notional * self.tx_cost

                    total_cost = filled_notional + tx
                    if total_cost <= 0 or total_cost > float(state.get("cash", 0.0)):
                        continue

                    shares = filled_notional / effective_entry_price
                    state["cash"] = float(state.get("cash", 0.0)) - total_cost
                    position = {
                        "strategy_id": sid,
                        "permno": permno,
                        "ticker": str(row.get("ticker", "")) if "ticker" in row.index else "",
                        "entry_date": str(as_of.date()),
                        "entry_price": float(effective_entry_price),
                        "shares": float(shares),
                        "holding_days": 0,
                        "max_holding_days": eff_max_holding_days,
                        "entry_threshold": eff_entry_threshold,
                        "confidence_threshold": eff_confidence_threshold,
                        "position_size_pct": float(position_size_pct),
                        "regime": regime,
                        "entry_regime": regime,
                        "confidence": confidence,
                        "highest_price": float(px),
                        "ab_variant": ab_variant,
                    }
                    state["positions"].append(position)
                    held_permnos.add(permno)
                    held_keys.add(key)
                    entries += 1

        state["last_update"] = datetime.now(timezone.utc).isoformat()
        equity = self._mark_to_market(state, as_of, price_data)
        self._prev_equity = equity

        # ── Equity curve tracking ──
        risk_metrics = self._update_equity_curve(
            state, str(as_of.date()), equity, daily_pnl, dd_status,
        )

        self._save_state(state)

        result = {
            "as_of": str(as_of.date()),
            "entries": entries,
            "exits": exits,
            "cash": float(state["cash"]),
            "equity": float(equity),
            "realized_pnl": float(state.get("realized_pnl", 0.0)),
            "open_positions": len(state.get("positions", [])),
            "active_strategies": len(active_strategies),
            "drawdown_state": dd_status.state.value,
        }
        if risk_metrics:
            result["risk_metrics"] = risk_metrics

        # ── A/B test early stopping check ──
        if active_ab_test is not None:
            try:
                early_stop = active_ab_test.check_early_stopping()
                result["ab_test"] = {
                    "test_id": active_ab_test.test_id,
                    "name": active_ab_test.name,
                    "early_stopping": early_stop,
                }
                if early_stop.get("stop"):
                    logger.info(
                        "A/B test %s early stopping triggered: %s",
                        active_ab_test.test_id,
                        early_stop.get("reason", ""),
                    )
            except Exception as e:
                logger.debug("A/B early stopping check failed: %s", e)

        # SPEC-E04: Trigger feedback recalibration if interval has elapsed
        if self._cost_calibrator is not None:
            try:
                feedback_result = (
                    self._cost_calibrator.run_feedback_recalibration()
                )
                if feedback_result:
                    result["cost_feedback_recalibration"] = feedback_result
                    logger.info(
                        "SPEC-E04: feedback recalibration updated %d segments",
                        len(feedback_result),
                    )
            except Exception as e:
                logger.warning(
                    "SPEC-E04: feedback recalibration failed: %s", e
                )

        return result
