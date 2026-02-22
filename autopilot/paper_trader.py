"""
Stateful paper-trading engine for promoted strategies.
"""
import json
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
)
from ..risk.position_sizer import PositionSizer
from .registry import ActiveStrategy


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
    ) -> float:
        """
        Compute entry size percentage. Uses bounded fractional-Kelly when enabled.
        """
        base = float(max(0.001, base_position_size_pct))
        if not self.use_kelly_sizing:
            return base

        min_size = max(0.001, base * self.kelly_min_mult)
        max_size = max(min_size, min(0.25, base * self.kelly_max_mult))

        trade_stats = self._historical_trade_stats(state, strategy_id)
        market = self._market_risk_stats(ticker, as_of, price_data)
        if market is None:
            return base

        sizer = PositionSizer(
            max_position_pct=max_size,
            min_position_pct=min_size,
            kelly_fraction=self.kelly_fraction,
        )
        ps = sizer.size_position(
            ticker=ticker,
            win_rate=float(trade_stats[0]),
            avg_win=float(trade_stats[1]),
            avg_loss=float(trade_stats[2]),
            realized_vol=float(market["realized_vol"]),
            atr=float(market["atr"]),
            price=float(market["price"]),
            holding_days=max(1, int(max_holding_days)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            n_current_positions=len(state.get("positions", [])),
            max_positions=self.max_total_positions,
        )

        regime_mult = float(REGIME_RISK_MULTIPLIER.get(int(regime), 1.0))
        sized = float(ps.final_size) * regime_mult
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

        exits = 0
        entries = 0
        remaining_positions = []

        # 1) Evaluate exits.
        for pos in state.get("positions", []):
            permno = self._position_id(pos)
            px = self._current_price(permno, as_of, price_data)
            if px is None:
                remaining_positions.append(pos)
                continue

            pos["holding_days"] = int(pos.get("holding_days", 0)) + 1
            pred_row = pred_map.get(permno)
            weak_signal = (
                pred_row is None
                or float(pred_row.get("predicted_return", 0.0)) < float(pos["entry_threshold"]) * 0.25
                or float(pred_row.get("confidence", 0.0)) < float(pos["confidence_threshold"]) * 0.75
            )
            timed_out = pos["holding_days"] >= int(pos["max_holding_days"])
            if not (weak_signal or timed_out):
                remaining_positions.append(pos)
                continue

            gross = float(pos["shares"]) * px
            tx = gross * self.tx_cost
            proceeds = gross - tx
            entry_notional = float(pos["shares"]) * float(pos["entry_price"])
            pnl = proceeds - entry_notional
            state["cash"] = float(state.get("cash", 0.0)) + proceeds
            state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
            state.setdefault("trades", []).append(
                {
                    "strategy_id": pos["strategy_id"],
                    "permno": permno,
                    "ticker": pos.get("ticker"),
                    "entry_date": pos["entry_date"],
                    "exit_date": str(as_of.date()),
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(px),
                    "shares": float(pos["shares"]),
                    "pnl": float(pnl),
                    "holding_days": int(pos["holding_days"]),
                    "reason": "time_exit" if timed_out else "signal_decay",
                    "position_size_pct": float(pos.get("position_size_pct", 0.0)),
                },
            )
            exits += 1

        state["positions"] = remaining_positions

        # 2) Evaluate entries.
        if active_strategies:
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

                    position_size_pct = self._position_size_pct(
                        state=state,
                        strategy_id=sid,
                        base_position_size_pct=base_position_size_pct,
                        max_holding_days=max_holding_days,
                        confidence=float(row.get("confidence", 0.5)),
                        regime=int(row.get("regime", 2)),
                        ticker=permno,
                        as_of=as_of,
                        price_data=price_data,
                    )

                    # ── Confidence-weighted position sizing (NEW 3) ──
                    # Scale the computed position size by the calibrated
                    # confidence score.  High-confidence predictions keep
                    # full size; low-confidence predictions get reduced
                    # exposure.  Defaults to 1.0 when not available.
                    confidence_weight = float(row.get("confidence_weight", 1.0))
                    if not np.isfinite(confidence_weight) or confidence_weight < 0.0:
                        confidence_weight = 1.0
                    confidence_weight = min(confidence_weight, 1.0)
                    position_size_pct *= confidence_weight

                    notional = current_equity * position_size_pct
                    tx = notional * self.tx_cost
                    total_cost = notional + tx
                    if total_cost <= 0 or total_cost > float(state.get("cash", 0.0)):
                        continue

                    shares = notional / px
                    state["cash"] = float(state.get("cash", 0.0)) - total_cost
                    position = {
                        "strategy_id": sid,
                        "permno": permno,
                        "ticker": str(row.get("ticker", "")) if "ticker" in row.index else "",
                        "entry_date": str(as_of.date()),
                        "entry_price": float(px),
                        "shares": float(shares),
                        "holding_days": 0,
                        "max_holding_days": max_holding_days,
                        "entry_threshold": entry_threshold,
                        "confidence_threshold": confidence_threshold,
                        "position_size_pct": float(position_size_pct),
                    }
                    state["positions"].append(position)
                    held_permnos.add(permno)
                    held_keys.add(key)
                    entries += 1

        state["last_update"] = datetime.now(timezone.utc).isoformat()
        equity = self._mark_to_market(state, as_of, price_data)
        self._save_state(state)

        return {
            "as_of": str(as_of.date()),
            "entries": entries,
            "exits": exits,
            "cash": float(state["cash"]),
            "equity": float(equity),
            "realized_pnl": float(state.get("realized_pnl", 0.0)),
            "open_positions": len(state.get("positions", [])),
            "active_strategies": len(active_strategies),
        }
