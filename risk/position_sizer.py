"""
Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.

Implements multiple sizing approaches that can be blended:
    - Kelly criterion (half-Kelly for conservatism)
    - Volatility targeting (inverse vol scaling)
    - ATR-based (risk per unit of volatility)
    - Composite blend of all methods
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    ticker: str  # PERMNO in PERMNO-first mode (legacy field name kept for compatibility)
    raw_kelly: float        # Full Kelly fraction
    half_kelly: float       # Half-Kelly (recommended)
    vol_scaled: float       # Volatility-targeted size
    atr_based: float        # ATR-based size
    composite: float        # Blended final size
    max_allowed: float      # Position limit
    final_size: float       # min(composite, max_allowed)
    sizing_details: dict


class PositionSizer:
    """
    Multi-method position sizer with conservative blending.

    Kelly Criterion:
        f* = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = 1 - p
        Use half-Kelly for practical conservatism.

    Volatility Targeting:
        size = target_portfolio_vol / (realized_vol * sqrt(252/holding_days))
        Ensures each position contributes roughly equally to portfolio risk.

    ATR-Based:
        size = max_risk_per_trade / (N_atr * ATR / price)
        Scales position size to risk a fixed fraction of capital.
    """

    def __init__(
        self,
        target_portfolio_vol: float = 0.15,     # 15% annual portfolio vol target
        max_position_pct: float = 0.10,          # 10% max per position
        min_position_pct: float = 0.01,          # 1% min per position
        max_risk_per_trade: float = 0.02,        # 2% max risk per trade
        kelly_fraction: float = 0.5,             # Half-Kelly
        atr_multiplier: float = 2.0,             # ATR stop distance
        blend_weights: Optional[dict] = None,    # Method blend weights
    ):
        """Initialize PositionSizer."""
        self.target_vol = target_portfolio_vol
        self.max_position = max_position_pct
        self.min_position = min_position_pct
        self.max_risk = max_risk_per_trade
        self.kelly_frac = kelly_fraction
        self.atr_mult = atr_multiplier
        self.blend_weights = blend_weights or {
            "kelly": 0.3,
            "vol_scaled": 0.4,
            "atr_based": 0.3,
        }

    def size_position(
        self,
        ticker: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        realized_vol: float,
        atr: float,
        price: float,
        holding_days: int = 10,
        confidence: float = 0.5,
        n_current_positions: int = 0,
        max_positions: int = 20,
    ) -> PositionSize:
        """
        Calculate position size using multiple methods and blend.

        Args:
            ticker: security id (PERMNO in PERMNO-first mode)
            win_rate: historical win rate (0-1)
            avg_win: average winning trade return
            avg_loss: average losing trade return (negative)
            realized_vol: annualized realized volatility
            atr: current ATR value
            price: current price
            holding_days: expected holding period
            confidence: model confidence (0-1)
            n_current_positions: number of existing positions
            max_positions: maximum total positions
        """
        # ── Kelly Criterion ──
        raw_kelly = self._kelly(win_rate, avg_win, avg_loss)
        half_kelly = raw_kelly * self.kelly_frac

        # ── Volatility-scaled ──
        vol_scaled = self._vol_scaled(realized_vol, holding_days, max_positions)

        # ── ATR-based ──
        atr_based = self._atr_based(atr, price)

        # ── Composite blend ──
        composite = (
            self.blend_weights["kelly"] * half_kelly
            + self.blend_weights["vol_scaled"] * vol_scaled
            + self.blend_weights["atr_based"] * atr_based
        )

        # Scale by model confidence (0.5 = no scaling, higher = boost)
        confidence_scalar = 0.5 + confidence  # range: 0.5 to 1.5
        composite *= confidence_scalar

        # Apply position limits
        max_allowed = self.max_position
        # Reduce size if near max positions
        if n_current_positions > 0 and max_positions > 0:
            capacity_ratio = 1 - (n_current_positions / max_positions)
            max_allowed = min(max_allowed, max(self.min_position, capacity_ratio * self.max_position))

        final_size = np.clip(composite, self.min_position, max_allowed)

        return PositionSize(
            ticker=ticker,
            raw_kelly=float(raw_kelly),
            half_kelly=float(half_kelly),
            vol_scaled=float(vol_scaled),
            atr_based=float(atr_based),
            composite=float(composite),
            max_allowed=float(max_allowed),
            final_size=float(final_size),
            sizing_details={
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "realized_vol": realized_vol,
                "atr": atr,
                "confidence": confidence,
                "confidence_scalar": confidence_scalar,
            },
        )

    def _kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly criterion: f* = (p*b - q) / b

        where p = win probability, b = win/loss ratio, q = 1-p
        """
        if avg_loss >= -1e-9 or win_rate <= 0 or win_rate >= 1:
            return self.min_position

        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss)  # win/loss ratio

        kelly = (p * b - q) / b

        # Clamp: Kelly can go negative (don't bet) or very large
        if kelly <= 0:
            return self.min_position
        return min(kelly, self.max_position)

    def _vol_scaled(
        self, realized_vol: float, holding_days: int, max_positions: int,
    ) -> float:
        """
        Volatility-targeted sizing.

        size = target_vol / (asset_vol * sqrt(holding_period_fraction) * sqrt(n_positions))
        """
        if realized_vol <= 0:
            return self.min_position

        # Holding period volatility
        holding_vol = realized_vol * np.sqrt(holding_days / 252)

        # Diversification factor: spread risk across positions
        n_pos_factor = np.sqrt(max(1, max_positions))

        size = self.target_vol / (holding_vol * n_pos_factor)

        return np.clip(size, self.min_position, self.max_position)

    def _atr_based(self, atr: float, price: float) -> float:
        """
        ATR-based sizing: risk a fixed fraction of capital.

        size = max_risk / (atr_multiplier * atr / price)
        """
        if atr <= 0 or price <= 0:
            return self.min_position

        # Risk per share as % of price
        risk_per_share_pct = (self.atr_mult * atr) / price

        if risk_per_share_pct <= 0:
            return self.min_position

        size = self.max_risk / risk_per_share_pct

        return np.clip(size, self.min_position, self.max_position)

    def size_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: dict,
        trade_history: Optional[pd.DataFrame] = None,
        max_positions: int = 20,
    ) -> pd.DataFrame:
        """
        Size all candidate positions for the portfolio.

        Args:
            signals: DataFrame with columns [permno|ticker, predicted_return, confidence, regime]
            price_data: dict of security-id -> OHLCV DataFrames
            trade_history: historical trades for win rate estimation
            max_positions: max simultaneous positions

        Returns:
            DataFrame with position sizes added
        """
        # Compute historical win rate and avg win/loss from trade history
        if trade_history is not None and len(trade_history) > 10:
            hist_returns = trade_history["net_return"].values
            win_rate = (hist_returns > 0).mean()
            avg_win = hist_returns[hist_returns > 0].mean() if (hist_returns > 0).any() else 0.01
            avg_loss = hist_returns[hist_returns < 0].mean() if (hist_returns < 0).any() else -0.01
        else:
            # Conservative defaults
            win_rate = 0.50
            avg_win = 0.02
            avg_loss = -0.02

        sizes = []
        id_col = "permno" if "permno" in signals.columns else "ticker"
        for _, row in signals.iterrows():
            ticker = str(row[id_col])
            if ticker not in price_data:
                continue

            ohlcv = price_data[ticker]
            if len(ohlcv) < 20:
                continue

            # Current ATR (14-period)
            high = ohlcv["High"].iloc[-14:]
            low = ohlcv["Low"].iloc[-14:]
            close = ohlcv["Close"].iloc[-15:]  # need one extra for prev close
            tr = pd.concat([
                high - low,
                (high - close.shift(1).iloc[-14:]).abs(),
                (low - close.shift(1).iloc[-14:]).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.mean())

            # Realized vol (annualized, 20-day)
            returns = ohlcv["Close"].pct_change().iloc[-20:]
            realized_vol = float(returns.std() * np.sqrt(252))

            price = float(ohlcv["Close"].iloc[-1])

            ps = self.size_position(
                ticker=ticker,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                realized_vol=realized_vol,
                atr=atr,
                price=price,
                confidence=float(row.get("confidence", 0.5)),
                n_current_positions=len(sizes),
                max_positions=max_positions,
            )
            sizes.append({
                "permno": ticker,
                "position_size": ps.final_size,
                "kelly_size": ps.half_kelly,
                "vol_size": ps.vol_scaled,
                "atr_size": ps.atr_based,
            })

        return pd.DataFrame(sizes)
