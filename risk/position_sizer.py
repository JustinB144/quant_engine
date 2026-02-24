"""
Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.

Implements multiple sizing approaches that can be blended:
    - Kelly criterion (half-Kelly for conservatism)
    - Volatility targeting (inverse vol scaling)
    - ATR-based (risk per unit of volatility)
    - Composite blend of all methods
    - Regime-conditional Kelly (adjusts parameters by market regime)
    - Drawdown governor (convex exponential curve reduces sizing as drawdown increases)
    - Bayesian updating (Beta-Binomial conjugate prior for win rate)
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
        max_portfolio_dd: float = 0.20,          # Max allowed portfolio drawdown
        bayesian_alpha: float = 2.0,             # Beta prior alpha (wins + 1)
        bayesian_beta: float = 2.0,              # Beta prior beta (losses + 1)
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
        self.max_portfolio_dd = max_portfolio_dd

        # Bayesian win-rate prior (Beta-Binomial conjugate)
        self.bayesian_alpha = bayesian_alpha
        self.bayesian_beta = bayesian_beta
        # Global Bayesian counters (fallback for cold-start)
        self._bayesian_wins = 0
        self._bayesian_losses = 0
        # Per-regime Bayesian counters: {regime_id (int): {wins, losses}}
        self._bayesian_regime: Dict[int, Dict[str, int]] = {
            0: {"wins": 0, "losses": 0},  # trending_bull
            1: {"wins": 0, "losses": 0},  # trending_bear
            2: {"wins": 0, "losses": 0},  # mean_reverting
            3: {"wins": 0, "losses": 0},  # high_volatility
        }

        # Regime-conditional historical statistics
        # Keys: regime label (str), Values: {"win_rate", "avg_win", "avg_loss", "n_trades"}
        self.regime_stats: Dict[str, Dict[str, float]] = {
            "trending_bull": {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": -0.02, "n_trades": 0},
            "trending_bear": {"win_rate": 0.45, "avg_win": 0.02, "avg_loss": -0.025, "n_trades": 0},
            "mean_reverting": {"win_rate": 0.50, "avg_win": 0.02, "avg_loss": -0.02, "n_trades": 0},
            "high_volatility": {"win_rate": 0.42, "avg_win": 0.04, "avg_loss": -0.035, "n_trades": 0},
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
        regime: Optional[str] = None,
        current_drawdown: float = 0.0,
        n_trades: int = 100,
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
            regime: current market regime label (e.g. "trending_bull")
            current_drawdown: current portfolio drawdown as negative fraction (e.g. -0.05)
            n_trades: number of historical trades (used for small-sample penalty in Kelly)
        """
        # Apply regime-conditional adjustments if regime is known
        if regime is not None and regime in self.regime_stats:
            rs = self.regime_stats[regime]
            if rs["n_trades"] >= 20:
                # Use regime-specific stats when we have enough data
                win_rate = rs["win_rate"]
                avg_win = rs["avg_win"]
                avg_loss = rs["avg_loss"]

        # ── Kelly Criterion ──
        self._current_n_trades = n_trades
        raw_kelly = self._kelly(win_rate, avg_win, avg_loss)
        half_kelly = raw_kelly * self.kelly_frac

        # Apply drawdown governor to Kelly
        half_kelly = self._apply_drawdown_governor(half_kelly, current_drawdown, self.max_portfolio_dd)

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

        # Confidence scales position DOWN from blend, never UP.
        # Range: [0.5, 1.0] — full confidence = 100% of blend, low confidence = 50%.
        confidence_scalar = 0.5 + 0.5 * min(1.0, max(0.0, confidence))
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

        Returns 0.0 for negative-edge signals (not min_position) — refuse to
        allocate capital when the expected edge is non-positive.  A small-sample
        penalty shrinks the Kelly fraction toward zero when ``_current_n_trades``
        is low, reaching full Kelly only after 50 trades.
        """
        if avg_loss >= -1e-9 or win_rate <= 0.0 or win_rate >= 1.0:
            return 0.0  # Invalid inputs — refuse to size

        p = win_rate
        q = 1.0 - p
        b = abs(avg_win / avg_loss)  # win/loss ratio

        kelly = (p * b - q) / b

        if kelly <= 0:
            return 0.0  # Negative edge — do not trade

        # Small-sample penalty: reduce Kelly when n_trades is low.
        # Shrink toward 0 as n_trades → 0; full Kelly only after 50 trades.
        n_trades = getattr(self, '_current_n_trades', 100)
        sample_penalty = min(1.0, n_trades / 50.0)
        kelly *= sample_penalty

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

    def _apply_drawdown_governor(
        self, kelly_fraction: float, current_drawdown: float, max_allowed_dd: float = -0.20,
    ) -> float:
        """Reduce Kelly as drawdown approaches max using a convex (exponential) curve.

        Uses ``scale = exp(-k * dd_ratio²)`` instead of linear ``1 - dd_ratio``.
        The squared term makes the curve more lenient early in the drawdown and
        more aggressive near the limit:

            At 50% of max_dd → sizing ≈ 61% (vs 50% linear)
            At 90% of max_dd → sizing ≈ 20% (vs 10% linear)

        Returns 0.0 if drawdown exceeds the maximum allowed.
        """
        if current_drawdown >= 0.0:
            return kelly_fraction  # No drawdown — full sizing

        dd_ratio = abs(current_drawdown) / abs(max_allowed_dd)

        if dd_ratio >= 1.0:
            return 0.0  # Breached max drawdown — no new positions

        # Convex scaling: slow reduction early, aggressive near limit.
        # exp(-k * dd_ratio²) with k=2 gives:
        #   dd_ratio=0.5 → scale≈0.61 (lenient early)
        #   dd_ratio=0.9 → scale≈0.20 (aggressive late)
        k = 2.0
        scale = math.exp(-k * dd_ratio * dd_ratio)

        return kelly_fraction * scale

    def update_regime_stats(self, trades: pd.DataFrame, regime_col: str = "regime"):
        """Update regime-specific historical stats from completed trades.

        Accepts a full trade DataFrame with a ``regime`` column (integer IDs
        0-3) and updates stats for every regime that has enough data. This
        replaces the old interface that required pre-filtering by regime.

        Args:
            trades: DataFrame with ``net_return`` and ``regime`` columns.
            regime_col: column name containing integer regime IDs.
        """
        from ..config import REGIME_NAMES

        if trades.empty or "net_return" not in trades.columns:
            return
        if regime_col not in trades.columns:
            return

        for regime_id, regime_name in REGIME_NAMES.items():
            regime_trades = trades[trades[regime_col] == regime_id]
            returns = regime_trades["net_return"].dropna().values
            if len(returns) < 5:
                continue
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            self.regime_stats[regime_name] = {
                "win_rate": float((returns > 0).mean()),
                "avg_win": float(wins.mean()) if len(wins) > 0 else 0.01,
                "avg_loss": float(losses.mean()) if len(losses) > 0 else -0.01,
                "n_trades": len(returns),
            }

    def update_kelly_bayesian(self, trades: pd.DataFrame, regime_col: str = "regime"):
        """Update per-regime and global Bayesian posteriors from trade history.

        Maintains separate Beta-Binomial priors for each regime so that
        bull-market wins don't inflate bear-market posteriors.

        Args:
            trades: DataFrame with ``net_return`` column and optionally a
                ``regime`` column (integer regime IDs 0-3).
            regime_col: column name containing regime labels.
        """
        if trades.empty or "net_return" not in trades.columns:
            return

        returns = trades["net_return"].dropna()

        # Update global counters
        self._bayesian_wins = int((returns > 0).sum())
        self._bayesian_losses = int((returns <= 0).sum())

        # Update per-regime counters
        if regime_col in trades.columns:
            for regime_id in self._bayesian_regime:
                regime_mask = trades[regime_col] == regime_id
                regime_returns = trades.loc[regime_mask, "net_return"].dropna()
                if len(regime_returns) > 0:
                    wins = int((regime_returns > 0).sum())
                    losses = int((regime_returns <= 0).sum())
                    self._bayesian_regime[regime_id]["wins"] += wins
                    self._bayesian_regime[regime_id]["losses"] += losses

    def get_bayesian_kelly(
        self, avg_win: float = 0.02, avg_loss: float = -0.02, regime: Optional[int] = None,
    ) -> float:
        """Compute Kelly fraction using regime-specific Bayesian posterior.

        When a ``regime`` is specified and sufficient regime-specific data is
        available (>=10 trades), the posterior for that regime is used.
        Otherwise falls back to the global posterior.

        Args:
            avg_win: average winning trade return.
            avg_loss: average losing trade return (negative).
            regime: integer regime ID (0-3) or ``None`` for global.

        Returns:
            Half-Kelly fraction from Bayesian posterior win rate.
        """
        # Try regime-specific posterior first
        if regime is not None and regime in self._bayesian_regime:
            stats = self._bayesian_regime[regime]
            wins, losses = stats["wins"], stats["losses"]
            if wins + losses >= 10:
                posterior_wr = (self.bayesian_alpha + wins) / (
                    self.bayesian_alpha + self.bayesian_beta + wins + losses
                )
                return self._kelly(posterior_wr, avg_win, avg_loss) * self.kelly_frac

        # Fallback to global posterior
        alpha = self.bayesian_alpha + self._bayesian_wins
        beta = self.bayesian_beta + self._bayesian_losses
        posterior_win_rate = alpha / (alpha + beta)
        raw_kelly = self._kelly(posterior_win_rate, avg_win, avg_loss)
        return raw_kelly * self.kelly_frac

    def size_portfolio_aware(
        self,
        ticker: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        realized_vol: float,
        atr: float,
        price: float,
        existing_positions: Dict[str, float],
        returns_data: Optional[Dict[str, pd.Series]] = None,
        **kwargs,
    ) -> float:
        """Size position considering correlation with existing portfolio.

        Wraps :meth:`size_position` and applies a correlation penalty:
        positions that are highly correlated with the existing portfolio
        receive reduced allocation.

        Args:
            ticker: security identifier.
            win_rate: historical win rate (0-1).
            avg_win: average winning trade return.
            avg_loss: average losing trade return (negative).
            realized_vol: annualized realized volatility.
            atr: current ATR value.
            price: current price.
            existing_positions: ``{ticker: weight}`` of current portfolio.
            returns_data: ``{ticker: pd.Series}`` of return series for
                correlation computation.
            **kwargs: forwarded to :meth:`size_position`.

        Returns:
            Adjusted position size as a fraction of portfolio.
        """
        base_result = self.size_position(
            ticker=ticker,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            realized_vol=realized_vol,
            atr=atr,
            price=price,
            **kwargs,
        )
        base_size = base_result.final_size

        if not existing_positions or returns_data is None:
            return base_size

        ticker_returns = returns_data.get(ticker)
        if ticker_returns is None or len(ticker_returns) < 20:
            return base_size

        # Compute weight-adjusted average correlation with existing positions
        correlations = []
        total_abs_weight = 0.0
        for pos_ticker, pos_weight in existing_positions.items():
            pos_returns = returns_data.get(pos_ticker)
            if pos_returns is None:
                continue
            # Align series on shared index
            aligned = pd.concat([ticker_returns, pos_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if not np.isfinite(corr):
                continue
            correlations.append(corr * abs(pos_weight))
            total_abs_weight += abs(pos_weight)

        if not correlations or total_abs_weight < 1e-9:
            return base_size

        # Weighted average correlation with portfolio
        avg_corr = sum(correlations) / total_abs_weight

        # Reduce size when highly correlated with existing positions.
        # corr=0 → full size, corr=1 → 50% size, corr<0 → bonus (capped at 100%)
        correlation_penalty = 1.0 - 0.5 * max(0.0, avg_corr)

        return base_size * correlation_penalty

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
