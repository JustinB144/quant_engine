"""
Test module for paper trader integration — spec 015.

Tests that DrawdownController, StopLossManager, PortfolioRiskManager, regime
tracking, equity curve, and risk metrics are all properly wired into the
paper trader's run_cycle.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_engine.autopilot.paper_trader import PaperTrader
from quant_engine.autopilot.registry import ActiveStrategy
from quant_engine.risk.drawdown import DrawdownState


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_price_data(
    tickers: dict,
    n_bars: int = 60,
    start_date: str = "2024-01-01",
) -> dict:
    """Create mock OHLCV price data.

    Args:
        tickers: {ticker_id: (base_price, final_price, daily_range_pct)}
            If final_price is None, price is constant at base_price.
            daily_range_pct controls High-Low spread (affects ATR).
    """
    idx = pd.bdate_range(start_date, periods=n_bars)
    data = {}
    for ticker_id, (base_price, final_price, range_pct) in tickers.items():
        if final_price is None:
            close = np.full(n_bars, base_price)
        else:
            close = np.linspace(base_price, final_price, n_bars)
        high = close * (1 + range_pct / 2)
        low = close * (1 - range_pct / 2)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.full(n_bars, 5_000_000.0),
            },
            index=idx,
        )
        data[ticker_id] = df
    return data


def _make_strategy(
    sid: str = "s1",
    entry_threshold: float = 0.001,
    confidence_threshold: float = 0.30,
    horizon: int = 10,
    position_size_pct: float = 0.05,
) -> ActiveStrategy:
    """Create an ActiveStrategy with standard test parameters."""
    return ActiveStrategy(
        strategy_id=sid,
        promoted_at="2026-01-01T00:00:00",
        params={
            "entry_threshold": entry_threshold,
            "confidence_threshold": confidence_threshold,
            "horizon": horizon,
            "position_size_pct": position_size_pct,
        },
        score=1.0,
        metrics={},
    )


def _make_predictions(rows: list) -> pd.DataFrame:
    """Create a predictions DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


def _seed_state(path: Path, state: dict):
    """Write a state dict to the paper state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, default=str)


def _load_state(path: Path) -> dict:
    """Read the paper state file."""
    with open(path, "r") as f:
        return json.load(f)


def _seed_trades(n: int = 60) -> list:
    """Create deterministic trade history for Kelly sizing."""
    trades = []
    for i in range(n):
        pnl = 120.0 if i % 5 != 0 else -45.0
        trades.append(
            {
                "strategy_id": "s1",
                "permno": "AAPL",
                "ticker": "AAPL",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "shares": 50.0,
                "pnl": pnl,
                "net_return": pnl / 5000.0,
                "holding_days": 5,
                "reason": "time_exit",
                "entry_regime": 0,
                "exit_regime": 0,
                "regime_changed": False,
            }
        )
    return trades


def _base_position(
    permno: str = "AAPL",
    entry_price: float = 100.0,
    shares: float = 50.0,
    holding_days: int = 3,
    regime: int = 0,
    highest_price: float = 100.0,
    max_holding_days: int = 30,
) -> dict:
    """Create a standard position record for seeding state."""
    return {
        "strategy_id": "s1",
        "permno": permno,
        "ticker": permno,
        "entry_date": "2024-02-01",
        "entry_price": entry_price,
        "shares": shares,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "entry_threshold": 0.001,
        "confidence_threshold": 0.30,
        "position_size_pct": 0.05,
        "regime": regime,
        "entry_regime": regime,
        "confidence": 0.90,
        "highest_price": highest_price,
    }


# ── Test Classes ─────────────────────────────────────────────────────────


class TestDrawdownBlocksEntries(unittest.TestCase):
    """After significant losses, new entries are blocked."""

    def test_drawdown_blocks_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"
            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )

            # Manipulate drawdown controller to CAUTION state (-12% drawdown).
            # This simulates accumulated losses over multiple prior cycles.
            trader._dd_controller.peak_equity = 100_000
            trader._dd_controller.current_equity = 88_000
            # Set prev_equity to match current mark-to-market (100k cash, no
            # positions) so daily_pnl = 0 and DD state persists.
            trader._prev_equity = 100_000

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )
            strategy = _make_strategy(position_size_pct=0.05)

            result = trader.run_cycle([strategy], preds, price_data)

            self.assertEqual(result["entries"], 0, "Entries should be blocked in CAUTION")
            self.assertNotEqual(result["drawdown_state"], "normal")


class TestDrawdownForceLiquidate(unittest.TestCase):
    """Critical drawdown forces liquidation of all positions."""

    def test_drawdown_force_liquidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Seed state with an open position
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [_base_position()],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )

            # Push drawdown controller past critical threshold (-17% drawdown)
            trader._dd_controller.peak_equity = 100_000
            trader._dd_controller.current_equity = 83_000
            trader._prev_equity = 100_000

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )
            strategy = _make_strategy()

            result = trader.run_cycle([strategy], preds, price_data)

            self.assertGreater(result["exits"], 0, "Positions should be force liquidated")
            self.assertEqual(result["open_positions"], 0, "No positions should remain")
            self.assertEqual(result["drawdown_state"], "critical")

            saved = _load_state(state_path)
            reasons = [t["reason"] for t in saved.get("trades", [])]
            self.assertIn("drawdown_liquidation", reasons)


class TestStopLossTriggersExit(unittest.TestCase):
    """Hard stop at -8% triggers exit."""

    def test_hard_stop_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Seed position at $100
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [_base_position(entry_price=100.0)],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )
            trader._prev_equity = 100_000.0

            # Price drops to $91 → -9% unrealized (hard stop threshold is -8%)
            price_data = _make_price_data({"AAPL": (100.0, 91.0, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )
            strategy = _make_strategy()

            result = trader.run_cycle([strategy], preds, price_data)

            self.assertEqual(result["exits"], 1)

            saved = _load_state(state_path)
            trades = saved.get("trades", [])
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["reason"], "hard_stop")


class TestTrailingStopTriggers(unittest.TestCase):
    """Trailing stop fires after position was in profit then pulls back."""

    def test_trailing_stop_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Position at $100, ran up to $108 (highest_price), now dropping.
            # With ATR ~2 (from 2% daily range), trailing stop = 108 - 1.5*2 = $105.
            # Current price at $103 (still +3% from entry, so trailing is active).
            # $103 < $105 → trailing stop fires.
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [
                    _base_position(
                        entry_price=100.0,
                        highest_price=108.0,
                        holding_days=8,
                    )
                ],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )
            trader._prev_equity = 100_000.0

            # Price at $103 (declined from $108 peak)
            price_data = _make_price_data({"AAPL": (100.0, 103.0, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )
            strategy = _make_strategy()

            result = trader.run_cycle([strategy], preds, price_data)

            self.assertEqual(result["exits"], 1)

            saved = _load_state(state_path)
            trades = saved.get("trades", [])
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["reason"], "trailing_stop")


class TestRiskManagerBlocksCorrelated(unittest.TestCase):
    """Highly correlated position rejected by PortfolioRiskManager."""

    def test_correlated_entry_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Seed state with an existing AAPL position
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [_base_position(permno="AAPL", holding_days=1)],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )
            trader._prev_equity = 100_000.0

            # Create two perfectly correlated stocks (same return stream)
            n_bars = 100
            idx = pd.bdate_range("2024-01-01", periods=n_bars)
            rng = np.random.default_rng(42)
            returns = rng.normal(0.001, 0.02, n_bars)
            close_a = 100.0 * np.cumprod(1 + returns)
            close_b = 50.0 * np.cumprod(1 + returns)

            price_data = {}
            for ticker_id, close in [("AAPL", close_a), ("MSFT", close_b)]:
                price_data[ticker_id] = pd.DataFrame(
                    {
                        "Open": close,
                        "High": close * 1.005,
                        "Low": close * 0.995,
                        "Close": close,
                        "Volume": np.full(n_bars, 5_000_000.0),
                    },
                    index=idx,
                )

            # MSFT is the only new candidate (AAPL already held)
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    },
                    {
                        "ticker": "MSFT",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    },
                ]
            )
            strategy = _make_strategy(position_size_pct=0.05)

            result = trader.run_cycle([strategy], preds, price_data)

            # MSFT should be blocked by correlation check (corr ≈ 1.0 > 0.85)
            saved = _load_state(state_path)
            position_ids = [p["permno"] for p in saved["positions"]]
            self.assertIn("AAPL", position_ids)
            self.assertNotIn("MSFT", position_ids)


class TestRegimeStoredInPosition(unittest.TestCase):
    """Position record has entry_regime field."""

    def test_regime_stored_in_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"
            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 1,  # trending_bear
                    }
                ]
            )
            strategy = _make_strategy()

            result = trader.run_cycle([strategy], preds, price_data)
            self.assertEqual(result["entries"], 1)

            saved = _load_state(state_path)
            pos = saved["positions"][0]

            self.assertIn("entry_regime", pos)
            self.assertEqual(pos["entry_regime"], 1)
            self.assertIn("confidence", pos)
            self.assertAlmostEqual(pos["confidence"], 0.90)
            self.assertIn("highest_price", pos)


class TestRegimeStoredInTrade(unittest.TestCase):
    """Trade record has entry_regime, exit_regime, and regime_changed."""

    def test_regime_stored_in_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Seed position with regime=1 that will time out
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [
                    _base_position(
                        regime=1,
                        holding_days=29,
                        max_holding_days=30,
                    )
                ],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )
            trader._prev_equity = 100_000.0

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            # Current prediction has regime=2 (different from entry regime=1)
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 2,
                    }
                ]
            )
            strategy = _make_strategy()

            result = trader.run_cycle([strategy], preds, price_data)
            self.assertGreater(result["exits"], 1 - 1)  # at least 1 exit

            saved = _load_state(state_path)
            trades = saved.get("trades", [])
            self.assertGreater(len(trades), 0)

            trade = trades[0]
            self.assertIn("entry_regime", trade)
            self.assertIn("exit_regime", trade)
            self.assertIn("regime_changed", trade)
            self.assertEqual(trade["entry_regime"], 1)
            self.assertEqual(trade["exit_regime"], 2)
            self.assertTrue(trade["regime_changed"])


class TestRegimePassedToSizer(unittest.TestCase):
    """Position sizer receives regime from prediction data."""

    def test_regime_affects_position_size(self):
        """With Kelly enabled, different regimes produce different sizes."""
        sizes = {}
        for regime_id in [0, 3]:  # trending_bull vs high_volatility
            with tempfile.TemporaryDirectory() as tmp:
                state_path = Path(tmp) / "paper_state.json"

                # Seed trade history so Kelly has data
                state = {
                    "cash": 100_000.0,
                    "realized_pnl": 0.0,
                    "positions": [],
                    "trades": _seed_trades(60),
                    "last_update": None,
                }
                _seed_state(state_path, state)

                trader = PaperTrader(
                    state_path=state_path,
                    initial_capital=100_000.0,
                    max_total_positions=10,
                    use_kelly_sizing=True,
                    kelly_fraction=0.5,
                    kelly_min_size_multiplier=0.25,
                    kelly_max_size_multiplier=1.5,
                )

                price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
                preds = _make_predictions(
                    [
                        {
                            "ticker": "AAPL",
                            "predicted_return": 0.05,
                            "confidence": 0.90,
                            "regime": regime_id,
                        }
                    ]
                )
                strategy = _make_strategy(position_size_pct=0.05)

                result = trader.run_cycle([strategy], preds, price_data)
                self.assertEqual(result["entries"], 1)

                saved = _load_state(state_path)
                pos = saved["positions"][0]
                sizes[regime_id] = pos["position_size_pct"]

        # Regime 3 (high_volatility) has REGIME_RISK_MULTIPLIER=0.60
        # vs regime 0 (trending_bull) at 1.00 → size should be smaller
        self.assertGreater(
            sizes[0], sizes[3],
            "Trending bull should have larger position than high volatility",
        )


class TestEquityCurveTracked(unittest.TestCase):
    """State has equity_curve with daily entries after a cycle."""

    def test_equity_curve_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"
            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                use_kelly_sizing=False,
            )

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )

            trader.run_cycle([], preds, price_data)

            saved = _load_state(state_path)
            self.assertIn("equity_curve", saved)
            ec = saved["equity_curve"]
            self.assertEqual(len(ec), 1)

            entry = ec[0]
            self.assertIn("date", entry)
            self.assertIn("equity", entry)
            self.assertIn("cash", entry)
            self.assertIn("positions_value", entry)
            self.assertIn("n_positions", entry)
            self.assertIn("daily_pnl", entry)
            self.assertIn("drawdown", entry)
            self.assertIn("dd_state", entry)


class TestRiskMetricsComputed(unittest.TestCase):
    """Summary includes Sharpe, drawdown, volatility when equity curve is long enough."""

    def test_risk_metrics_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Seed state with 19 equity curve entries (need 20 total for metrics)
            equity_curve = []
            for i in range(19):
                equity_curve.append(
                    {
                        "date": f"2024-01-{i + 1:02d}",
                        "equity": 100_000.0 + i * 100,
                        "cash": 100_000.0,
                        "positions_value": float(i * 100),
                        "n_positions": 0,
                        "daily_pnl": 0.001,
                        "drawdown": 0.0,
                        "dd_state": "NORMAL",
                    }
                )

            state = {
                "cash": 100_000.0,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "last_update": None,
                "equity_curve": equity_curve,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                use_kelly_sizing=False,
            )

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )

            result = trader.run_cycle([], preds, price_data)

            # After this cycle, equity_curve has 20 entries → risk_metrics computed
            self.assertIn("risk_metrics", result)
            rm = result["risk_metrics"]
            self.assertIn("sharpe", rm)
            self.assertIn("max_drawdown", rm)
            self.assertIn("volatility", rm)
            self.assertIn("calmar", rm)
            self.assertIn("win_rate", rm)

            # Basic sanity: Sharpe should be a finite number
            self.assertTrue(np.isfinite(rm["sharpe"]))
            self.assertTrue(np.isfinite(rm["volatility"]))
            self.assertGreaterEqual(rm["win_rate"], 0.0)
            self.assertLessEqual(rm["win_rate"], 1.0)

    def test_no_risk_metrics_when_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"
            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                use_kelly_sizing=False,
            )

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )

            result = trader.run_cycle([], preds, price_data)

            # Only 1 equity curve entry → no risk_metrics
            self.assertNotIn("risk_metrics", result)


class TestEquityCurveCapped(unittest.TestCase):
    """Equity curve is capped at 2520 entries."""

    def test_equity_curve_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            equity_curve = [
                {
                    "date": f"day_{i}",
                    "equity": 100_000.0 + i,
                    "cash": 100_000.0,
                    "positions_value": float(i),
                    "n_positions": 0,
                    "daily_pnl": 0.0001,
                    "drawdown": 0.0,
                    "dd_state": "NORMAL",
                }
                for i in range(2520)
            ]

            state = {
                "cash": 100_000.0,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "last_update": None,
                "equity_curve": equity_curve,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                use_kelly_sizing=False,
            )

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 0,
                    }
                ]
            )

            trader.run_cycle([], preds, price_data)

            saved = _load_state(state_path)
            ec = saved["equity_curve"]
            # 2520 existing + 1 new = 2521 → capped to 2520
            self.assertLessEqual(len(ec), 2520)


class TestBackwardCompatOldStateFormat(unittest.TestCase):
    """Old state files with 'regime' field (no 'entry_regime') still work."""

    def test_backward_compat(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "paper_state.json"

            # Old-format position: has 'regime' but no 'entry_regime'
            state = {
                "cash": 95_000.0,
                "realized_pnl": 0.0,
                "positions": [
                    {
                        "strategy_id": "s1",
                        "permno": "AAPL",
                        "ticker": "AAPL",
                        "entry_date": "2024-02-01",
                        "entry_price": 100.0,
                        "shares": 50.0,
                        "holding_days": 28,
                        "max_holding_days": 30,
                        "entry_threshold": 0.001,
                        "confidence_threshold": 0.30,
                        "position_size_pct": 0.05,
                        "regime": 1,
                        # No entry_regime, no highest_price, no confidence
                    }
                ],
                "trades": [],
                "last_update": None,
            }
            _seed_state(state_path, state)

            trader = PaperTrader(
                state_path=state_path,
                initial_capital=100_000.0,
                max_total_positions=10,
                use_kelly_sizing=False,
            )
            trader._prev_equity = 100_000.0

            price_data = _make_price_data({"AAPL": (100.0, None, 0.02)})
            preds = _make_predictions(
                [
                    {
                        "ticker": "AAPL",
                        "predicted_return": 0.05,
                        "confidence": 0.90,
                        "regime": 1,
                    }
                ]
            )
            strategy = _make_strategy()

            # Should not crash — graceful handling of old state format
            result = trader.run_cycle([strategy], preds, price_data)
            self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
