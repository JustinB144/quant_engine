"""
Test module for paper trader kelly behavior and regressions.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_engine.autopilot.paper_trader import PaperTrader
from quant_engine.autopilot.registry import ActiveStrategy


def _mock_price_data() -> dict:
    """Create mock inputs used by the test cases in this module."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(100.0 + np.linspace(0, 4, len(idx)), index=idx)
    return {
        "AAPL": pd.DataFrame(
            {
                "Open": close.values,
                "High": close.values * 1.003,
                "Low": close.values * 0.997,
                "Close": close.values,
                "Volume": np.full(len(idx), 2_000_000.0),
            },
            index=idx,
        ),
    }


def _seed_state(path: Path):
    """Seed deterministic test state used by this module."""
    trades = []
    # Strong recent performance to produce non-trivial Kelly sizing.
    for i in range(60):
        pnl = 120.0 if i % 5 != 0 else -45.0
        trades.append(
            {
                "strategy_id": "s1",
                "ticker": "AAPL",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "shares": 50.0,
                "pnl": pnl,
                "holding_days": 5,
                "reason": "time_exit",
            },
        )
    state = {
        "cash": 100_000.0,
        "realized_pnl": 0.0,
        "positions": [],
        "trades": trades,
        "last_update": None,
    }
    with open(path, "w") as f:
        json.dump(state, f)


def _run_cycle(use_kelly: bool) -> float:
    """Run the local test helper workflow and return intermediate outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "paper_state.json"
        _seed_state(state_path)

        trader = PaperTrader(
            state_path=state_path,
            initial_capital=100_000.0,
            max_total_positions=10,
            use_kelly_sizing=use_kelly,
            kelly_fraction=0.5,
            kelly_min_size_multiplier=0.25,
            kelly_max_size_multiplier=1.5,
        )
        active = [
            ActiveStrategy(
                strategy_id="s1",
                promoted_at="2026-02-20T00:00:00",
                params={
                    "entry_threshold": 0.001,
                    "confidence_threshold": 0.50,
                    "horizon": 10,
                    "position_size_pct": 0.05,
                },
                score=1.0,
                metrics={},
            ),
        ]
        latest_predictions = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "predicted_return": 0.02,
                    "confidence": 0.95,
                    "regime": 0,
                },
            ],
        )
        report = trader.run_cycle(
            active_strategies=active,
            latest_predictions=latest_predictions,
            price_data=_mock_price_data(),
        )
        assert report["entries"] == 1

        with open(state_path, "r") as f:
            state = json.load(f)
        return float(state["positions"][0]["position_size_pct"])


class PaperTraderKellyTests(unittest.TestCase):
    """Test cases covering paper trader kelly behavior and system invariants."""
    def test_kelly_sizing_changes_position_size_with_bounds(self):
        fixed = _run_cycle(use_kelly=False)
        kelly = _run_cycle(use_kelly=True)

        self.assertAlmostEqual(fixed, 0.05, places=6)
        self.assertGreater(kelly, fixed)
        self.assertLessEqual(kelly, 0.075 + 1e-9)
        self.assertGreaterEqual(kelly, 0.001)


if __name__ == "__main__":
    unittest.main()
