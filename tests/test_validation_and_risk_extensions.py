"""
Test module for validation and risk extensions behavior and regressions.
"""

import unittest

import numpy as np
import pandas as pd

from quant_engine.backtest.validation import (
    combinatorial_purged_cv,
    superior_predictive_ability,
    strategy_signal_returns,
)
from quant_engine.risk.portfolio_risk import PortfolioRiskManager


def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
    """Build synthetic test data for the scenarios in this module."""
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values * 1.01,
            "Low": close.values * 0.99,
            "Close": close.values,
            "Volume": np.full(len(close), 1_000_000.0),
        },
        index=close.index,
    )


class ValidationAndRiskExtensionTests(unittest.TestCase):
    """Test cases covering validation and risk extensions behavior and system invariants."""
    def test_cpcv_detects_positive_signal_quality(self):
        rng = np.random.RandomState(7)
        idx = pd.date_range("2020-01-01", periods=700, freq="D")

        latent = rng.normal(0.002, 0.01, size=len(idx))
        predictions = pd.Series(latent + rng.normal(0, 0.002, size=len(idx)), index=idx)
        actuals = pd.Series(latent + rng.normal(0, 0.003, size=len(idx)), index=idx)

        cpcv = combinatorial_purged_cv(
            predictions=predictions,
            actuals=actuals,
            entry_threshold=0.001,
            n_partitions=8,
            n_test_partitions=4,
            purge_gap=10,
            embargo=5,
        )

        self.assertGreater(cpcv.n_combinations, 0)
        self.assertGreater(cpcv.mean_oos_corr, 0.0)
        self.assertGreater(cpcv.positive_oos_fraction, 0.5)

    def test_spa_passes_for_consistently_positive_signal_returns(self):
        rng = np.random.RandomState(11)
        idx = pd.date_range("2021-01-01", periods=500, freq="D")
        preds = pd.Series(0.004 + rng.normal(0, 0.001, size=len(idx)), index=idx)
        actuals = pd.Series(0.003 + rng.normal(0, 0.0015, size=len(idx)), index=idx)
        signal_returns = strategy_signal_returns(preds, actuals, entry_threshold=0.001)

        spa = superior_predictive_ability(
            strategy_returns=signal_returns,
            benchmark_returns=None,
            n_bootstraps=200,
            block_size=10,
        )

        self.assertGreater(spa.observed_mean, 0.0)
        self.assertTrue(spa.passes)

    def test_portfolio_risk_rejects_high_projected_volatility(self):
        rng = np.random.RandomState(5)
        idx = pd.date_range("2022-01-01", periods=120, freq="D")
        r1 = rng.normal(0.0, 0.035, size=len(idx))
        r2 = r1 + rng.normal(0.0, 0.01, size=len(idx))
        c1 = pd.Series(100.0 * np.cumprod(1 + r1), index=idx)
        c2 = pd.Series(80.0 * np.cumprod(1 + r2), index=idx)

        price_data = {
            "AAPL": _make_ohlcv(c1),
            "MSFT": _make_ohlcv(c2),
        }

        risk = PortfolioRiskManager(
            max_sector_pct=1.0,
            max_corr_between=1.0,
            max_gross_exposure=2.0,
            max_single_name_pct=1.0,
            max_portfolio_vol=0.05,
            correlation_lookback=100,
        )

        check = risk.check_new_position(
            ticker="MSFT",
            position_size=0.45,
            current_positions={"AAPL": 0.45},
            price_data=price_data,
        )

        self.assertFalse(check.passed)
        self.assertTrue(any("Portfolio vol" in v for v in check.violations))


if __name__ == "__main__":
    unittest.main()
