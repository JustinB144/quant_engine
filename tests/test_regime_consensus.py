"""Tests for cross-sectional regime consensus (SPEC_10 T6).

Verifies:
  - Consensus correctly computed from regime counts
  - High consensus flagged when threshold met
  - Early warning triggered at low consensus
  - Divergence detected on falling consensus trend
  - Consensus series computation over time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Basic consensus computation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeConsensus:
    """Test consensus computation mechanics."""

    def test_unanimous_consensus(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        result = rc.compute_consensus([1, 1, 1, 1, 1])

        assert result["consensus"] == 1.0
        assert result["consensus_regime"] == 1
        assert result["is_high_consensus"] is True
        assert result["is_early_warning"] is False

    def test_80_percent_consensus(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(consensus_threshold=0.80)
        # 80 out of 100 in regime 1
        regimes = [1] * 80 + [0] * 10 + [2] * 5 + [3] * 5
        result = rc.compute_consensus(regimes)

        assert result["consensus"] == 0.80
        assert result["consensus_regime"] == 1
        assert result["is_high_consensus"] is True

    def test_low_consensus_early_warning(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(early_warning_threshold=0.60)
        # Split evenly: max pct = 0.30
        regimes = [0] * 30 + [1] * 25 + [2] * 25 + [3] * 20
        result = rc.compute_consensus(regimes)

        assert result["consensus"] < 0.60
        assert result["is_early_warning"] is True

    def test_empty_securities(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        result = rc.compute_consensus([])

        assert result["consensus"] == 0.0
        assert result["n_securities"] == 0
        assert result["is_early_warning"] is True

    def test_regime_pcts_sum_to_one(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        regimes = list(np.random.randint(0, 4, 100))
        result = rc.compute_consensus(regimes)

        total = sum(result["regime_pcts"].values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDivergenceDetection:
    """Test consensus trend and divergence detection."""

    def test_falling_consensus_detected(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(divergence_window=20, divergence_slope_threshold=-0.01)

        # Simulate falling consensus over 20 days
        history = list(np.linspace(0.80, 0.40, 20))
        diverging, details = rc.detect_divergence(history, window=20)

        assert diverging is True
        assert details["slope"] < -0.01

    def test_stable_consensus_not_diverging(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(divergence_window=20, divergence_slope_threshold=-0.01)

        # Stable consensus
        history = [0.80 + np.random.normal(0, 0.01) for _ in range(20)]
        diverging, details = rc.detect_divergence(history, window=20)

        assert diverging is False

    def test_insufficient_history(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(divergence_window=20)
        diverging, details = rc.detect_divergence([0.75, 0.80])

        assert diverging is False
        assert details["reason"] == "insufficient_history"


# ---------------------------------------------------------------------------
# Early warning
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEarlyWarning:
    """Test the early warning signal."""

    def test_warning_below_threshold(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(early_warning_threshold=0.60)
        warning, reason = rc.early_warning(0.55)

        assert warning is True
        assert "regime transition" in reason.lower()

    def test_no_warning_above_threshold(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus(early_warning_threshold=0.60)
        warning, reason = rc.early_warning(0.75)

        assert warning is False
        assert reason == ""


# ---------------------------------------------------------------------------
# Time series consensus
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConsensusSeries:
    """Test consensus computation over a full time series."""

    def test_consensus_series_shape(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        np.random.seed(42)
        n_dates = 50
        n_securities = 20
        idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        # All in regime 0 initially, then transition to regime 1
        data = np.zeros((n_dates, n_securities), dtype=int)
        data[25:, :] = 1  # Regime change at midpoint

        regime_matrix = pd.DataFrame(
            data,
            index=idx,
            columns=[f"SEC_{i}" for i in range(n_securities)],
        )
        result = rc.compute_consensus_series(regime_matrix)

        assert len(result) == n_dates
        assert "consensus" in result.columns
        assert "consensus_regime" in result.columns
        assert "regime_pct_0" in result.columns

    def test_consensus_drops_during_transition(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        n_dates = 30
        n_securities = 100
        idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")

        # Gradual transition: securities switch from regime 0 to regime 1
        data = np.zeros((n_dates, n_securities), dtype=int)
        for t in range(n_dates):
            n_switched = min(int(t * 5), n_securities)  # 5 more per day
            data[t, :n_switched] = 1

        regime_matrix = pd.DataFrame(
            data,
            index=idx,
            columns=[f"SEC_{i}" for i in range(n_securities)],
        )
        result = rc.compute_consensus_series(regime_matrix)

        # Early days: high consensus (regime 0)
        assert result["consensus"].iloc[0] == 1.0
        # Mid transition: lower consensus
        assert result["consensus"].iloc[10] < 0.80
