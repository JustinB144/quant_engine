"""
Tests for SPEC-P01: Replace hardcoded regime_stats with learned statistics.

Covers:
    1. update_regime_stats uses MIN_REGIME_TRADES_FOR_STATS threshold (30)
    2. Stats below threshold keep Bayesian prior defaults
    3. Persistence: save_regime_stats writes JSON to disk
    4. Persistence: _load_regime_stats restores from disk on init
    5. Corrupt/missing persist files handled gracefully
    6. size_position uses learned stats when n_trades >= threshold
    7. size_position falls back to caller-supplied stats when n_trades < threshold
    8. Config constants exist and have correct values
    9. End-to-end: after 200+ trades, regime_stats differ from defaults
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.risk.position_sizer import PositionSizer
from quant_engine.config import (
    MIN_REGIME_TRADES_FOR_STATS,
    REGIME_STATS_PERSIST_PATH,
    REGIME_NAMES,
)


def _fresh_position_sizer(**kwargs) -> PositionSizer:
    """Create a PositionSizer isolated from on-disk persisted state.

    The default persist path may contain real learned stats, which would
    contaminate tests that expect hardcoded defaults.  This helper points
    the persist path to a non-existent file and resets regime_stats.
    """
    ps = PositionSizer(**kwargs)
    # Override to a non-existent path so _load_regime_stats is a no-op
    ps._regime_stats_persist_path = Path("/tmp/_nonexistent_regime_stats.json")
    # Reset to pristine defaults (undo any load from the real persist file)
    ps.regime_stats = {
        "trending_bull": {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": -0.02, "n_trades": 0},
        "trending_bear": {"win_rate": 0.45, "avg_win": 0.02, "avg_loss": -0.025, "n_trades": 0},
        "mean_reverting": {"win_rate": 0.50, "avg_win": 0.02, "avg_loss": -0.02, "n_trades": 0},
        "high_volatility": {"win_rate": 0.42, "avg_win": 0.04, "avg_loss": -0.035, "n_trades": 0},
    }
    return ps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime_trades(
    regime_id: int,
    n_wins: int,
    n_losses: int,
    avg_win: float = 0.03,
    avg_loss: float = -0.02,
) -> pd.DataFrame:
    """Create a synthetic trade DataFrame for a single regime."""
    wins = np.full(n_wins, avg_win)
    losses = np.full(n_losses, avg_loss)
    returns = np.concatenate([wins, losses])
    np.random.seed(42)
    np.random.shuffle(returns)
    return pd.DataFrame({
        "net_return": returns,
        "regime": [regime_id] * len(returns),
    })


def _make_multi_regime_trades(n_per_regime: int = 50) -> pd.DataFrame:
    """Create trades spanning all 4 regimes with distinct win rates."""
    parts = []
    # regime 0 (bull): 70% win
    parts.append(_make_regime_trades(0, int(n_per_regime * 0.7), int(n_per_regime * 0.3)))
    # regime 1 (bear): 40% win
    parts.append(_make_regime_trades(1, int(n_per_regime * 0.4), int(n_per_regime * 0.6)))
    # regime 2 (mean-revert): 55% win
    parts.append(_make_regime_trades(2, int(n_per_regime * 0.55), int(n_per_regime * 0.45)))
    # regime 3 (high-vol): 35% win
    parts.append(_make_regime_trades(3, int(n_per_regime * 0.35), int(n_per_regime * 0.65)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Config constants
# ---------------------------------------------------------------------------

class TestConfigConstants:
    """Verify SPEC-P01 config values exist and are correct."""

    def test_min_regime_trades_for_stats(self):
        assert MIN_REGIME_TRADES_FOR_STATS == 30

    def test_regime_stats_persist_path(self):
        assert str(REGIME_STATS_PERSIST_PATH).endswith("regime_trade_stats.json")

    def test_persist_path_is_under_trained_models(self):
        assert "trained_models" in str(REGIME_STATS_PERSIST_PATH)


# ---------------------------------------------------------------------------
# 2. Threshold enforcement
# ---------------------------------------------------------------------------

class TestThresholdEnforcement:
    """update_regime_stats only updates when trades >= threshold."""

    def test_at_threshold_updates(self):
        """Exactly 30 trades should update stats."""
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 20, 10)
        assert len(trades) == 30
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 30

    def test_below_threshold_keeps_defaults(self):
        """29 trades should NOT update stats."""
        ps = _fresh_position_sizer()
        default_wr = ps.regime_stats["trending_bull"]["win_rate"]
        trades = _make_regime_trades(0, 19, 10)
        assert len(trades) == 29
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["win_rate"] == default_wr
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_above_threshold_updates(self):
        """50 trades should update stats."""
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 35, 15)
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 50
        assert ps.regime_stats["trending_bull"]["win_rate"] == pytest.approx(0.70, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Win rate / avg_win / avg_loss accuracy
# ---------------------------------------------------------------------------

class TestStatsAccuracy:
    """Learned stats should reflect actual trade outcomes."""

    def test_win_rate_computed_correctly(self):
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 24, 6)  # 80% win rate
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["win_rate"] == pytest.approx(0.80, abs=0.01)

    def test_avg_win_computed_correctly(self):
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 24, 6, avg_win=0.05)
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["avg_win"] == pytest.approx(0.05, abs=0.001)

    def test_avg_loss_computed_correctly(self):
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 24, 6, avg_loss=-0.03)
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["avg_loss"] == pytest.approx(-0.03, abs=0.001)

    def test_all_winning_trades(self):
        """100% win rate should be 1.0 with avg_loss=0."""
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(0, 30, 0, avg_win=0.02)
        ps.update_regime_stats(trades, persist=False)
        stats = ps.regime_stats["trending_bull"]
        assert stats["win_rate"] == 1.0
        assert stats["avg_loss"] == 0.0
        assert stats["avg_win"] == pytest.approx(0.02, abs=0.001)

    def test_all_losing_trades(self):
        """0% win rate should be 0.0 with avg_win=0."""
        ps = _fresh_position_sizer()
        trades = _make_regime_trades(1, 0, 30, avg_loss=-0.015)
        ps.update_regime_stats(trades, persist=False)
        stats = ps.regime_stats["trending_bear"]
        assert stats["win_rate"] == 0.0
        assert stats["avg_win"] == 0.0
        assert stats["avg_loss"] == pytest.approx(-0.015, abs=0.001)


# ---------------------------------------------------------------------------
# 4. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """Save/load regime stats to/from disk."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Stats saved to disk should be restored on new PositionSizer init."""
        persist_path = tmp_path / "regime_trade_stats.json"

        # Create sizer with custom persist path, update stats, save
        ps1 = _fresh_position_sizer()
        ps1._regime_stats_persist_path = persist_path
        trades = _make_multi_regime_trades(n_per_regime=50)
        ps1.update_regime_stats(trades, persist=True)

        # Verify file was written
        assert persist_path.exists()
        data = json.loads(persist_path.read_text())
        assert "trending_bull" in data
        assert data["trending_bull"]["n_trades"] >= 30

        # Create new sizer pointing to the same persist path
        ps2 = _fresh_position_sizer()
        ps2._regime_stats_persist_path = persist_path
        ps2._load_regime_stats()

        # Verify loaded stats match
        for regime_name in REGIME_NAMES.values():
            if ps1.regime_stats[regime_name]["n_trades"] >= 30:
                assert ps2.regime_stats[regime_name]["n_trades"] == \
                    ps1.regime_stats[regime_name]["n_trades"]
                assert ps2.regime_stats[regime_name]["win_rate"] == pytest.approx(
                    ps1.regime_stats[regime_name]["win_rate"], abs=0.001
                )

    def test_missing_file_keeps_defaults(self, tmp_path):
        """Missing persist file should leave defaults untouched."""
        ps = _fresh_position_sizer()
        ps._regime_stats_persist_path = tmp_path / "nonexistent.json"
        ps._load_regime_stats()
        # Defaults should be unchanged
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0
        assert ps.regime_stats["trending_bull"]["win_rate"] == 0.55

    def test_corrupt_json_handled_gracefully(self, tmp_path):
        """Corrupt JSON should not crash; defaults remain."""
        persist_path = tmp_path / "corrupt.json"
        persist_path.write_text("{{not valid json")

        ps = _fresh_position_sizer()
        ps._regime_stats_persist_path = persist_path
        ps._load_regime_stats()  # Should not raise
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_persisted_stats_below_threshold_ignored(self, tmp_path):
        """Persisted stats with n_trades < threshold are not loaded."""
        persist_path = tmp_path / "low_trades.json"
        data = {
            "trending_bull": {"win_rate": 0.99, "avg_win": 0.10, "avg_loss": -0.01, "n_trades": 5},
        }
        persist_path.write_text(json.dumps(data))

        ps = _fresh_position_sizer()
        ps._regime_stats_persist_path = persist_path
        ps._load_regime_stats()
        # Should still be the default, not 0.99
        assert ps.regime_stats["trending_bull"]["win_rate"] == 0.55
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_empty_dict_handled(self, tmp_path):
        """Empty JSON dict should not crash."""
        persist_path = tmp_path / "empty.json"
        persist_path.write_text("{}")

        ps = _fresh_position_sizer()
        ps._regime_stats_persist_path = persist_path
        ps._load_regime_stats()  # Should not raise
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0


# ---------------------------------------------------------------------------
# 5. size_position uses learned stats
# ---------------------------------------------------------------------------

class TestSizePositionUsesLearnedStats:
    """size_position should use learned regime stats when available."""

    def test_learned_stats_override_caller_args(self):
        """When learned stats have enough trades, they override win_rate/avg_win/avg_loss."""
        ps = _fresh_position_sizer(max_position_pct=1.0)
        # Load 40 trades with 90% win rate for trending_bull
        trades = _make_regime_trades(0, 36, 4, avg_win=0.05, avg_loss=-0.01)
        ps.update_regime_stats(trades, persist=False)

        # Call size_position with mediocre caller-supplied stats
        result = ps.size_position(
            "TEST",
            win_rate=0.50,   # mediocre — should be overridden
            avg_win=0.01,
            avg_loss=-0.01,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
            regime="trending_bull",
            n_trades=40,
        )
        # Kelly with 90% win rate and 5:1 payoff should be much higher
        # than with 50% win rate and 1:1 payoff
        result_no_regime = ps.size_position(
            "TEST",
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=-0.01,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
            regime=None,  # No regime — uses caller stats
            n_trades=40,
        )
        assert result.raw_kelly > result_no_regime.raw_kelly

    def test_insufficient_learned_stats_uses_caller_args(self):
        """When learned stats have too few trades, caller args are used."""
        ps = _fresh_position_sizer()
        # Only 10 trades — below threshold
        trades = _make_regime_trades(0, 9, 1, avg_win=0.10)
        ps.update_regime_stats(trades, persist=False)

        result_with_regime = ps.size_position(
            "TEST",
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=-0.02,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
            regime="trending_bull",
        )
        result_no_regime = ps.size_position(
            "TEST",
            win_rate=0.50,
            avg_win=0.02,
            avg_loss=-0.02,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
            regime=None,
        )
        # Kelly should be the same since learned stats were not applied
        assert result_with_regime.raw_kelly == result_no_regime.raw_kelly


# ---------------------------------------------------------------------------
# 6. End-to-end verification
# ---------------------------------------------------------------------------

class TestEndToEndVerification:
    """After 200+ trades, regime_stats should differ from hardcoded defaults."""

    def test_stats_differ_from_defaults_after_200_trades(self):
        """SPEC-P01 verification: after 200+ trades, stats reflect real data."""
        ps = _fresh_position_sizer()
        defaults = {
            rn: dict(ps.regime_stats[rn]) for rn in REGIME_NAMES.values()
        }

        trades = _make_multi_regime_trades(n_per_regime=55)
        assert len(trades) >= 200  # Verify test has 200+ trades
        ps.update_regime_stats(trades, persist=False)

        changed = 0
        for regime_name in REGIME_NAMES.values():
            stats = ps.regime_stats[regime_name]
            if stats["n_trades"] >= 30:
                changed += 1
                assert stats["n_trades"] > 0
                # Win rate should differ from the hardcoded default
                assert stats["win_rate"] != defaults[regime_name]["win_rate"]

        # At least 3 regimes should have been updated (all 4 get 50+ trades)
        assert changed >= 3

    def test_multi_regime_win_rates_reflect_data(self):
        """Each regime should have a distinct learned win rate."""
        ps = _fresh_position_sizer()
        trades = _make_multi_regime_trades(n_per_regime=50)
        ps.update_regime_stats(trades, persist=False)

        # Bull should have highest win rate
        bull_wr = ps.regime_stats["trending_bull"]["win_rate"]
        bear_wr = ps.regime_stats["trending_bear"]["win_rate"]
        hv_wr = ps.regime_stats["high_volatility"]["win_rate"]
        mr_wr = ps.regime_stats["mean_reverting"]["win_rate"]

        assert bull_wr > bear_wr, "Bull should beat bear"
        assert mr_wr > hv_wr, "Mean-revert should beat high-vol"
        assert bull_wr > hv_wr, "Bull should beat high-vol"


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for update_regime_stats."""

    def test_empty_dataframe(self):
        """Empty DataFrame should not crash or change stats."""
        ps = _fresh_position_sizer()
        ps.update_regime_stats(pd.DataFrame(), persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_missing_net_return_column(self):
        """DataFrame without net_return should be a no-op."""
        ps = _fresh_position_sizer()
        trades = pd.DataFrame({"regime": [0] * 40, "pnl": [0.01] * 40})
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_missing_regime_column(self):
        """DataFrame without regime column should be a no-op."""
        ps = _fresh_position_sizer()
        trades = pd.DataFrame({"net_return": [0.01] * 40})
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 0

    def test_nan_returns_excluded(self):
        """NaN returns should be excluded from stats computation."""
        ps = _fresh_position_sizer()
        returns = [0.02] * 25 + [-0.01] * 10 + [float("nan")] * 5
        trades = pd.DataFrame({
            "net_return": returns,
            "regime": [0] * 40,
        })
        ps.update_regime_stats(trades, persist=False)
        stats = ps.regime_stats["trending_bull"]
        assert stats["n_trades"] == 35  # 40 - 5 NaN = 35

    def test_zero_returns_counted_as_losses(self):
        """Returns exactly equal to 0 should be counted as losses."""
        ps = _fresh_position_sizer()
        returns = [0.02] * 20 + [0.0] * 10 + [-0.01] * 5
        trades = pd.DataFrame({
            "net_return": returns,
            "regime": [0] * 35,
        })
        ps.update_regime_stats(trades, persist=False)
        stats = ps.regime_stats["trending_bull"]
        # wins = 20, losses = 15 (10 zeros + 5 negatives)
        assert stats["win_rate"] == pytest.approx(20 / 35, abs=0.01)
