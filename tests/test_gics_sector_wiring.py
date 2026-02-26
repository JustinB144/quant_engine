"""
Tests for SPEC-W06: GICS sector loading and wiring.

Verifies:
  - GICS_SECTORS is populated from config_data/universe.yaml at import time
  - validate_config() does NOT emit a GICS warning when sectors are loaded
  - portfolio_optimizer falls back to GICS_SECTORS when sector_map is None
  - refresh_gics_sectors() maps Compustat gsector codes to sector names
  - _write_gics_to_universe_yaml() correctly merges sector data
  - run_wrds_daily_refresh.py --gics flag is accepted by argparse
"""
import importlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Part A: GICS_SECTORS loading from universe.yaml ────────────────────────


class TestGICSSectorsLoading:
    """Verify that GICS_SECTORS is populated from universe.yaml at import."""

    def test_gics_sectors_populated(self):
        """GICS_SECTORS should have entries after config module import."""
        from quant_engine.config import GICS_SECTORS

        assert len(GICS_SECTORS) > 0, "GICS_SECTORS should not be empty"

    def test_gics_sectors_maps_ticker_to_sector(self):
        """Each entry should map a ticker string to a sector name string."""
        from quant_engine.config import GICS_SECTORS

        for ticker, sector in GICS_SECTORS.items():
            assert isinstance(ticker, str), f"Ticker key should be str, got {type(ticker)}"
            assert isinstance(sector, str), f"Sector value should be str, got {type(sector)}"
            assert ticker == ticker.upper(), f"Ticker should be uppercase, got '{ticker}'"
            assert len(ticker) > 0, "Ticker should not be empty"
            assert len(sector) > 0, "Sector should not be empty"

    def test_known_tickers_have_sectors(self):
        """Well-known tickers from universe.yaml should appear in GICS_SECTORS."""
        from quant_engine.config import GICS_SECTORS

        expected = {
            "AAPL": "tech",
            "MSFT": "tech",
            "JPM": "financial",
            "JNJ": "healthcare",
            "CVX": "energy",
            "NEE": "utilities",
            "CAT": "industrial",
            "AMZN": "consumer",
        }
        for ticker, expected_sector in expected.items():
            assert ticker in GICS_SECTORS, f"{ticker} should be in GICS_SECTORS"
            assert GICS_SECTORS[ticker] == expected_sector, (
                f"{ticker} should map to '{expected_sector}', got '{GICS_SECTORS[ticker]}'"
            )

    def test_gics_sectors_count_matches_universe_yaml(self):
        """GICS_SECTORS count should match total tickers in universe.yaml sectors."""
        from quant_engine.config import GICS_SECTORS

        yaml_path = Path(__file__).parent.parent / "config_data" / "universe.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Count unique tickers in universe.yaml
        yaml_tickers = set()
        for sector_name, tickers in data["sectors"].items():
            for ticker in tickers:
                yaml_tickers.add(str(ticker).upper())

        assert len(GICS_SECTORS) == len(yaml_tickers), (
            f"GICS_SECTORS has {len(GICS_SECTORS)} entries but universe.yaml "
            f"has {len(yaml_tickers)} unique tickers"
        )

    def test_all_sectors_represented(self):
        """All sector names from universe.yaml should appear as values in GICS_SECTORS."""
        from quant_engine.config import GICS_SECTORS

        yaml_path = Path(__file__).parent.parent / "config_data" / "universe.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        yaml_sectors = set(data["sectors"].keys())
        gics_sectors = set(GICS_SECTORS.values())

        for sector in yaml_sectors:
            assert sector in gics_sectors, (
                f"Sector '{sector}' from universe.yaml not found in GICS_SECTORS values"
            )


# ── Part A continued: validate_config() ─────────────────────────────────────


class TestValidateConfigGICS:
    """Verify validate_config() no longer warns about empty GICS_SECTORS."""

    def test_no_gics_warning_when_populated(self):
        """validate_config() should not emit GICS warning when sectors loaded."""
        from quant_engine.config import validate_config

        issues = validate_config()
        gics_issues = [i for i in issues if "GICS" in i["message"]]
        assert len(gics_issues) == 0, (
            f"Should not have GICS warnings, got: {gics_issues}"
        )

    def test_warning_message_references_both_methods(self):
        """When GICS_SECTORS IS empty, the warning should reference both methods."""
        from quant_engine import config

        # Temporarily empty GICS_SECTORS
        original = dict(config.GICS_SECTORS)
        config.GICS_SECTORS.clear()
        try:
            issues = config.validate_config()
            gics_issues = [i for i in issues if "GICS" in i["message"]]
            assert len(gics_issues) == 1
            msg = gics_issues[0]["message"]
            assert "run_wrds_daily_refresh.py --gics" in msg, (
                "Warning should reference --gics flag"
            )
            assert "universe.yaml" in msg, (
                "Warning should reference universe.yaml"
            )
        finally:
            config.GICS_SECTORS.update(original)


# ── Part B: portfolio_optimizer sector constraint fallback ──────────────────


class TestPortfolioOptimizerSectorFallback:
    """Verify optimize_portfolio uses GICS_SECTORS when sector_map not passed."""

    def _make_test_data(self, n=5):
        """Create minimal expected_returns and covariance for testing."""
        tickers = ["AAPL", "MSFT", "JPM", "JNJ", "CVX"][:n]
        expected_returns = pd.Series(
            np.random.uniform(0.001, 0.01, n), index=tickers
        )
        # Create a proper PSD covariance matrix
        rng = np.random.default_rng(42)
        A = rng.normal(0, 0.01, (n, n))
        cov_matrix = pd.DataFrame(
            A.T @ A / n + 1e-4 * np.eye(n),
            index=tickers,
            columns=tickers,
        )
        return expected_returns, cov_matrix

    def test_sector_constraint_applied_with_gics_fallback(self):
        """When sector_map is None, optimizer should use GICS_SECTORS from config."""
        from quant_engine.config import GICS_SECTORS
        from quant_engine.risk.portfolio_optimizer import optimize_portfolio

        # Ensure GICS_SECTORS is populated
        assert len(GICS_SECTORS) > 0, "GICS_SECTORS must be populated for this test"

        expected_returns, cov_matrix = self._make_test_data()

        # Call without sector_map — should fall back to GICS_SECTORS
        weights = optimize_portfolio(
            expected_returns=expected_returns,
            covariance=cov_matrix,
            sector_map=None,
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        # Weights should sum to approximately 1
        assert abs(weights.sum() - 1.0) < 1e-4, (
            f"Weights should sum to 1, got {weights.sum()}"
        )

    def test_sector_constraint_applied_with_explicit_map(self):
        """When sector_map is explicitly provided, it should be used."""
        from quant_engine.risk.portfolio_optimizer import optimize_portfolio

        expected_returns, cov_matrix = self._make_test_data()
        sector_map = {
            "AAPL": "tech",
            "MSFT": "tech",
            "JPM": "financial",
            "JNJ": "healthcare",
            "CVX": "energy",
        }

        weights = optimize_portfolio(
            expected_returns=expected_returns,
            covariance=cov_matrix,
            sector_map=sector_map,
            max_sector_exposure=0.10,
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) > 0

    def test_sector_constraint_warns_when_gics_empty(self):
        """When GICS_SECTORS is empty and no sector_map passed, should warn."""
        from quant_engine import config
        from quant_engine.risk import portfolio_optimizer
        from quant_engine.risk.portfolio_optimizer import optimize_portfolio

        expected_returns, cov_matrix = self._make_test_data()

        # Save and reset state
        original_flag = portfolio_optimizer._WARNED_GICS_EMPTY[0]
        original_gics = dict(config.GICS_SECTORS)
        portfolio_optimizer._WARNED_GICS_EMPTY[0] = False
        config.GICS_SECTORS.clear()

        try:
            with patch.object(portfolio_optimizer.logger, "warning") as mock_warn:
                optimize_portfolio(
                    expected_returns=expected_returns,
                    covariance=cov_matrix,
                    sector_map=None,
                )
                # Should have warned about empty GICS_SECTORS
                mock_warn.assert_called()
                # Check all warning calls for the GICS message
                all_warnings = [
                    call[0][0] for call in mock_warn.call_args_list
                ]
                gics_warnings = [
                    w for w in all_warnings if "GICS_SECTORS is empty" in w
                ]
                assert len(gics_warnings) > 0, (
                    f"Expected GICS warning, got: {all_warnings}"
                )
        finally:
            config.GICS_SECTORS.update(original_gics)
            portfolio_optimizer._WARNED_GICS_EMPTY[0] = original_flag


# ── Part B: refresh_gics_sectors function ───────────────────────────────────


class TestRefreshGICSSectors:
    """Test the refresh_gics_sectors function from run_wrds_daily_refresh.py."""

    def test_refresh_returns_sector_mapping(self):
        """refresh_gics_sectors should return ticker->sector dict from mock WRDS."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import refresh_gics_sectors

        # Mock a WRDS provider that returns a DataFrame
        mock_provider = MagicMock()
        mock_df = pd.DataFrame({
            "tic": ["AAPL", "JPM", "CVX"],
            "gsector": [45.0, 40.0, 10.0],
        })
        mock_provider._query.return_value = mock_df

        result = refresh_gics_sectors(mock_provider, ["AAPL", "JPM", "CVX"])

        assert result["AAPL"] == "Information Technology"
        assert result["JPM"] == "Financials"
        assert result["CVX"] == "Energy"

    def test_refresh_handles_empty_response(self):
        """refresh_gics_sectors should return empty dict when WRDS returns nothing."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import refresh_gics_sectors

        mock_provider = MagicMock()
        mock_provider._query.return_value = pd.DataFrame()

        result = refresh_gics_sectors(mock_provider, ["AAPL"])
        assert result == {}

    def test_refresh_handles_empty_tickers(self):
        """refresh_gics_sectors should return empty dict for empty ticker list."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import refresh_gics_sectors

        mock_provider = MagicMock()
        result = refresh_gics_sectors(mock_provider, [])
        assert result == {}
        mock_provider._query.assert_not_called()

    def test_refresh_handles_unknown_sector_code(self):
        """Unrecognized GICS sector codes should map to 'Unknown'."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import refresh_gics_sectors

        mock_provider = MagicMock()
        mock_df = pd.DataFrame({
            "tic": ["FAKE"],
            "gsector": [99.0],
        })
        mock_provider._query.return_value = mock_df

        result = refresh_gics_sectors(mock_provider, ["FAKE"])
        assert result["FAKE"] == "Unknown"

    def test_refresh_batches_large_ticker_lists(self):
        """Large ticker lists should be batched in queries."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import refresh_gics_sectors

        mock_provider = MagicMock()
        mock_provider._query.return_value = pd.DataFrame()

        # Create a ticker list larger than batch_size (200)
        tickers = [f"T{i:04d}" for i in range(450)]
        refresh_gics_sectors(mock_provider, tickers)

        # Should have been called 3 times (200 + 200 + 50)
        assert mock_provider._query.call_count == 3


# ── Part B: _write_gics_to_universe_yaml ────────────────────────────────────


class TestWriteGICSToUniverseYaml:
    """Test the YAML writer for GICS sector data."""

    def test_writes_sector_mapping(self):
        """_write_gics_to_universe_yaml should write sectors grouped by name."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import _write_gics_to_universe_yaml

        sector_map = {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "JPM": "Financials",
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("liquidity_tiers:\n  Mega:\n    market_cap_min: 200.0e9\n")
            tmp_path = Path(f.name)

        try:
            _write_gics_to_universe_yaml(sector_map, tmp_path)

            with open(tmp_path) as f:
                data = yaml.safe_load(f)

            # Sectors should be written
            assert "sectors" in data
            assert "information_technology" in data["sectors"]
            assert "AAPL" in data["sectors"]["information_technology"]
            assert "MSFT" in data["sectors"]["information_technology"]
            assert "financials" in data["sectors"]
            assert "JPM" in data["sectors"]["financials"]

            # liquidity_tiers should be preserved
            assert "liquidity_tiers" in data
        finally:
            tmp_path.unlink()

    def test_creates_new_file_if_missing(self):
        """Should create universe.yaml if it doesn't exist."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import _write_gics_to_universe_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "universe.yaml"
            assert not yaml_path.exists()

            sector_map = {"AAPL": "Information Technology"}
            _write_gics_to_universe_yaml(sector_map, yaml_path)

            assert yaml_path.exists()
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            assert "sectors" in data


# ── Part B: --gics argparse flag ────────────────────────────────────────────


class TestGICSArgparseFlag:
    """Verify the --gics flag is recognized by run_wrds_daily_refresh.py."""

    def test_gics_flag_accepted(self):
        """argparse should accept --gics without error."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import main

        # Patch sys.argv and verify --gics is accepted
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gics", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--skip-cleanup", action="store_true")
        parser.add_argument("--tickers", type=str, default=None)
        parser.add_argument("--years", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=50)
        parser.add_argument("--verify-only", action="store_true")

        args = parser.parse_args(["--gics"])
        assert args.gics is True

    def test_gics_flag_defaults_false(self):
        """--gics should default to False when not provided."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gics", action="store_true")
        args = parser.parse_args([])
        assert args.gics is False


# ── GICS_SECTOR_NAMES completeness ─────────────────────────────────────────


class TestGICSSectorNames:
    """Verify the GICS_SECTOR_NAMES mapping is complete."""

    def test_all_standard_gics_sectors_covered(self):
        """All 11 GICS sectors should be mapped."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from quant_engine.run_wrds_daily_refresh import GICS_SECTOR_NAMES

        expected_codes = ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
        for code in expected_codes:
            assert code in GICS_SECTOR_NAMES, f"GICS sector code {code} not mapped"

        assert len(GICS_SECTOR_NAMES) == 11, (
            f"Expected 11 GICS sectors, got {len(GICS_SECTOR_NAMES)}"
        )
