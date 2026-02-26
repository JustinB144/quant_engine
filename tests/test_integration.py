"""
End-to-end integration tests for the quant engine pipeline.

Tests the full data -> features -> regimes -> training -> prediction -> backtest
flow using synthetic data, verifying PIT semantics, CV gap gating, regime 2
suppression, and cross-sectional ranking.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure quant_engine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Synthetic data generation helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_ohlcv(
    n_stocks: int = 10,
    n_days: int = 500,
    seed: int = 42,
) -> dict:
    """Generate synthetic OHLCV data for *n_stocks* over *n_days*.

    Each stock follows a geometric random walk with a small positive drift
    and realistic volume.  Returns a dict keyed by PERMNO-like string IDs.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    data = {}

    for i in range(n_stocks):
        permno = str(10000 + i)
        # Random walk with drift
        log_returns = rng.normal(loc=0.0003, scale=0.015, size=n_days)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        # Intraday range
        intraday_noise = rng.uniform(0.005, 0.025, size=n_days)
        high = close * (1.0 + intraday_noise)
        low = close * (1.0 - intraday_noise)
        open_ = close * (1.0 + rng.normal(0, 0.003, size=n_days))
        volume = rng.lognormal(mean=14.0, sigma=0.6, size=n_days).astype(int)

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "date"
        data[permno] = df

    return data


# ---------------------------------------------------------------------------
# 5.2  Integration tests
# ---------------------------------------------------------------------------


class TestFullPipelineSynthetic:
    """End-to-end test: data -> features -> regimes -> training -> prediction -> backtest."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        return _generate_synthetic_ohlcv(n_stocks=10, n_days=500, seed=42)

    @pytest.fixture(scope="class")
    def pipeline_outputs(self, synthetic_data):
        """Run the pipeline once and cache results for all tests in this class."""
        from quant_engine.features.pipeline import FeaturePipeline
        from quant_engine.regime.detector import RegimeDetector

        # Step 1: Feature computation
        pipeline = FeaturePipeline(
            feature_mode="core",
            include_interactions=False,
            include_research_factors=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            verbose=False,
        )
        features, targets = pipeline.compute_universe(
            synthetic_data,
            verbose=False,
            compute_targets_flag=True,
            benchmark_close=None,
        )

        # Step 2: Regime detection
        detector = RegimeDetector(method="rule")
        regime_dfs = []
        regime_prob_dfs = []
        for permno in synthetic_data:
            permno_feats = features.loc[permno]
            regime_df = detector.regime_features(permno_feats)
            regime_df["permno"] = permno
            regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
            regime_dfs.append(regime_df)
            prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
            if prob_cols:
                regime_prob_dfs.append(regime_df[prob_cols])

        regime_data = pd.concat(regime_dfs)
        regimes = regime_data["regime"]
        regime_probs = pd.concat(regime_prob_dfs) if regime_prob_dfs else None

        return {
            "features": features,
            "targets": targets,
            "regimes": regimes,
            "regime_data": regime_data,
            "regime_probs": regime_probs,
            "detector": detector,
        }

    def test_features_shape(self, pipeline_outputs, synthetic_data):
        """Features DataFrame has expected shape."""
        features = pipeline_outputs["features"]
        assert isinstance(features, pd.DataFrame)
        assert isinstance(features.index, pd.MultiIndex)
        assert features.index.nlevels == 2
        n_stocks = len(synthetic_data)
        # At least some rows per stock
        permnos = features.index.get_level_values(0).unique()
        assert len(permnos) == n_stocks

    def test_targets_shape(self, pipeline_outputs):
        """Targets DataFrame is aligned with features."""
        features = pipeline_outputs["features"]
        targets = pipeline_outputs["targets"]
        assert targets is not None
        assert len(targets) == len(features)
        # Should have forward horizon columns
        assert any(c.startswith("target_") for c in targets.columns)

    def test_regimes_aligned(self, pipeline_outputs):
        """Regimes series is aligned with features index."""
        regimes = pipeline_outputs["regimes"]
        features = pipeline_outputs["features"]
        # Regimes should have entries for our data
        assert len(regimes) > 0
        # Regime values are in {0, 1, 2, 3}
        assert set(regimes.unique()).issubset({0, 1, 2, 3})

    def test_pit_no_future_in_features(self, pipeline_outputs, synthetic_data):
        """PIT semantics: features at date t use only data up to date t."""
        features = pipeline_outputs["features"]
        # Pick a random stock
        permno = list(synthetic_data.keys())[0]
        stock_feats = features.loc[permno]
        # Moving averages / rolling stats should not have values in the first
        # few rows (warm-up period). If features exist at row 0 for a 20-day
        # SMA, that would indicate look-ahead.
        sma_cols = [c for c in stock_feats.columns if "SMA" in c and "200" in c]
        if sma_cols:
            # SMA_200 should be NaN for the first ~199 rows
            first_199 = stock_feats[sma_cols[0]].iloc[:199]
            assert first_199.isna().all(), (
                "SMA_200 should be NaN during warm-up (first 199 bars)"
            )

    def test_pit_no_future_in_targets(self, pipeline_outputs, synthetic_data):
        """PIT semantics: targets use forward returns (shift -h), so the last h
        rows should be NaN."""
        targets = pipeline_outputs["targets"]
        permno = list(synthetic_data.keys())[0]
        stock_targets = targets.loc[permno]
        # For target_10d, last 10 rows should be NaN
        if "target_10d" in stock_targets.columns:
            last_10 = stock_targets["target_10d"].iloc[-10:]
            assert last_10.isna().all(), (
                "target_10d should be NaN for last 10 bars (forward return not available)"
            )

    def test_training_produces_result(self, pipeline_outputs):
        """ModelTrainer produces a valid EnsembleResult with small config."""
        from quant_engine.models.trainer import ModelTrainer, EnsembleResult

        features = pipeline_outputs["features"]
        targets = pipeline_outputs["targets"]
        regimes = pipeline_outputs["regimes"]

        trainer = ModelTrainer(
            model_params={
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_leaf": 20,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "max_features": "sqrt",
            },
            max_features=10,
            cv_folds=3,
            holdout_fraction=0.15,
            max_gap=0.15,
        )

        target_col = "target_10d"
        if target_col not in targets.columns:
            target_col = targets.columns[0]

        result = trainer.train_ensemble(
            features=features,
            targets=targets[target_col],
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
            recency_weight=False,
        )

        assert isinstance(result, EnsembleResult)
        assert result.total_samples > 0
        assert result.total_features > 0
        # Global model should have been attempted
        # (may be None if quality gates reject, but we use lenient thresholds)


class TestCvGapHardBlock:
    """Verify that the CV gap hard block rejects overfit models."""

    def test_cv_gap_hard_block(self):
        """A trainer with max_gap=0 should reject any model with nonzero CV gap."""
        from quant_engine.models.trainer import ModelTrainer

        data = _generate_synthetic_ohlcv(n_stocks=3, n_days=500, seed=99)

        from quant_engine.features.pipeline import FeaturePipeline
        from quant_engine.regime.detector import RegimeDetector

        pipeline = FeaturePipeline(
            feature_mode="core",
            include_interactions=False,
            include_research_factors=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            verbose=False,
        )
        features, targets = pipeline.compute_universe(
            data, verbose=False, compute_targets_flag=True, benchmark_close=None,
        )

        detector = RegimeDetector(method="rule")
        regime_dfs = []
        for permno in data:
            permno_feats = features.loc[permno]
            regime_df = detector.regime_features(permno_feats)
            regime_df["permno"] = permno
            regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
            regime_dfs.append(regime_df)
        regime_data = pd.concat(regime_dfs)
        regimes = regime_data["regime"]

        # Use an extremely tight max_gap so the hard block (3x max_gap) is very low.
        # With max_gap=0.001, the hard block threshold = 0.003.
        # A deliberately overfit model (deep trees, many estimators, no subsampling)
        # should have IS >> OOS, triggering the block.
        trainer = ModelTrainer(
            model_params={
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "learning_rate": 0.3,
                "subsample": 1.0,
                "max_features": 1.0,
            },
            max_features=30,
            cv_folds=3,
            holdout_fraction=0.15,
            max_gap=0.001,  # very tight => hard block at 0.003
        )

        target_col = "target_10d"
        if target_col not in targets.columns:
            target_col = targets.columns[0]

        result = trainer.train_ensemble(
            features=features,
            targets=targets[target_col],
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
        )

        # The global model should be rejected (None) because the CV gap
        # exceeds the hard block threshold
        assert result.global_model is None, (
            "Overfit model should be rejected by CV gap hard block, "
            "but global_model was not None"
        )


class TestRegimeTradePolicy:
    """SPEC-E02: Verify per-regime trade gating via REGIME_TRADE_POLICY."""

    @staticmethod
    def _make_price_data(permno: str, n_days: int = 100, seed: int = 123):
        dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
        rng = np.random.RandomState(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n_days)))
        price_data = {
            permno: pd.DataFrame(
                {
                    "Open": close * (1 + rng.normal(0, 0.002, n_days)),
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": np.full(n_days, 1_000_000),
                },
                index=dates,
            )
        }
        return dates, price_data

    @staticmethod
    def _make_predictions(permno, dates, regime, confidence, n_signals=5, offset=10):
        signal_dates = dates[offset:offset + n_signals]
        return pd.DataFrame(
            {
                "predicted_return": np.full(n_signals, 0.02),
                "confidence": np.full(n_signals, confidence),
                "regime": np.full(n_signals, regime, dtype=int),
            },
            index=pd.MultiIndex.from_arrays(
                [np.full(n_signals, permno), signal_dates],
                names=["permno", "date"],
            ),
        )

    def test_regime_2_low_confidence_suppressed(self):
        """Regime 2 (disabled, min_confidence=0.70): signals with confidence
        below 0.70 should be suppressed."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10000")
        predictions = self._make_predictions(
            "10000", dates, regime=2, confidence=0.65, n_signals=20,
        )

        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.6,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades == 0, (
            f"Expected 0 trades (regime 2 suppressed, conf 0.65 < 0.70), "
            f"got {result.total_trades}"
        )

    def test_regime_2_high_confidence_allowed(self):
        """Regime 2 (disabled, min_confidence=0.70): signals with confidence
        >= 0.70 should be allowed through (high-confidence override)."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10000", seed=789)
        predictions = self._make_predictions(
            "10000", dates, regime=2, confidence=0.80, n_signals=5,
        )

        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.6,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades > 0, (
            "Regime 2 signals with confidence >= 0.70 should be allowed "
            "(high-confidence override), but got 0 trades"
        )

    def test_regime_0_not_suppressed(self):
        """Regime 0 (enabled, min_confidence=0.0): signals should never be
        suppressed regardless of confidence."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10001", seed=456)
        predictions = self._make_predictions(
            "10001", dates, regime=0, confidence=0.8, n_signals=5,
        )

        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.6,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades > 0, (
            "Regime 0 signals should not be suppressed but got 0 trades"
        )

    def test_regime_3_below_min_confidence_suppressed(self):
        """Regime 3 (enabled, min_confidence=0.60): signals with confidence
        below 0.60 should be suppressed by the per-regime floor."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10002", seed=999)
        predictions = self._make_predictions(
            "10002", dates, regime=3, confidence=0.55, n_signals=10,
        )

        # Use a LOW base confidence_threshold so signals pass the base
        # filter but still fail the per-regime min_confidence for regime 3.
        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.50,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades == 0, (
            f"Expected 0 trades (regime 3 min_confidence=0.60, signal conf=0.55), "
            f"got {result.total_trades}"
        )

    def test_regime_3_above_min_confidence_allowed(self):
        """Regime 3 (enabled, min_confidence=0.60): signals meeting the
        confidence floor should be allowed."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10003", seed=111)
        predictions = self._make_predictions(
            "10003", dates, regime=3, confidence=0.75, n_signals=5,
        )

        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.6,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades > 0, (
            "Regime 3 signals with confidence >= 0.60 should be allowed "
            "but got 0 trades"
        )

    def test_unknown_regime_not_suppressed(self):
        """Signals with an unknown regime ID should use the permissive
        default policy (enabled=True, min_confidence=0.0)."""
        from quant_engine.backtest.engine import Backtester

        dates, price_data = self._make_price_data("10004", seed=222)
        predictions = self._make_predictions(
            "10004", dates, regime=99, confidence=0.8, n_signals=5,
        )

        backtester = Backtester(
            entry_threshold=0.005, confidence_threshold=0.6,
            holding_days=5, max_positions=10,
        )
        result = backtester.run(predictions, price_data, verbose=False)

        assert result.total_trades > 0, (
            "Unknown regime should use permissive default, but got 0 trades"
        )


class TestCrossSectionalRanking:
    """Verify cross-sectional ranker produces valid output."""

    def test_cross_sectional_rank_basic(self):
        """Basic ranking: stocks get percentile ranks within each date."""
        from quant_engine.models.cross_sectional import cross_sectional_rank

        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        n_stocks = 4
        rows = []
        rng = np.random.RandomState(77)
        for dt in dates:
            for stock_id in range(n_stocks):
                rows.append(
                    {
                        "date": dt,
                        "asset": f"S{stock_id}",
                        "predicted_return": rng.normal(0, 0.02),
                    }
                )
        df = pd.DataFrame(rows)

        result = cross_sectional_rank(
            df,
            date_col="date",
            prediction_col="predicted_return",
            asset_col="asset",
        )

        # Output should contain the new columns
        assert "cs_rank" in result.columns
        assert "cs_zscore" in result.columns
        assert "long_short_signal" in result.columns

        # cs_rank should be in [0, 1]
        assert result["cs_rank"].min() >= 0.0
        assert result["cs_rank"].max() <= 1.0

        # long_short_signal values should be in {-1, 0, 1}
        assert set(result["long_short_signal"].unique()).issubset({-1, 0, 1})

    def test_cross_sectional_rank_multiindex(self):
        """Ranking works with MultiIndex (permno, date) DataFrames."""
        from quant_engine.models.cross_sectional import cross_sectional_rank

        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        n_stocks = 6
        rng = np.random.RandomState(88)

        permnos = []
        date_vals = []
        preds = []
        for dt in dates:
            for s in range(n_stocks):
                permnos.append(f"{10000 + s}")
                date_vals.append(dt)
                preds.append(rng.normal(0, 0.015))

        idx = pd.MultiIndex.from_arrays(
            [permnos, date_vals], names=["permno", "date"]
        )
        df = pd.DataFrame({"predicted_return": preds}, index=idx)

        result = cross_sectional_rank(
            df,
            date_col="date",
            prediction_col="predicted_return",
        )

        assert "cs_rank" in result.columns
        assert len(result) == len(df)

        # Within each date, ranks should span the full range
        for dt in dates:
            day_data = result[result["date"] == dt]
            ranks = day_data["cs_rank"]
            assert ranks.min() < 0.5, "Lowest rank should be below 0.5"
            assert ranks.max() > 0.5, "Highest rank should be above 0.5"

    def test_cross_sectional_rank_zscore_centered(self):
        """Z-scores should be approximately mean-zero within each date."""
        from quant_engine.models.cross_sectional import cross_sectional_rank

        dates = pd.date_range("2024-06-01", periods=5, freq="B")
        n_stocks = 20
        rng = np.random.RandomState(99)

        rows = []
        for dt in dates:
            for s in range(n_stocks):
                rows.append(
                    {
                        "date": dt,
                        "predicted_return": rng.normal(0.01, 0.03),
                    }
                )
        df = pd.DataFrame(rows)

        result = cross_sectional_rank(
            df,
            date_col="date",
            prediction_col="predicted_return",
        )

        # Within each date, z-score mean should be ~0
        for dt in dates:
            day_z = result.loc[result["date"] == dt, "cs_zscore"]
            assert abs(day_z.mean()) < 1e-10, (
                f"Z-score mean should be ~0, got {day_z.mean():.6f}"
            )

    def test_cross_sectional_rank_signals_count(self):
        """Long/short signals respect quantile thresholds."""
        from quant_engine.models.cross_sectional import cross_sectional_rank

        dates = pd.date_range("2024-01-02", periods=1, freq="B")
        n_stocks = 100
        rng = np.random.RandomState(111)

        rows = []
        for s in range(n_stocks):
            rows.append(
                {
                    "date": dates[0],
                    "predicted_return": rng.normal(0, 0.02),
                }
            )
        df = pd.DataFrame(rows)

        result = cross_sectional_rank(
            df,
            date_col="date",
            prediction_col="predicted_return",
            long_quantile=0.80,
            short_quantile=0.20,
        )

        n_long = (result["long_short_signal"] == 1).sum()
        n_short = (result["long_short_signal"] == -1).sum()

        # With 100 stocks, top 20% = 20 longs, bottom 20% = 20 shorts
        assert n_long >= 15, f"Expected ~20 longs, got {n_long}"
        assert n_short >= 15, f"Expected ~20 shorts, got {n_short}"
        assert n_long <= 25, f"Expected ~20 longs, got {n_long}"
        assert n_short <= 25, f"Expected ~20 shorts, got {n_short}"
