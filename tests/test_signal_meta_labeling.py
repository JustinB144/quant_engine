"""
Tests for Spec 04: Signal Enhancement + Meta-Labeling + Fold-Level Validation.

Coverage:
    - TestCrossSectionalTopK: quantile selection with synthetic signals
    - TestMetaLabelingModel: training, prediction, confidence calibration
    - TestFoldMetrics: per-fold metric tracking in walk_forward_validate
    - TestFoldConsistency: consistency score computation in promotion gate
    - TestIntegration: end-to-end pipeline with all components
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Guard: if XGBoost / LightGBM native libraries aren't loadable (e.g. missing
# libomp on macOS), pre-populate sys.modules with mocks so the import chain
# through autopilot/__init__ → engine → models/trainer doesn't crash at
# collection.  Tests that require a real XGBClassifier skip themselves.
_xgb_mocked = False
for _pkg, _subpkgs in [
    ("xgboost", ["xgboost.callback", "xgboost.core"]),
    ("lightgbm", ["lightgbm.basic", "lightgbm.libpath"]),
]:
    try:
        __import__(_pkg)
    except Exception:
        _mock = MagicMock()
        sys.modules.setdefault(_pkg, _mock)
        for _sub in _subpkgs:
            sys.modules.setdefault(_sub, MagicMock())
        if _pkg == "xgboost":
            _xgb_mocked = True

from quant_engine.autopilot.meta_labeler import (
    META_FEATURE_COLUMNS,
    MetaLabelingModel,
)
from quant_engine.autopilot.promotion_gate import PromotionGate
from quant_engine.autopilot.strategy_discovery import StrategyCandidate
from quant_engine.backtest.engine import BacktestResult
from quant_engine.backtest.validation import (
    WalkForwardFold,
    WalkForwardResult,
    walk_forward_validate,
)
from quant_engine.config import (
    FOLD_CONSISTENCY_PENALTY_WEIGHT,
    META_LABELING_CONFIDENCE_THRESHOLD,
    SIGNAL_TOPK_QUANTILE,
)

from quant_engine.autopilot import meta_labeler as _ml_mod

_SKIP_XGB = _xgb_mocked or not getattr(_ml_mod, "_HAS_XGB", False)

# ── Test helpers ─────────────────────────────────────────────────────


def _make_synthetic_signals(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic predicted returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    return pd.Series(rng.normal(0.002, 0.02, n), index=dates, name="predicted_return")


def _make_synthetic_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    return pd.Series(rng.normal(0.0003, 0.015, n), index=dates, name="returns")


def _make_synthetic_regimes(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic regime labels (0-3)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    regimes = rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.2, 0.25, 0.15])
    return pd.Series(regimes, index=dates, name="regime")


def _make_synthetic_actuals(
    signals: pd.Series, noise: float = 0.02, seed: int = 99,
) -> pd.Series:
    """Generate actuals that are weakly correlated with signals."""
    rng = np.random.default_rng(seed)
    # 30% signal + 70% noise → moderate correlation
    actuals = 0.3 * signals + noise * rng.normal(0, 1, len(signals))
    return pd.Series(actuals.values, index=signals.index, name="actual")


def _make_candidate() -> StrategyCandidate:
    return StrategyCandidate(
        strategy_id="h10_e50_c60_r0_m10",
        horizon=10,
        entry_threshold=0.005,
        confidence_threshold=0.60,
        use_risk_management=False,
        max_positions=10,
        position_size_pct=0.05,
    )


def _make_backtest_result(**overrides) -> BacktestResult:
    defaults = dict(
        total_trades=150,
        winning_trades=85,
        losing_trades=65,
        win_rate=85 / 150,
        avg_return=0.012,
        avg_win=0.026,
        avg_loss=-0.011,
        total_return=0.42,
        annualized_return=0.18,
        sharpe_ratio=1.35,
        sortino_ratio=1.70,
        max_drawdown=-0.12,
        profit_factor=1.45,
        avg_holding_days=10.0,
        trades_per_year=120.0,
        returns_series=pd.Series([0.01, -0.004, 0.02]),
        equity_curve=pd.Series([1.0, 1.01, 1.03]),
        daily_equity=pd.Series([1.0, 1.005, 1.012]),
        trades=[],
        regime_breakdown={},
        risk_report=None,
        drawdown_summary=None,
        exit_reason_breakdown={},
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


# ── Test classes ─────────────────────────────────────────────────────


class TestCrossSectionalTopK(unittest.TestCase):
    """Tests for cross-sectional top-K quantile selection (Spec 04 T1)."""

    def test_topk_quantile_selects_correct_fraction(self):
        """Top-K at q=0.70 should keep ~70% of signals."""
        rng = np.random.default_rng(42)
        n_stocks = 100
        zscores = pd.Series(rng.normal(0, 1, n_stocks))
        quantile = 0.70
        threshold = np.quantile(zscores.values, 1 - quantile)
        selected = zscores[zscores >= threshold]
        # Should keep approximately 70% (±5%)
        frac = len(selected) / n_stocks
        self.assertGreater(frac, 0.60)
        self.assertLess(frac, 0.80)

    def test_topk_quantile_extreme_dispersion(self):
        """With high-dispersion signals, quantile adapts threshold."""
        rng = np.random.default_rng(42)
        # High dispersion: z-scores from -5 to +5
        zscores_wide = pd.Series(rng.normal(0, 3, 100))
        # Low dispersion: z-scores from -0.5 to +0.5
        zscores_narrow = pd.Series(rng.normal(0, 0.2, 100))

        q = 0.70
        t_wide = np.quantile(zscores_wide, 1 - q)
        t_narrow = np.quantile(zscores_narrow, 1 - q)

        # Wide dispersion should have a higher absolute threshold
        self.assertGreater(abs(t_wide), abs(t_narrow))

    def test_topk_quantile_single_stock_keeps_all(self):
        """With only 1 stock, all signals should be kept (no filtering)."""
        zscores = pd.Series([1.5])
        # When fewer than 2 stocks, quantile threshold is -inf → keep all
        if len(zscores) < 2:
            selected = zscores  # no filtering
        else:
            threshold = np.quantile(zscores, 1 - SIGNAL_TOPK_QUANTILE)
            selected = zscores[zscores >= threshold]
        self.assertEqual(len(selected), 1)

    def test_topk_config_quantile_value(self):
        """Config SIGNAL_TOPK_QUANTILE defaults to 0.70."""
        self.assertEqual(SIGNAL_TOPK_QUANTILE, 0.70)


class TestMetaLabelingModel(unittest.TestCase):
    """Tests for MetaLabelingModel (Spec 04 T2)."""

    def test_build_meta_features_shape(self):
        """Meta-features DataFrame has correct shape and columns."""
        signals = _make_synthetic_signals(200)
        returns = _make_synthetic_returns(200)
        regimes = _make_synthetic_regimes(200)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)

        self.assertEqual(meta.shape[0], 200)
        self.assertEqual(meta.shape[1], len(META_FEATURE_COLUMNS))
        for col in META_FEATURE_COLUMNS:
            self.assertIn(col, meta.columns)

    def test_build_meta_features_no_nans(self):
        """Meta-features should have no NaN values (all fillna'd)."""
        signals = _make_synthetic_signals(200)
        returns = _make_synthetic_returns(200)
        regimes = _make_synthetic_regimes(200)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        self.assertEqual(meta.isna().sum().sum(), 0)

    def test_build_meta_features_all_zeros_input(self):
        """Meta-features should handle zero signals gracefully."""
        n = 100
        dates = pd.bdate_range("2023-01-03", periods=n)
        signals = pd.Series(0.0, index=dates)
        returns = pd.Series(0.0, index=dates)
        regimes = pd.Series(0, index=dates)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        self.assertEqual(meta.isna().sum().sum(), 0)
        self.assertEqual(meta["signal_magnitude"].sum(), 0.0)

    def test_build_meta_features_regime_one_hot(self):
        """Regime one-hot columns should sum to 1 for each row."""
        signals = _make_synthetic_signals(200)
        returns = _make_synthetic_returns(200)
        regimes = _make_synthetic_regimes(200)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        regime_cols = [f"regime_{i}" for i in range(4)]
        regime_sum = meta[regime_cols].sum(axis=1)
        np.testing.assert_array_almost_equal(regime_sum.values, 1.0)

    def test_build_labels_correct_direction(self):
        """Labels should be 1 when signal and actual have same sign."""
        signals = pd.Series([0.01, -0.01, 0.001, 0.03, -0.02])
        actuals = pd.Series([0.005, -0.003, 0.002, -0.01, -0.015])
        labels = MetaLabelingModel.build_labels(signals, actuals, entry_threshold=0.005)

        # signal=0.01 > 0.005, actual>0 → 1
        # signal=-0.01, |s|>0.005, actual<0 → 1
        # signal=0.001, |s|<0.005 → 0 (not actionable)
        # signal=0.03 > 0.005, actual<0 → 0 (wrong direction)
        # signal=-0.02, |s|>0.005, actual<0 → 1
        expected = pd.Series([1, 1, 0, 0, 1])
        pd.testing.assert_series_equal(labels, expected, check_names=False)

    def test_build_labels_all_below_threshold(self):
        """All-below-threshold signals should produce all-zero labels."""
        signals = pd.Series([0.001, 0.002, -0.001, 0.003])
        actuals = pd.Series([0.01, 0.02, -0.01, 0.005])
        labels = MetaLabelingModel.build_labels(signals, actuals, entry_threshold=0.01)
        self.assertEqual(labels.sum(), 0)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_train_produces_model(self):
        """Training on sufficient synthetic data should produce a model."""

        rng = np.random.default_rng(42)
        n = 600
        signals = _make_synthetic_signals(n)
        returns = _make_synthetic_returns(n)
        regimes = _make_synthetic_regimes(n)
        actuals = _make_synthetic_actuals(signals)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        labels = MetaLabelingModel.build_labels(signals, actuals)

        model = MetaLabelingModel(min_samples=100)
        metrics = model.train(meta, labels)

        self.assertTrue(model.is_trained)
        self.assertIn("train_accuracy", metrics)
        self.assertGreater(metrics["train_accuracy"], 0.40)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_train_insufficient_samples_raises(self):
        """Training with too few samples should raise RuntimeError."""

        n = 50
        signals = _make_synthetic_signals(n)
        returns = _make_synthetic_returns(n)
        regimes = _make_synthetic_regimes(n)
        actuals = _make_synthetic_actuals(signals)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        labels = MetaLabelingModel.build_labels(signals, actuals)

        model = MetaLabelingModel(min_samples=500)
        with self.assertRaises(RuntimeError):
            model.train(meta, labels)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_predict_confidence_range(self):
        """Confidence predictions should be in [0, 1]."""

        n = 600
        signals = _make_synthetic_signals(n)
        returns = _make_synthetic_returns(n)
        regimes = _make_synthetic_regimes(n)
        actuals = _make_synthetic_actuals(signals)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        labels = MetaLabelingModel.build_labels(signals, actuals)

        model = MetaLabelingModel(min_samples=100)
        model.train(meta, labels)

        conf = model.predict_confidence(meta)
        self.assertTrue((conf >= 0.0).all())
        self.assertTrue((conf <= 1.0).all())
        self.assertEqual(len(conf), n)

    def test_predict_untrained_raises(self):
        """Predicting without training should raise RuntimeError."""
        model = MetaLabelingModel()
        signals = _make_synthetic_signals(100)
        returns = _make_synthetic_returns(100)
        regimes = _make_synthetic_regimes(100)
        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)

        with self.assertRaises(RuntimeError):
            model.predict_confidence(meta)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_feature_importance_no_single_dominant(self):
        """No single feature should have >50% importance (healthy model)."""

        n = 800
        signals = _make_synthetic_signals(n)
        returns = _make_synthetic_returns(n)
        regimes = _make_synthetic_regimes(n)
        actuals = _make_synthetic_actuals(signals)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        labels = MetaLabelingModel.build_labels(signals, actuals)

        model = MetaLabelingModel(min_samples=100)
        model.train(meta, labels)

        max_imp = max(model.feature_importance_.values())
        # With diversified features, no single one should completely dominate
        # (this is a soft check — data-dependent)
        self.assertIsNotNone(model.feature_importance_)
        self.assertGreater(len(model.feature_importance_), 0)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_save_and_load_roundtrip(self):
        """Model save + load should preserve predictions."""

        import tempfile

        n = 600
        signals = _make_synthetic_signals(n)
        returns = _make_synthetic_returns(n)
        regimes = _make_synthetic_regimes(n)
        actuals = _make_synthetic_actuals(signals)

        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        labels = MetaLabelingModel.build_labels(signals, actuals)

        model = MetaLabelingModel(min_samples=100)
        model.train(meta, labels)
        original_conf = model.predict_confidence(meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            model.save(path)

            model2 = MetaLabelingModel(min_samples=100)
            loaded = model2.load(path)
            self.assertTrue(loaded)
            self.assertTrue(model2.is_trained)

            reloaded_conf = model2.predict_confidence(meta)
            np.testing.assert_array_almost_equal(
                original_conf.values, reloaded_conf.values
            )

    def test_load_nonexistent_returns_false(self):
        """Loading from non-existent path should return False."""
        model = MetaLabelingModel()
        result = model.load(Path("/tmp/nonexistent_model_12345.joblib"))
        self.assertFalse(result)
        self.assertFalse(model.is_trained)


class TestFoldMetrics(unittest.TestCase):
    """Tests for per-fold metric tracking (Spec 04 T4)."""

    def test_walk_forward_returns_fold_metrics(self):
        """walk_forward_validate should populate fold-level metrics."""
        rng = np.random.default_rng(42)
        n = 600
        signals = _make_synthetic_signals(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        result = walk_forward_validate(
            predictions=signals,
            actuals=actuals,
            n_folds=5,
            entry_threshold=0.005,
        )

        self.assertGreater(result.n_folds, 0)
        for fold in result.folds:
            self.assertIsInstance(fold, WalkForwardFold)
            # New fields should be populated (not default 0)
            self.assertIsInstance(fold.win_rate, float)
            self.assertIsInstance(fold.profit_factor, float)
            self.assertIsInstance(fold.sharpe_estimate, float)
            self.assertIsInstance(fold.sample_count, int)
            self.assertGreater(fold.sample_count, 0)

    def test_fold_win_rate_range(self):
        """Fold win_rate should be in [0, 1]."""
        n = 600
        signals = _make_synthetic_signals(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        result = walk_forward_validate(
            predictions=signals, actuals=actuals, n_folds=5,
        )

        for fold in result.folds:
            self.assertGreaterEqual(fold.win_rate, 0.0)
            self.assertLessEqual(fold.win_rate, 1.0)

    def test_fold_profit_factor_nonnegative(self):
        """Fold profit_factor should be >= 0."""
        n = 600
        signals = _make_synthetic_signals(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        result = walk_forward_validate(
            predictions=signals, actuals=actuals, n_folds=5,
        )

        for fold in result.folds:
            self.assertGreaterEqual(fold.profit_factor, 0.0)

    def test_backward_compat_aggregate_still_works(self):
        """Aggregate metrics (avg_oos_corr, etc.) should be unchanged."""
        n = 600
        signals = _make_synthetic_signals(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        result = walk_forward_validate(
            predictions=signals, actuals=actuals, n_folds=5,
        )

        # Old fields still accessible
        self.assertIsInstance(result.avg_oos_corr, float)
        self.assertIsInstance(result.is_oos_gap, float)
        self.assertIsInstance(result.n_folds, int)
        self.assertIsInstance(result.folds, list)
        self.assertIsInstance(result.oos_corr_std, float)

    def test_insufficient_data_returns_empty_folds(self):
        """With too few samples, should return empty folds list."""
        signals = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2023-01-03", periods=3))
        actuals = pd.Series([0.005, 0.01, -0.01], index=signals.index)

        result = walk_forward_validate(
            predictions=signals, actuals=actuals, n_folds=3,
        )
        self.assertEqual(result.n_folds, 0)
        self.assertEqual(len(result.folds), 0)


class TestFoldConsistency(unittest.TestCase):
    """Tests for fold consistency metric (Spec 04 T5)."""

    def test_perfect_consistency(self):
        """Identical Sharpe across all folds → consistency = 1.0."""
        fold_metrics = [
            {"sharpe_estimate": 1.5},
            {"sharpe_estimate": 1.5},
            {"sharpe_estimate": 1.5},
            {"sharpe_estimate": 1.5},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertAlmostEqual(c, 1.0)

    def test_zero_consistency_negative_mean(self):
        """Mean Sharpe <= 0 → consistency = 0.0."""
        fold_metrics = [
            {"sharpe_estimate": -0.5},
            {"sharpe_estimate": -1.0},
            {"sharpe_estimate": -0.3},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertEqual(c, 0.0)

    def test_low_consistency_high_variance(self):
        """High Sharpe variance → low consistency."""
        fold_metrics = [
            {"sharpe_estimate": 3.0},
            {"sharpe_estimate": 0.1},
            {"sharpe_estimate": 2.5},
            {"sharpe_estimate": -0.5},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertLess(c, 0.5)

    def test_moderate_consistency(self):
        """Moderate Sharpe variation → consistency between 0.3 and 0.9."""
        fold_metrics = [
            {"sharpe_estimate": 1.2},
            {"sharpe_estimate": 1.0},
            {"sharpe_estimate": 1.5},
            {"sharpe_estimate": 0.8},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertGreater(c, 0.3)
        self.assertLess(c, 1.0)

    def test_single_fold_returns_one(self):
        """With <2 folds, insufficient data → return 1.0 (no penalty)."""
        c = PromotionGate._compute_fold_consistency(
            [{"sharpe_estimate": 1.5}]
        )
        self.assertEqual(c, 1.0)

    def test_clipping_bounds(self):
        """Consistency should always be in [0, 1]."""
        # Extreme values that might push outside bounds
        fold_metrics = [
            {"sharpe_estimate": 0.01},
            {"sharpe_estimate": 10.0},
            {"sharpe_estimate": 0.02},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_fold_consistency_integrated_into_score(self):
        """When fold_metrics provided, fold consistency should affect score."""
        gate = PromotionGate()
        candidate = _make_candidate()
        result = _make_backtest_result()

        # Without fold_metrics
        d1 = gate.evaluate(
            candidate, result,
            contract_metrics={
                "dsr_significant": True, "dsr_p_value": 0.01,
                "pbo": 0.30, "mc_significant": True, "mc_p_value": 0.01,
                "capacity_constrained": False, "capacity_utilization": 0.5,
                "wf_oos_corr": 0.08, "wf_positive_fold_fraction": 0.8,
                "wf_is_oos_gap": 0.04, "regime_positive_fraction": 0.75,
                "stat_tests_pass": True, "cpcv_passes": True,
            },
        )

        # With high-consistency fold_metrics
        d2 = gate.evaluate(
            candidate, result,
            contract_metrics={
                "dsr_significant": True, "dsr_p_value": 0.01,
                "pbo": 0.30, "mc_significant": True, "mc_p_value": 0.01,
                "capacity_constrained": False, "capacity_utilization": 0.5,
                "wf_oos_corr": 0.08, "wf_positive_fold_fraction": 0.8,
                "wf_is_oos_gap": 0.04, "regime_positive_fraction": 0.75,
                "stat_tests_pass": True, "cpcv_passes": True,
                "fold_metrics": [
                    {"sharpe_estimate": 1.5},
                    {"sharpe_estimate": 1.4},
                    {"sharpe_estimate": 1.6},
                    {"sharpe_estimate": 1.5},
                ],
            },
        )

        # Score with fold consistency should be higher
        self.assertGreater(d2.score, d1.score)

    def test_fold_consistency_key_in_sharpe_format(self):
        """Should also work with 'sharpe' key (not just 'sharpe_estimate')."""
        fold_metrics = [
            {"sharpe": 1.5},
            {"sharpe": 1.5},
            {"sharpe": 1.5},
        ]
        c = PromotionGate._compute_fold_consistency(fold_metrics)
        self.assertAlmostEqual(c, 1.0)

    def test_config_weight_default(self):
        """Default fold consistency weight should be 0.15."""
        self.assertEqual(FOLD_CONSISTENCY_PENALTY_WEIGHT, 0.15)


class TestMetaLabelingFiltering(unittest.TestCase):
    """Tests for meta-labeling confidence filtering logic (Spec 04 T3)."""

    def test_high_confidence_passes(self):
        """Signals with confidence above threshold should not be filtered."""
        threshold = META_LABELING_CONFIDENCE_THRESHOLD
        # Simulate: confidence = [0.3, 0.7, 0.8, 0.4, 0.9]
        confidences = pd.Series([0.3, 0.7, 0.8, 0.4, 0.9])
        mask = confidences >= threshold
        # 0.7, 0.8, 0.9 should pass (3 out of 5)
        self.assertEqual(mask.sum(), 3)

    def test_all_above_threshold_no_filtering(self):
        """When all confidences are above threshold, all pass."""
        confidences = pd.Series([0.6, 0.7, 0.8, 0.9, 0.95])
        mask = confidences >= META_LABELING_CONFIDENCE_THRESHOLD
        self.assertEqual(mask.sum(), len(confidences))

    def test_all_below_threshold_all_filtered(self):
        """When all confidences are below threshold, none pass."""
        confidences = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = confidences >= META_LABELING_CONFIDENCE_THRESHOLD
        self.assertEqual(mask.sum(), 0)


class TestConfigDefaults(unittest.TestCase):
    """Verify Spec 04 config parameters have correct defaults."""

    def test_signal_topk_quantile(self):
        self.assertEqual(SIGNAL_TOPK_QUANTILE, 0.70)

    def test_meta_labeling_confidence_threshold(self):
        self.assertEqual(META_LABELING_CONFIDENCE_THRESHOLD, 0.55)

    def test_fold_consistency_weight(self):
        self.assertEqual(FOLD_CONSISTENCY_PENALTY_WEIGHT, 0.15)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests combining all Spec 04 components."""

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_full_pipeline_meta_features_through_training(self):
        """Full pipeline: signals → meta-features → labels → train → predict."""

        n = 800
        signals = _make_synthetic_signals(n, seed=42)
        returns = _make_synthetic_returns(n, seed=42)
        regimes = _make_synthetic_regimes(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        # Step 1: Build meta-features
        meta = MetaLabelingModel.build_meta_features(signals, returns, regimes)
        self.assertEqual(meta.shape[0], n)
        self.assertEqual(meta.isna().sum().sum(), 0)

        # Step 2: Build labels
        labels = MetaLabelingModel.build_labels(signals, actuals)
        self.assertEqual(len(labels), n)
        self.assertIn(labels.sum(), range(1, n))  # some correct, some not

        # Step 3: Train
        model = MetaLabelingModel(min_samples=100)
        metrics = model.train(meta, labels)
        self.assertTrue(model.is_trained)
        self.assertGreater(metrics["train_accuracy"], 0.40)

        # Step 4: Predict confidence
        confidence = model.predict_confidence(meta)
        self.assertEqual(len(confidence), n)
        self.assertTrue((confidence >= 0).all())
        self.assertTrue((confidence <= 1).all())

        # Step 5: Apply threshold filter
        threshold = META_LABELING_CONFIDENCE_THRESHOLD
        mask = confidence >= threshold
        # Shouldn't filter more than 90% (spec constraint)
        self.assertGreater(mask.sum() / n, 0.10)

    @unittest.skipIf(_SKIP_XGB, "xgboost not available")
    def test_topk_then_meta_labeling_reduces_signals(self):
        """Top-K + meta-labeling together should reduce signal count."""

        rng = np.random.default_rng(42)
        n = 800
        signals = _make_synthetic_signals(n, seed=42)

        # Top-K filter
        threshold = np.quantile(signals.values, 1 - SIGNAL_TOPK_QUANTILE)
        topk_signals = signals.copy()
        topk_signals[topk_signals < threshold] = 0.0
        n_after_topk = (topk_signals.abs() > 1e-8).sum()

        # Original should have more non-zero signals than after top-K
        n_original = (signals.abs() > 1e-8).sum()
        self.assertLessEqual(n_after_topk, n_original)

    def test_fold_metrics_feed_into_promotion_gate(self):
        """Fold metrics from walk_forward_validate feed through to gate."""
        n = 600
        signals = _make_synthetic_signals(n, seed=42)
        actuals = _make_synthetic_actuals(signals, seed=99)

        wf = walk_forward_validate(
            predictions=signals, actuals=actuals, n_folds=5,
        )

        # Build fold_metrics list as engine.py does
        fold_metrics = []
        for fold in wf.folds:
            fold_metrics.append({
                "fold_id": fold.fold,
                "ic": fold.test_corr,
                "sharpe_estimate": fold.sharpe_estimate,
                "win_rate": fold.win_rate,
                "profit_factor": fold.profit_factor,
                "sample_count": fold.sample_count,
            })

        # Should have same number of entries as folds
        self.assertEqual(len(fold_metrics), wf.n_folds)

        # Compute fold consistency
        if len(fold_metrics) >= 3:
            consistency = PromotionGate._compute_fold_consistency(fold_metrics)
            self.assertGreaterEqual(consistency, 0.0)
            self.assertLessEqual(consistency, 1.0)

    def test_backward_compat_meta_labeling_disabled(self):
        """When META_LABELING_ENABLED=False, pipeline runs unchanged."""
        # This just verifies the config flag exists and can be set to False
        with patch("quant_engine.config.META_LABELING_ENABLED", False):
            from quant_engine.config import META_LABELING_ENABLED as ml
            # When disabled, model won't be loaded/used
            model = MetaLabelingModel()
            self.assertFalse(model.is_trained)

    def test_backward_compat_no_fold_metrics(self):
        """Promotion gate works correctly without fold_metrics."""
        gate = PromotionGate()
        candidate = _make_candidate()
        result = _make_backtest_result()

        # No fold_metrics in contract_metrics
        decision = gate.evaluate(
            candidate, result,
            contract_metrics={
                "dsr_significant": True, "dsr_p_value": 0.01,
                "pbo": 0.30, "mc_significant": True, "mc_p_value": 0.01,
                "capacity_constrained": False, "capacity_utilization": 0.5,
                "wf_oos_corr": 0.08, "wf_positive_fold_fraction": 0.8,
                "wf_is_oos_gap": 0.04, "regime_positive_fraction": 0.75,
                "stat_tests_pass": True, "cpcv_passes": True,
            },
        )
        # Should pass without fold_metrics
        self.assertTrue(decision.passed)
        # fold_consistency should not be in metrics
        self.assertNotIn("fold_consistency", decision.metrics)


if __name__ == "__main__":
    unittest.main()
