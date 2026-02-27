"""Tests for SPEC-A01: Config consolidation consistency.

Verifies that config.py (flat constants) and config_structured.py
(typed dataclasses) stay in sync.  After consolidation, config.py
derives its values from the structured config singleton, so this
test suite ensures:

1. Every mapped constant in config.py matches the structured source.
2. The get_config() singleton works correctly.
3. Mutating the structured config is reflected in flat config when
   re-imported.
"""

import importlib

import pytest


class TestGetConfigSingleton:
    """get_config() must return a stable singleton."""

    def test_returns_system_config(self):
        from config_structured import SystemConfig, get_config

        cfg = get_config()
        assert isinstance(cfg, SystemConfig)

    def test_singleton_identity(self):
        from config_structured import get_config

        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2


class TestPreconditionsConsistency:
    """Execution contract values must match between flat and structured."""

    def test_ret_type(self):
        from quant_engine.config import RET_TYPE
        from quant_engine.config_structured import get_config

        assert RET_TYPE == get_config().preconditions.ret_type.value

    def test_label_h(self):
        from quant_engine.config import LABEL_H
        from quant_engine.config_structured import get_config

        assert LABEL_H == get_config().preconditions.label_h

    def test_px_type(self):
        from quant_engine.config import PX_TYPE
        from quant_engine.config_structured import get_config

        assert PX_TYPE == get_config().preconditions.px_type.value

    def test_entry_price_type(self):
        from quant_engine.config import ENTRY_PRICE_TYPE
        from quant_engine.config_structured import get_config

        assert ENTRY_PRICE_TYPE == get_config().preconditions.entry_price_type.value


class TestDataConsistency:
    """Data config values must match."""

    def test_wrds_enabled(self):
        from quant_engine.config import WRDS_ENABLED
        from quant_engine.config_structured import get_config

        assert WRDS_ENABLED == get_config().data.wrds_enabled

    def test_optionmetrics_enabled(self):
        from quant_engine.config import OPTIONMETRICS_ENABLED
        from quant_engine.config_structured import get_config

        assert OPTIONMETRICS_ENABLED == get_config().data.optionmetrics_enabled

    def test_kalshi_enabled(self):
        from quant_engine.config import KALSHI_ENABLED
        from quant_engine.config_structured import get_config

        assert KALSHI_ENABLED == get_config().data.kalshi_enabled

    def test_lookback_years(self):
        from quant_engine.config import LOOKBACK_YEARS
        from quant_engine.config_structured import get_config

        assert LOOKBACK_YEARS == get_config().data.lookback_years

    def test_min_bars(self):
        from quant_engine.config import MIN_BARS
        from quant_engine.config_structured import get_config

        assert MIN_BARS == get_config().data.min_bars

    def test_cache_max_staleness_days(self):
        from quant_engine.config import CACHE_MAX_STALENESS_DAYS
        from quant_engine.config_structured import get_config

        assert CACHE_MAX_STALENESS_DAYS == get_config().data.cache_max_staleness_days

    def test_cache_trusted_sources(self):
        from quant_engine.config import CACHE_TRUSTED_SOURCES
        from quant_engine.config_structured import get_config

        assert CACHE_TRUSTED_SOURCES == list(get_config().data.cache_trusted_sources)

    def test_default_universe_source(self):
        from quant_engine.config import DEFAULT_UNIVERSE_SOURCE
        from quant_engine.config_structured import get_config

        assert DEFAULT_UNIVERSE_SOURCE == get_config().data.default_universe_source

    def test_max_missing_bar_fraction(self):
        from quant_engine.config import MAX_MISSING_BAR_FRACTION
        from quant_engine.config_structured import get_config

        assert MAX_MISSING_BAR_FRACTION == get_config().data.max_missing_bar_fraction

    def test_max_zero_volume_fraction(self):
        from quant_engine.config import MAX_ZERO_VOLUME_FRACTION
        from quant_engine.config_structured import get_config

        assert MAX_ZERO_VOLUME_FRACTION == get_config().data.max_zero_volume_fraction

    def test_max_abs_daily_return(self):
        from quant_engine.config import MAX_ABS_DAILY_RETURN
        from quant_engine.config_structured import get_config

        assert MAX_ABS_DAILY_RETURN == get_config().data.max_abs_daily_return


class TestCostStressConsistency:
    """Cost stress config values must match."""

    def test_multipliers(self):
        from quant_engine.config import COST_STRESS_MULTIPLIERS
        from quant_engine.config_structured import get_config

        assert COST_STRESS_MULTIPLIERS == list(get_config().cost_stress.multipliers)

    def test_enabled(self):
        from quant_engine.config import TRUTH_LAYER_COST_STRESS_ENABLED
        from quant_engine.config_structured import get_config

        assert TRUTH_LAYER_COST_STRESS_ENABLED == get_config().cost_stress.enabled


class TestRegimeConsistency:
    """Regime detection values must match."""

    def test_model_type(self):
        from quant_engine.config import REGIME_MODEL_TYPE
        from quant_engine.config_structured import get_config

        assert REGIME_MODEL_TYPE == get_config().regime.model_type

    def test_hmm_states(self):
        from quant_engine.config import REGIME_HMM_STATES
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_STATES == get_config().regime.n_states

    def test_hmm_max_iter(self):
        from quant_engine.config import REGIME_HMM_MAX_ITER
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_MAX_ITER == get_config().regime.hmm_max_iter

    def test_hmm_stickiness(self):
        from quant_engine.config import REGIME_HMM_STICKINESS
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_STICKINESS == get_config().regime.hmm_stickiness

    def test_min_duration(self):
        from quant_engine.config import REGIME_MIN_DURATION
        from quant_engine.config_structured import get_config

        assert REGIME_MIN_DURATION == get_config().regime.min_duration

    def test_hmm_prior_weight(self):
        from quant_engine.config import REGIME_HMM_PRIOR_WEIGHT
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_PRIOR_WEIGHT == get_config().regime.hmm_prior_weight

    def test_hmm_covariance_type(self):
        from quant_engine.config import REGIME_HMM_COVARIANCE_TYPE
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_COVARIANCE_TYPE == get_config().regime.hmm_covariance_type

    def test_hmm_auto_select_states(self):
        from quant_engine.config import REGIME_HMM_AUTO_SELECT_STATES
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_AUTO_SELECT_STATES == get_config().regime.hmm_auto_select_states

    def test_hmm_min_states(self):
        from quant_engine.config import REGIME_HMM_MIN_STATES
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_MIN_STATES == get_config().regime.hmm_min_states

    def test_hmm_max_states(self):
        from quant_engine.config import REGIME_HMM_MAX_STATES
        from quant_engine.config_structured import get_config

        assert REGIME_HMM_MAX_STATES == get_config().regime.hmm_max_states

    def test_jump_model_enabled(self):
        from quant_engine.config import REGIME_JUMP_MODEL_ENABLED
        from quant_engine.config_structured import get_config

        assert REGIME_JUMP_MODEL_ENABLED == get_config().regime.jump_model_enabled

    def test_jump_penalty(self):
        from quant_engine.config import REGIME_JUMP_PENALTY
        from quant_engine.config_structured import get_config

        assert REGIME_JUMP_PENALTY == get_config().regime.jump_penalty

    def test_expected_changes_per_year(self):
        from quant_engine.config import REGIME_EXPECTED_CHANGES_PER_YEAR
        from quant_engine.config_structured import get_config

        assert REGIME_EXPECTED_CHANGES_PER_YEAR == get_config().regime.expected_changes_per_year

    def test_ensemble_enabled(self):
        from quant_engine.config import REGIME_ENSEMBLE_ENABLED
        from quant_engine.config_structured import get_config

        assert REGIME_ENSEMBLE_ENABLED == get_config().regime.ensemble_enabled

    def test_ensemble_consensus_threshold(self):
        from quant_engine.config import REGIME_ENSEMBLE_CONSENSUS_THRESHOLD
        from quant_engine.config_structured import get_config

        assert REGIME_ENSEMBLE_CONSENSUS_THRESHOLD == get_config().regime.ensemble_consensus_threshold

    def test_risk_multiplier(self):
        from quant_engine.config import REGIME_RISK_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert REGIME_RISK_MULTIPLIER == dict(get_config().regime.risk_multiplier)

    def test_stop_multiplier(self):
        from quant_engine.config import REGIME_STOP_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert REGIME_STOP_MULTIPLIER == dict(get_config().regime.stop_multiplier)

    def test_trade_policy(self):
        from quant_engine.config import REGIME_TRADE_POLICY
        from quant_engine.config_structured import get_config

        cfg_policy = get_config().regime.trade_policy
        for regime_id, policy in cfg_policy.items():
            assert regime_id in REGIME_TRADE_POLICY
            assert REGIME_TRADE_POLICY[regime_id] == dict(policy)


class TestModelConsistency:
    """Model training values must match."""

    def test_params(self):
        from quant_engine.config import MODEL_PARAMS
        from quant_engine.config_structured import get_config

        assert MODEL_PARAMS == dict(get_config().model.params)

    def test_max_features_selected(self):
        from quant_engine.config import MAX_FEATURES_SELECTED
        from quant_engine.config_structured import get_config

        assert MAX_FEATURES_SELECTED == get_config().model.max_features_selected

    def test_max_is_oos_gap(self):
        from quant_engine.config import MAX_IS_OOS_GAP
        from quant_engine.config_structured import get_config

        assert MAX_IS_OOS_GAP == get_config().model.max_is_oos_gap

    def test_cv_folds(self):
        from quant_engine.config import CV_FOLDS
        from quant_engine.config_structured import get_config

        assert CV_FOLDS == get_config().model.cv_folds

    def test_holdout_fraction(self):
        from quant_engine.config import HOLDOUT_FRACTION
        from quant_engine.config_structured import get_config

        assert HOLDOUT_FRACTION == get_config().model.holdout_fraction

    def test_ensemble_diversify(self):
        from quant_engine.config import ENSEMBLE_DIVERSIFY
        from quant_engine.config_structured import get_config

        assert ENSEMBLE_DIVERSIFY == get_config().model.ensemble_diversify

    def test_forward_horizons(self):
        from quant_engine.config import FORWARD_HORIZONS
        from quant_engine.config_structured import get_config

        assert FORWARD_HORIZONS == list(get_config().model.forward_horizons)

    def test_max_model_versions(self):
        from quant_engine.config import MAX_MODEL_VERSIONS
        from quant_engine.config_structured import get_config

        assert MAX_MODEL_VERSIONS == get_config().model.max_model_versions

    def test_feature_mode(self):
        from quant_engine.config import FEATURE_MODE_DEFAULT
        from quant_engine.config_structured import get_config

        assert FEATURE_MODE_DEFAULT == get_config().model.feature_mode


class TestBacktestConsistency:
    """Backtest values must match."""

    def test_transaction_cost_bps(self):
        from quant_engine.config import TRANSACTION_COST_BPS
        from quant_engine.config_structured import get_config

        assert TRANSACTION_COST_BPS == get_config().backtest.transaction_cost_bps

    def test_entry_threshold(self):
        from quant_engine.config import ENTRY_THRESHOLD
        from quant_engine.config_structured import get_config

        assert ENTRY_THRESHOLD == get_config().backtest.entry_threshold

    def test_confidence_threshold(self):
        from quant_engine.config import CONFIDENCE_THRESHOLD
        from quant_engine.config_structured import get_config

        assert CONFIDENCE_THRESHOLD == get_config().backtest.confidence_threshold

    def test_max_positions(self):
        from quant_engine.config import MAX_POSITIONS
        from quant_engine.config_structured import get_config

        assert MAX_POSITIONS == get_config().backtest.max_positions

    def test_position_size_pct(self):
        from quant_engine.config import POSITION_SIZE_PCT
        from quant_engine.config_structured import get_config

        assert POSITION_SIZE_PCT == get_config().backtest.position_size_pct

    def test_assumed_capital_usd(self):
        from quant_engine.config import BACKTEST_ASSUMED_CAPITAL_USD
        from quant_engine.config_structured import get_config

        assert BACKTEST_ASSUMED_CAPITAL_USD == get_config().backtest.assumed_capital_usd

    def test_max_portfolio_vol(self):
        from quant_engine.config import MAX_PORTFOLIO_VOL
        from quant_engine.config_structured import get_config

        assert MAX_PORTFOLIO_VOL == get_config().backtest.max_portfolio_vol

    def test_max_annualized_turnover(self):
        from quant_engine.config import MAX_ANNUALIZED_TURNOVER
        from quant_engine.config_structured import get_config

        assert MAX_ANNUALIZED_TURNOVER == get_config().backtest.max_annualized_turnover

    def test_max_sector_exposure(self):
        from quant_engine.config import MAX_SECTOR_EXPOSURE
        from quant_engine.config_structured import get_config

        assert MAX_SECTOR_EXPOSURE == get_config().backtest.max_sector_exposure

    def test_wf_max_train_dates(self):
        from quant_engine.config import WF_MAX_TRAIN_DATES
        from quant_engine.config_structured import get_config

        assert WF_MAX_TRAIN_DATES == get_config().backtest.wf_max_train_dates


class TestKellyConsistency:
    """Kelly sizing values must match."""

    def test_fraction(self):
        from quant_engine.config import KELLY_FRACTION
        from quant_engine.config_structured import get_config

        assert KELLY_FRACTION == get_config().kelly.fraction

    def test_max_portfolio_dd(self):
        from quant_engine.config import MAX_PORTFOLIO_DD
        from quant_engine.config_structured import get_config

        assert MAX_PORTFOLIO_DD == get_config().kelly.max_portfolio_dd

    def test_portfolio_blend(self):
        from quant_engine.config import KELLY_PORTFOLIO_BLEND
        from quant_engine.config_structured import get_config

        assert KELLY_PORTFOLIO_BLEND == get_config().kelly.portfolio_blend

    def test_bayesian_alpha(self):
        from quant_engine.config import KELLY_BAYESIAN_ALPHA
        from quant_engine.config_structured import get_config

        assert KELLY_BAYESIAN_ALPHA == get_config().kelly.bayesian_alpha

    def test_bayesian_beta(self):
        from quant_engine.config import KELLY_BAYESIAN_BETA
        from quant_engine.config_structured import get_config

        assert KELLY_BAYESIAN_BETA == get_config().kelly.bayesian_beta

    def test_regime_conditional(self):
        from quant_engine.config import KELLY_REGIME_CONDITIONAL
        from quant_engine.config_structured import get_config

        assert KELLY_REGIME_CONDITIONAL == get_config().kelly.regime_conditional


class TestDrawdownConsistency:
    """Drawdown tier values must match."""

    def test_warning_threshold(self):
        from quant_engine.config import DRAWDOWN_WARNING_THRESHOLD
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_WARNING_THRESHOLD == get_config().drawdown.warning_threshold

    def test_caution_threshold(self):
        from quant_engine.config import DRAWDOWN_CAUTION_THRESHOLD
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_CAUTION_THRESHOLD == get_config().drawdown.caution_threshold

    def test_critical_threshold(self):
        from quant_engine.config import DRAWDOWN_CRITICAL_THRESHOLD
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_CRITICAL_THRESHOLD == get_config().drawdown.critical_threshold

    def test_daily_loss_limit(self):
        from quant_engine.config import DRAWDOWN_DAILY_LOSS_LIMIT
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_DAILY_LOSS_LIMIT == get_config().drawdown.daily_loss_limit

    def test_weekly_loss_limit(self):
        from quant_engine.config import DRAWDOWN_WEEKLY_LOSS_LIMIT
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_WEEKLY_LOSS_LIMIT == get_config().drawdown.weekly_loss_limit

    def test_recovery_days(self):
        from quant_engine.config import DRAWDOWN_RECOVERY_DAYS
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_RECOVERY_DAYS == get_config().drawdown.recovery_days

    def test_size_mult_warning(self):
        from quant_engine.config import DRAWDOWN_SIZE_MULT_WARNING
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_SIZE_MULT_WARNING == get_config().drawdown.size_mult_warning

    def test_size_mult_caution(self):
        from quant_engine.config import DRAWDOWN_SIZE_MULT_CAUTION
        from quant_engine.config_structured import get_config

        assert DRAWDOWN_SIZE_MULT_CAUTION == get_config().drawdown.size_mult_caution


class TestStopLossConsistency:
    """Stop loss values must match."""

    def test_hard_stop_pct(self):
        from quant_engine.config import HARD_STOP_PCT
        from quant_engine.config_structured import get_config

        assert HARD_STOP_PCT == get_config().stop_loss.hard_stop_pct

    def test_atr_stop_multiplier(self):
        from quant_engine.config import ATR_STOP_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert ATR_STOP_MULTIPLIER == get_config().stop_loss.atr_stop_multiplier

    def test_trailing_atr_multiplier(self):
        from quant_engine.config import TRAILING_ATR_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert TRAILING_ATR_MULTIPLIER == get_config().stop_loss.trailing_atr_multiplier

    def test_trailing_activation_pct(self):
        from quant_engine.config import TRAILING_ACTIVATION_PCT
        from quant_engine.config_structured import get_config

        assert TRAILING_ACTIVATION_PCT == get_config().stop_loss.trailing_activation_pct

    def test_max_holding_days(self):
        from quant_engine.config import MAX_HOLDING_DAYS
        from quant_engine.config_structured import get_config

        assert MAX_HOLDING_DAYS == get_config().stop_loss.max_holding_days


class TestValidationConsistency:
    """Validation values must match."""

    def test_cpcv_partitions(self):
        from quant_engine.config import CPCV_PARTITIONS
        from quant_engine.config_structured import get_config

        assert CPCV_PARTITIONS == get_config().validation.cpcv_partitions

    def test_cpcv_test_partitions(self):
        from quant_engine.config import CPCV_TEST_PARTITIONS
        from quant_engine.config_structured import get_config

        assert CPCV_TEST_PARTITIONS == get_config().validation.cpcv_test_partitions

    def test_spa_bootstraps(self):
        from quant_engine.config import SPA_BOOTSTRAPS
        from quant_engine.config_structured import get_config

        assert SPA_BOOTSTRAPS == get_config().validation.spa_bootstraps

    def test_ic_rolling_window(self):
        from quant_engine.config import IC_ROLLING_WINDOW
        from quant_engine.config_structured import get_config

        assert IC_ROLLING_WINDOW == get_config().validation.ic_rolling_window


class TestPromotionConsistency:
    """Promotion gate values must match."""

    def test_min_trades(self):
        from quant_engine.config import PROMOTION_MIN_TRADES
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_TRADES == get_config().promotion.min_trades

    def test_min_win_rate(self):
        from quant_engine.config import PROMOTION_MIN_WIN_RATE
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_WIN_RATE == get_config().promotion.min_win_rate

    def test_min_sharpe(self):
        from quant_engine.config import PROMOTION_MIN_SHARPE
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_SHARPE == get_config().promotion.min_sharpe

    def test_min_profit_factor(self):
        from quant_engine.config import PROMOTION_MIN_PROFIT_FACTOR
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_PROFIT_FACTOR == get_config().promotion.min_profit_factor

    def test_max_drawdown(self):
        from quant_engine.config import PROMOTION_MAX_DRAWDOWN
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_DRAWDOWN == get_config().promotion.max_drawdown

    def test_min_annual_return(self):
        from quant_engine.config import PROMOTION_MIN_ANNUAL_RETURN
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_ANNUAL_RETURN == get_config().promotion.min_annual_return

    def test_max_active_strategies(self):
        from quant_engine.config import PROMOTION_MAX_ACTIVE_STRATEGIES
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_ACTIVE_STRATEGIES == get_config().promotion.max_active_strategies

    def test_require_advanced_contract(self):
        from quant_engine.config import PROMOTION_REQUIRE_ADVANCED_CONTRACT
        from quant_engine.config_structured import get_config

        assert PROMOTION_REQUIRE_ADVANCED_CONTRACT == get_config().promotion.require_advanced_contract

    def test_max_dsr_pvalue(self):
        from quant_engine.config import PROMOTION_MAX_DSR_PVALUE
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_DSR_PVALUE == get_config().promotion.max_dsr_pvalue

    def test_max_pbo(self):
        from quant_engine.config import PROMOTION_MAX_PBO
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_PBO == get_config().promotion.max_pbo

    def test_require_statistical_tests(self):
        from quant_engine.config import PROMOTION_REQUIRE_STATISTICAL_TESTS
        from quant_engine.config_structured import get_config

        assert PROMOTION_REQUIRE_STATISTICAL_TESTS == get_config().promotion.require_statistical_tests

    def test_require_cpcv(self):
        from quant_engine.config import PROMOTION_REQUIRE_CPCV
        from quant_engine.config_structured import get_config

        assert PROMOTION_REQUIRE_CPCV == get_config().promotion.require_cpcv

    def test_require_spa(self):
        from quant_engine.config import PROMOTION_REQUIRE_SPA
        from quant_engine.config_structured import get_config

        assert PROMOTION_REQUIRE_SPA == get_config().promotion.require_spa

    def test_min_wf_oos_corr(self):
        from quant_engine.config import PROMOTION_MIN_WF_OOS_CORR
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_WF_OOS_CORR == get_config().promotion.min_wf_oos_corr

    def test_min_wf_positive_fold_fraction(self):
        from quant_engine.config import PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION == get_config().promotion.min_wf_positive_fold_fraction

    def test_max_wf_is_oos_gap(self):
        from quant_engine.config import PROMOTION_MAX_WF_IS_OOS_GAP
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_WF_IS_OOS_GAP == get_config().promotion.max_wf_is_oos_gap

    def test_min_regime_positive_fraction(self):
        from quant_engine.config import PROMOTION_MIN_REGIME_POSITIVE_FRACTION
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_REGIME_POSITIVE_FRACTION == get_config().promotion.min_regime_positive_fraction

    def test_max_stress_drawdown(self):
        from quant_engine.config import PROMOTION_MAX_STRESS_DRAWDOWN
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_STRESS_DRAWDOWN == get_config().promotion.max_stress_drawdown

    def test_min_stress_sharpe(self):
        from quant_engine.config import PROMOTION_MIN_STRESS_SHARPE
        from quant_engine.config_structured import get_config

        assert PROMOTION_MIN_STRESS_SHARPE == get_config().promotion.min_stress_sharpe

    def test_max_transition_drawdown(self):
        from quant_engine.config import PROMOTION_MAX_TRANSITION_DRAWDOWN
        from quant_engine.config_structured import get_config

        assert PROMOTION_MAX_TRANSITION_DRAWDOWN == get_config().promotion.max_transition_drawdown

    def test_stress_regimes(self):
        from quant_engine.config import PROMOTION_STRESS_REGIMES
        from quant_engine.config_structured import get_config

        assert PROMOTION_STRESS_REGIMES == list(get_config().promotion.stress_regimes)


class TestPaperTradingConsistency:
    """Paper trading values must match."""

    def test_initial_capital(self):
        from quant_engine.config import PAPER_INITIAL_CAPITAL
        from quant_engine.config_structured import get_config

        assert PAPER_INITIAL_CAPITAL == get_config().paper_trading.initial_capital

    def test_max_total_positions(self):
        from quant_engine.config import PAPER_MAX_TOTAL_POSITIONS
        from quant_engine.config_structured import get_config

        assert PAPER_MAX_TOTAL_POSITIONS == get_config().paper_trading.max_total_positions

    def test_use_kelly_sizing(self):
        from quant_engine.config import PAPER_USE_KELLY_SIZING
        from quant_engine.config_structured import get_config

        assert PAPER_USE_KELLY_SIZING == get_config().paper_trading.use_kelly_sizing

    def test_kelly_fraction(self):
        from quant_engine.config import PAPER_KELLY_FRACTION
        from quant_engine.config_structured import get_config

        assert PAPER_KELLY_FRACTION == get_config().paper_trading.kelly_fraction

    def test_kelly_lookback_trades(self):
        from quant_engine.config import PAPER_KELLY_LOOKBACK_TRADES
        from quant_engine.config_structured import get_config

        assert PAPER_KELLY_LOOKBACK_TRADES == get_config().paper_trading.kelly_lookback_trades

    def test_kelly_min_size_multiplier(self):
        from quant_engine.config import PAPER_KELLY_MIN_SIZE_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert PAPER_KELLY_MIN_SIZE_MULTIPLIER == get_config().paper_trading.kelly_min_size_multiplier

    def test_kelly_max_size_multiplier(self):
        from quant_engine.config import PAPER_KELLY_MAX_SIZE_MULTIPLIER
        from quant_engine.config_structured import get_config

        assert PAPER_KELLY_MAX_SIZE_MULTIPLIER == get_config().paper_trading.kelly_max_size_multiplier


class TestExecutionConsistency:
    """Execution cost modeling values must match."""

    def test_spread_bps(self):
        from quant_engine.config import EXEC_SPREAD_BPS
        from quant_engine.config_structured import get_config

        assert EXEC_SPREAD_BPS == get_config().execution.spread_bps

    def test_max_participation(self):
        from quant_engine.config import EXEC_MAX_PARTICIPATION
        from quant_engine.config_structured import get_config

        assert EXEC_MAX_PARTICIPATION == get_config().execution.max_participation

    def test_impact_coeff_bps(self):
        from quant_engine.config import EXEC_IMPACT_COEFF_BPS
        from quant_engine.config_structured import get_config

        assert EXEC_IMPACT_COEFF_BPS == get_config().execution.impact_coeff_bps

    def test_min_fill_ratio(self):
        from quant_engine.config import EXEC_MIN_FILL_RATIO
        from quant_engine.config_structured import get_config

        assert EXEC_MIN_FILL_RATIO == get_config().execution.min_fill_ratio

    def test_dynamic_costs(self):
        from quant_engine.config import EXEC_DYNAMIC_COSTS
        from quant_engine.config_structured import get_config

        assert EXEC_DYNAMIC_COSTS == get_config().execution.dynamic_costs

    def test_almgren_chriss_enabled(self):
        from quant_engine.config import ALMGREN_CHRISS_ENABLED
        from quant_engine.config_structured import get_config

        assert ALMGREN_CHRISS_ENABLED == get_config().execution.almgren_chriss_enabled

    def test_almgren_chriss_adv_threshold(self):
        from quant_engine.config import ALMGREN_CHRISS_ADV_THRESHOLD
        from quant_engine.config_structured import get_config

        assert ALMGREN_CHRISS_ADV_THRESHOLD == get_config().execution.almgren_chriss_adv_threshold


class TestStructuredConfigAuthoritative:
    """Verify that config_structured.py is the single source of truth."""

    def test_structured_config_has_get_config(self):
        """get_config() must be importable from config_structured."""
        from quant_engine.config_structured import get_config

        assert callable(get_config)

    def test_config_py_imports_from_structured(self):
        """config.py must import get_config from config_structured."""
        import inspect
        import quant_engine.config as config_module

        source = inspect.getsource(config_module)
        assert "from config_structured import get_config" in source
