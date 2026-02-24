# `tests` Package Guide

## Purpose

Project-level test suite package.

## Package Summary

- Modules: 37
- Classes: 67
- Top-level functions: 94
- LOC: 6,186

## How This Package Fits Into The System

- Executable behavioral specification suite spanning engine modules plus API endpoint tests in `tests/api/`.
- Guards leakage, identity correctness, execution realism, promotion contracts, Kalshi hardening, and API envelopes.
- Use this package as the first source of truth for intended behavior when changing runtime logic.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `tests/__init__.py` | 6 | 0 | 0 | Project-level test suite package. |
| `tests/api/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `tests/api/test_compute_routers.py` | 65 | 0 | 6 | Tests for POST compute endpoints — verify job creation. |
| `tests/api/test_envelope.py` | 51 | 0 | 6 | Tests for ApiResponse envelope and ResponseMeta. |
| `tests/api/test_integration.py` | 72 | 0 | 4 | Integration tests — full app startup, envelope consistency, config patch. |
| `tests/api/test_jobs.py` | 135 | 0 | 12 | Tests for the job system: store, runner, lifecycle. |
| `tests/api/test_main.py` | 48 | 0 | 5 | Tests for app factory and basic middleware. |
| `tests/api/test_read_routers.py` | 118 | 0 | 14 | Tests for all GET endpoints — verify ApiResponse envelope. |
| `tests/api/test_services.py` | 80 | 0 | 10 | Tests for service wrappers — verify dict outputs. |
| `tests/conftest.py` | 240 | 1 | 9 | Shared test fixtures for the quant_engine test suite. |
| `tests/test_autopilot_predictor_fallback.py` | 58 | 1 | 0 | Test module for autopilot predictor fallback behavior and regressions. |
| `tests/test_backtest_realism.py` | 561 | 7 | 5 | Spec 011 — Backtest Execution Realism & Validation Enforcement tests. |
| `tests/test_cache_metadata_rehydrate.py` | 97 | 1 | 1 | Test module for cache metadata rehydrate behavior and regressions. |
| `tests/test_conceptual_fixes.py` | 324 | 4 | 2 | Tests for Spec 008 — Ensemble & Conceptual Fixes. |
| `tests/test_covariance_estimator.py` | 25 | 1 | 0 | Test module for covariance estimator behavior and regressions. |
| `tests/test_data_diagnostics.py` | 183 | 4 | 0 | Tests for data loading diagnostics (Spec 005). |
| `tests/test_delisting_total_return.py` | 76 | 1 | 0 | Test module for delisting total return behavior and regressions. |
| `tests/test_drawdown_liquidation.py` | 138 | 6 | 0 | Test module for drawdown liquidation behavior and regressions. |
| `tests/test_execution_dynamic_costs.py` | 48 | 1 | 0 | Test module for execution dynamic costs behavior and regressions. |
| `tests/test_feature_fixes.py` | 377 | 7 | 2 | Tests verifying Spec 012 feature engineering fixes: |
| `tests/test_integration.py` | 556 | 4 | 1 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 38 | 1 | 0 | Test module for iv arbitrage builder behavior and regressions. |
| `tests/test_kalshi_asof_features.py` | 65 | 1 | 0 | Test module for kalshi asof features behavior and regressions. |
| `tests/test_kalshi_distribution.py` | 114 | 1 | 0 | Test module for kalshi distribution behavior and regressions. |
| `tests/test_kalshi_hardening.py` | 605 | 1 | 0 | Test module for kalshi hardening behavior and regressions. |
| `tests/test_loader_and_predictor.py` | 230 | 3 | 0 | Test module for loader and predictor behavior and regressions. |
| `tests/test_lookahead_detection.py` | 144 | 1 | 1 | Automated lookahead bias detection for the feature pipeline. |
| `tests/test_panel_split.py` | 55 | 1 | 0 | Test module for panel split behavior and regressions. |
| `tests/test_paper_trader_kelly.py` | 128 | 1 | 3 | Test module for paper trader kelly behavior and regressions. |
| `tests/test_position_sizing_overhaul.py` | 414 | 8 | 0 | Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul. |
| `tests/test_promotion_contract.py` | 105 | 1 | 2 | Test module for promotion contract behavior and regressions. |
| `tests/test_provider_registry.py` | 29 | 1 | 0 | Test module for provider registry behavior and regressions. |
| `tests/test_research_factors.py` | 133 | 1 | 1 | Test module for research factors behavior and regressions. |
| `tests/test_survivorship_pit.py` | 74 | 1 | 0 | Test module for survivorship pit behavior and regressions. |
| `tests/test_training_pipeline_fixes.py` | 567 | 5 | 3 | Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance. |
| `tests/test_validation_and_risk_extensions.py` | 107 | 1 | 1 | Test module for validation and risk extensions behavior and regressions. |
| `tests/test_zero_errors.py` | 120 | 1 | 6 | Integration test: common operations must produce zero ERROR-level log entries. |

## Module Details

### `tests/__init__.py`
- Intent: Project-level test suite package.
- Classes: none
- Top-level functions: none

### `tests/api/__init__.py`
- Intent: No module docstring.
- Classes: none
- Top-level functions: none

### `tests/api/test_compute_routers.py`
- Intent: Tests for POST compute endpoints — verify job creation.
- Classes: none
- Top-level functions: `test_train_creates_job`, `test_predict_creates_job`, `test_backtest_creates_job`, `test_autopilot_creates_job`, `test_job_status_queryable`, `test_nonexistent_job_404`

### `tests/api/test_envelope.py`
- Intent: Tests for ApiResponse envelope and ResponseMeta.
- Classes: none
- Top-level functions: `test_success_response`, `test_fail_response`, `test_from_cached`, `test_meta_defaults`, `test_meta_custom_fields`, `test_serialization_roundtrip`

### `tests/api/test_integration.py`
- Intent: Integration tests — full app startup, envelope consistency, config patch.
- Classes: none
- Top-level functions: `test_all_gets_return_envelope`, `test_config_patch_and_read`, `test_config_patch_invalid_key`, `test_meta_has_generated_at`

### `tests/api/test_jobs.py`
- Intent: Tests for the job system: store, runner, lifecycle.
- Classes: none
- Top-level functions: `store`, `runner`, `test_create_and_get_job`, `test_list_jobs`, `test_update_status`, `test_update_progress`, `test_cancel_queued_job`, `test_cancel_completed_job_fails`, `test_get_nonexistent_job`, `test_job_runner_succeeds`, `test_job_runner_failure`, `test_sse_events`

### `tests/api/test_main.py`
- Intent: Tests for app factory and basic middleware.
- Classes: none
- Top-level functions: `test_create_app`, `test_openapi_schema`, `test_routes_registered`, `test_404_wrapped`, `test_cors_headers`

### `tests/api/test_read_routers.py`
- Intent: Tests for all GET endpoints — verify ApiResponse envelope.
- Classes: none
- Top-level functions: `test_health`, `test_dashboard_summary`, `test_data_universe`, `test_models_versions`, `test_signals_latest`, `test_backtests_latest`, `test_backtests_trades`, `test_backtests_equity_curve`, `test_autopilot_latest_cycle`, `test_autopilot_strategies`, `test_autopilot_paper_state`, `test_config_get`, `test_logs`, `test_jobs_list_empty`

### `tests/api/test_services.py`
- Intent: Tests for service wrappers — verify dict outputs.
- Classes: none
- Top-level functions: `test_data_service_universe_info`, `test_data_service_cached_tickers`, `test_backtest_service_no_results`, `test_autopilot_service_latest_cycle`, `test_autopilot_service_strategy_registry`, `test_autopilot_service_paper_state`, `test_health_service_quick`, `test_model_service_list_versions`, `test_model_service_champion_info`, `test_results_service_list`

### `tests/conftest.py`
- Intent: Shared test fixtures for the quant_engine test suite.
- Classes:
  - `_InMemoryJobStore`: Lightweight in-memory job store for tests (avoids aiosqlite threads).
    - Methods: `initialize`, `close`, `create_job`, `get_job`, `list_jobs`, `update_status`, `update_progress`, `cancel_job`
- Top-level functions: `pytest_sessionfinish`, `synthetic_ohlcv_data`, `synthetic_trades_csv`, `synthetic_model_meta`, `tmp_results_dir`, `tmp_model_dir`, `tmp_data_cache_dir`, `app`, `client`

### `tests/test_autopilot_predictor_fallback.py`
- Intent: Test module for autopilot predictor fallback behavior and regressions.
- Classes:
  - `AutopilotPredictorFallbackTests`: Test cases covering autopilot predictor fallback behavior and system invariants.
    - Methods: `test_ensure_predictor_falls_back_when_model_import_fails`
- Top-level functions: none

### `tests/test_backtest_realism.py`
- Intent: Spec 011 — Backtest Execution Realism & Validation Enforcement tests.
- Classes:
  - `TestEntryTimingConsistency`: T1: Both simple and risk-managed modes use next-bar Open for entry.
    - Methods: `test_simple_mode_entry_at_next_bar_open`, `test_risk_managed_mode_entry_at_next_bar_open`
  - `TestAlmgrenChrissCalibration`: T2: AC risk_aversion calibrated to institutional levels.
    - Methods: `test_config_risk_aversion_not_risk_neutral`, `test_trajectory_default_matches_config`, `test_higher_risk_aversion_frontloads_execution`
  - `TestExitVolumeConstraints`: T3: Exit simulation respects volume constraints.
    - Methods: `test_execution_model_limits_fill_ratio`, `test_force_full_bypasses_volume_constraint`, `test_moderate_order_gets_partial_fill`
  - `TestNegativeSharpeFailsDSR`: T5: Negative Sharpe strategies always fail DSR.
    - Methods: `test_negative_sharpe_rejected`, `test_zero_sharpe_rejected`, `test_positive_sharpe_can_pass`, `test_weak_positive_sharpe_with_many_trials_fails`
  - `TestPBOThreshold`: T6: PBO threshold tightened to 0.45.
    - Methods: `test_config_pbo_threshold`, `test_pbo_above_045_is_overfit`, `test_pbo_max_combinations_increased`
  - `TestValidationRequiredForPromotion`: T4: Validation required for autopilot promotion.
    - Methods: `test_no_validation_rejects_promotion`, `test_negative_sharpe_always_rejected`, `test_mc_not_significant_rejects_promotion`, `test_all_validations_pass_allows_promotion`, `test_pbo_above_045_rejects_promotion`
  - `TestPartialExitMultiBar`: T7: Residual shares tracked across bars for multi-bar exits.
    - Methods: `test_backtester_has_residual_tracking`, `test_volume_constrained_exit_records_fill_ratio`, `test_residual_position_exit_concept`
- Top-level functions: `_make_ohlcv`, `_make_predictions`, `_candidate`, `_passing_result`, `_all_pass_metrics`

### `tests/test_cache_metadata_rehydrate.py`
- Intent: Test module for cache metadata rehydrate behavior and regressions.
- Classes:
  - `CacheMetadataRehydrateTests`: Test cases covering cache metadata rehydrate behavior and system invariants.
    - Methods: `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`
- Top-level functions: `_write_daily_csv`

### `tests/test_conceptual_fixes.py`
- Intent: Tests for Spec 008 — Ensemble & Conceptual Fixes.
- Classes:
  - `TestEnsembleNoPhantomVote`: When REGIME_JUMP_MODEL_ENABLED=False, ensemble must use 2 methods, not 3.
    - Methods: `test_two_method_ensemble_does_not_call_jump`, `test_two_method_ensemble_requires_unanimity`
  - `TestEnsembleWithJumpModel`: When REGIME_JUMP_MODEL_ENABLED=True, ensemble uses 3 independent methods.
    - Methods: `test_three_method_ensemble_calls_all_methods`, `test_three_method_probabilities_sum_to_one`
  - `TestRuleDetectMinDuration`: Rule-based detection must enforce REGIME_MIN_DURATION.
    - Methods: `test_no_short_runs_with_min_duration`, `test_min_duration_1_is_noop`, `test_smoothing_preserves_regime_count`
  - `TestConfigEndpointDefaults`: The /api/config endpoint returns values that match config.py.
    - Methods: `test_adjustable_config_includes_backtest_keys`, `test_adjustable_config_values_match_config_py`, `test_config_status_includes_training_section`, `test_config_status_backtest_section_matches`
- Top-level functions: `_make_synthetic_features`, `_make_features_with_regime_flickers`

### `tests/test_covariance_estimator.py`
- Intent: Test module for covariance estimator behavior and regressions.
- Classes:
  - `CovarianceEstimatorTests`: Test cases covering covariance estimator behavior and system invariants.
    - Methods: `test_single_asset_covariance_is_2d_and_positive`
- Top-level functions: none

### `tests/test_data_diagnostics.py`
- Intent: Tests for data loading diagnostics (Spec 005).
- Classes:
  - `TestOrchestratorImport`: T1: DATA_DIR -> DATA_CACHE_DIR fix.
    - Methods: `test_orchestrator_imports_data_cache_dir`, `test_orchestrator_uses_data_cache_dir_in_diagnostics`
  - `TestLoadUniverseSkipReasons`: T2: Skip reasons logged at WARNING even with verbose=False.
    - Methods: `test_load_universe_logs_skip_reasons`, `test_skip_reasons_include_reason_text`
  - `TestErrorMessageDiagnostics`: T3: RuntimeError includes ticker list, WRDS, cache count.
    - Methods: `test_error_message_includes_diagnostics`
  - `TestDataStatusService`: T4: DataService.get_cache_status returns valid ticker info.
    - Methods: `test_data_status_returns_summary`, `test_data_status_ticker_entries`, `test_data_status_with_missing_cache_dir`, `test_data_status_freshness_categories`
- Top-level functions: none

### `tests/test_delisting_total_return.py`
- Intent: Test module for delisting total return behavior and regressions.
- Classes:
  - `DelistingTotalReturnTests`: Test cases covering delisting total return behavior and system invariants.
    - Methods: `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`
- Top-level functions: none

### `tests/test_drawdown_liquidation.py`
- Intent: Test module for drawdown liquidation behavior and regressions.
- Classes:
  - `_FakePositionSizer`: Test double used to isolate behavior in this test module.
    - Methods: `size_position`
  - `_FakeDrawdownController`: Test double used to isolate behavior in this test module.
    - Methods: `update`, `get_summary`
  - `_FakeStopLossManager`: Test double used to isolate behavior in this test module.
    - Methods: `evaluate`
  - `_FakePortfolioRisk`: Test double used to isolate behavior in this test module.
    - Methods: `check_new_position`
  - `_FakeRiskMetrics`: Test double used to isolate behavior in this test module.
    - Methods: `compute_full_report`
  - `DrawdownLiquidationTests`: Test cases covering drawdown liquidation behavior and system invariants.
    - Methods: `test_critical_drawdown_forces_liquidation`
- Top-level functions: none

### `tests/test_execution_dynamic_costs.py`
- Intent: Test module for execution dynamic costs behavior and regressions.
- Classes:
  - `ExecutionDynamicCostTests`: Test cases covering execution dynamic costs behavior and system invariants.
    - Methods: `test_dynamic_costs_increase_under_stress`
- Top-level functions: none

### `tests/test_feature_fixes.py`
- Intent: Tests verifying Spec 012 feature engineering fixes:
- Classes:
  - `TestTSMomBackwardShift`: Verify TSMom features are backward-looking (causal).
    - Methods: `test_tsmom_uses_backward_shift`, `test_tsmom_values_dont_change_when_future_removed`
  - `TestIVShockNonFuture`: Verify IV shock features use only backward-looking shifts.
    - Methods: `test_iv_shock_no_future_shift`, `test_iv_shock_no_lookahead`
  - `TestFeatureMetadata`: Verify feature metadata registry coverage.
    - Methods: `test_feature_metadata_exists`, `test_all_metadata_has_valid_type`, `test_pipeline_features_have_metadata`
  - `TestProductionModeFilter`: Verify production_mode filters RESEARCH_ONLY features.
    - Methods: `test_production_mode_filters_research`, `test_production_mode_default_false`
  - `TestRollingVWAPIsCausal`: Verify rolling VWAP uses only past data.
    - Methods: `test_rolling_vwap_is_causal`, `test_rolling_vwap_first_rows_nan`
  - `TestMultiHorizonVRP`: Verify enhanced VRP computation at multiple horizons.
    - Methods: `test_vrp_multi_horizon`
  - `TestFullPipelineLookahead`: End-to-end lookahead check on the complete feature pipeline.
    - Methods: `test_automated_lookahead_detection`
- Top-level functions: `_make_ohlcv`, `_make_ohlcv_with_iv`

### `tests/test_integration.py`
- Intent: End-to-end integration tests for the quant engine pipeline.
- Classes:
  - `TestFullPipelineSynthetic`: End-to-end test: data -> features -> regimes -> training -> prediction -> backtest.
    - Methods: `synthetic_data`, `pipeline_outputs`, `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`
  - `TestCvGapHardBlock`: Verify that the CV gap hard block rejects overfit models.
    - Methods: `test_cv_gap_hard_block`
  - `TestRegime2Suppression`: Verify regime 2 gating suppresses trades.
    - Methods: `test_regime_2_suppression`, `test_regime_0_not_suppressed`
  - `TestCrossSectionalRanking`: Verify cross-sectional ranker produces valid output.
    - Methods: `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`
- Top-level functions: `_generate_synthetic_ohlcv`

### `tests/test_iv_arbitrage_builder.py`
- Intent: Test module for iv arbitrage builder behavior and regressions.
- Classes:
  - `ArbitrageFreeSVIBuilderTests`: Test cases covering iv arbitrage builder behavior and system invariants.
    - Methods: `test_build_surface_has_valid_shape_and_monotone_total_variance`
- Top-level functions: none

### `tests/test_kalshi_asof_features.py`
- Intent: Test module for kalshi asof features behavior and regressions.
- Classes:
  - `KalshiAsofFeatureTests`: Test cases covering kalshi asof features behavior and regression protections.
    - Methods: `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`
- Top-level functions: none

### `tests/test_kalshi_distribution.py`
- Intent: Test module for kalshi distribution behavior and regressions.
- Classes:
  - `KalshiDistributionTests`: Test cases covering kalshi distribution behavior and regression protections.
    - Methods: `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`
- Top-level functions: none

### `tests/test_kalshi_hardening.py`
- Intent: Test module for kalshi hardening behavior and regressions.
- Classes:
  - `KalshiHardeningTests`: Test cases covering kalshi hardening behavior and regression protections.
    - Methods: `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof` (+4 more)
- Top-level functions: none

### `tests/test_loader_and_predictor.py`
- Intent: Test module for loader and predictor behavior and regressions.
- Classes:
  - `_FakeWRDSProvider`: Test double used to isolate behavior in this test module.
    - Methods: `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_option_surface_features`, `resolve_permno`
  - `_UnavailableWRDSProvider`: Test double representing an unavailable dependency or provider.
    - Methods: `available`
  - `LoaderAndPredictorTests`: Test cases covering loader and predictor behavior and system invariants.
    - Methods: `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`
- Top-level functions: none

### `tests/test_lookahead_detection.py`
- Intent: Automated lookahead bias detection for the feature pipeline.
- Classes:
  - `TestLookaheadDetection`: Detect any feature that uses future data.
    - Methods: `test_no_feature_uses_future_data`, `test_no_feature_uses_future_data_deep`
- Top-level functions: `_make_synthetic_ohlcv`

### `tests/test_panel_split.py`
- Intent: Test module for panel split behavior and regressions.
- Classes:
  - `PanelSplitTests`: Test cases covering panel split behavior and system invariants.
    - Methods: `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`
- Top-level functions: none

### `tests/test_paper_trader_kelly.py`
- Intent: Test module for paper trader kelly behavior and regressions.
- Classes:
  - `PaperTraderKellyTests`: Test cases covering paper trader kelly behavior and system invariants.
    - Methods: `test_kelly_sizing_changes_position_size_with_bounds`
- Top-level functions: `_mock_price_data`, `_seed_state`, `_run_cycle`

### `tests/test_position_sizing_overhaul.py`
- Intent: Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul.
- Classes:
  - `TestKellyNegativeEdge`: T1: Kelly returns 0.0 for negative-edge signals, not min_position.
    - Methods: `test_kelly_negative_edge_returns_zero`, `test_kelly_positive_edge_returns_positive`, `test_kelly_invalid_inputs_return_zero`, `test_kelly_small_sample_penalty`, `test_kelly_n_trades_passed_through_size_position`
  - `TestDrawdownGovernor`: T2: Exponential drawdown governor is more lenient early.
    - Methods: `test_convex_at_50pct_drawdown`, `test_aggressive_at_90pct_drawdown`, `test_no_drawdown_full_sizing`, `test_beyond_max_dd_returns_zero`, `test_positive_drawdown_returns_full`, `test_convex_curve_is_smooth`
  - `TestPerRegimeBayesian`: T3: Bayesian win-rate tracked per-regime with separate priors.
    - Methods: `test_bull_kelly_greater_than_hv`, `test_global_fallback_when_few_regime_trades`, `test_per_regime_counters_populated`
  - `TestRegimeStatsUpdate`: T4: Regime stats updated from trade DataFrame with regime column.
    - Methods: `test_update_from_trade_df`, `test_too_few_trades_keeps_defaults`, `test_multiple_regimes_updated`
  - `TestConfidenceScalar`: T5: Confidence scalar range is [0.5, 1.0], never 1.5.
    - Methods: `test_max_confidence_scalar_is_one`, `test_zero_confidence_scalar_is_half`, `test_confidence_never_amplifies`
  - `TestPortfolioCorrelationPenalty`: T6: High-correlation positions get reduced allocation.
    - Methods: `returns_data`, `test_high_corr_smaller_than_uncorr`, `test_no_positions_returns_base`, `test_negative_corr_no_penalty`
  - `TestDrawdownControllerIntegration`: T7: DrawdownController blocks entries and forces liquidation.
    - Methods: `test_normal_state_allows_entries`, `test_large_loss_triggers_caution`, `test_critical_drawdown_forces_liquidation`, `test_paper_trader_has_drawdown_controller`, `test_recovery_allows_gradual_reentry`
  - `TestSizePositionIntegration`: End-to-end tests for the updated size_position method.
    - Methods: `test_negative_edge_signal_still_gets_sized_via_blend`, `test_drawdown_reduces_kelly_in_blend`
- Top-level functions: none

### `tests/test_promotion_contract.py`
- Intent: Test module for promotion contract behavior and regressions.
- Classes:
  - `PromotionContractTests`: Test cases covering promotion contract behavior and system invariants.
    - Methods: `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`
- Top-level functions: `_candidate`, `_result`

### `tests/test_provider_registry.py`
- Intent: Test module for provider registry behavior and regressions.
- Classes:
  - `ProviderRegistryTests`: Test cases covering provider registry behavior and system invariants.
    - Methods: `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`
- Top-level functions: none

### `tests/test_research_factors.py`
- Intent: Test module for research factors behavior and regressions.
- Classes:
  - `ResearchFactorTests`: Test cases covering research factors behavior and system invariants.
    - Methods: `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`
- Top-level functions: `_make_ohlcv`

### `tests/test_survivorship_pit.py`
- Intent: Test module for survivorship pit behavior and regressions.
- Classes:
  - `SurvivorshipPointInTimeTests`: Test cases covering survivorship pit behavior and system invariants.
    - Methods: `test_filter_panel_by_point_in_time_universe`
- Top-level functions: none

### `tests/test_training_pipeline_fixes.py`
- Intent: Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance.
- Classes:
  - `TestPerFoldFeatureSelection`: Verify that feature selection runs independently per CV fold.
    - Methods: `test_feature_selection_per_fold`, `test_stable_features_selected_most_folds`, `test_compute_stable_features_fallback`
  - `TestCalibrationValidationSplit`: Verify calibration uses a separate split and computes ECE.
    - Methods: `test_calibration_uses_separate_split`, `test_ece_computed`, `test_ece_perfect_calibration`, `test_ece_worst_calibration`, `test_reliability_curve`
  - `TestRegimeMinSamples`: Verify regime models require minimum sample count.
    - Methods: `test_regime_model_min_samples`, `test_regime_model_skipped_for_low_samples`, `test_regime_fallback_to_global`
  - `TestCorrelationPruning`: Verify feature correlation pruning at 0.80 threshold.
    - Methods: `test_correlation_threshold_080`, `test_old_threshold_would_keep_correlated`, `test_default_threshold_is_080`
  - `TestGovernanceScoring`: Verify governance scoring includes validation metrics.
    - Methods: `test_governance_score_includes_validation`, `test_governance_score_backward_compatible`, `test_governance_promotion_with_validation`, `test_dsr_penalty_reduces_score`
- Top-level functions: `_make_feature_matrix`, `_make_targets`, `_make_regimes`

### `tests/test_validation_and_risk_extensions.py`
- Intent: Test module for validation and risk extensions behavior and regressions.
- Classes:
  - `ValidationAndRiskExtensionTests`: Test cases covering validation and risk extensions behavior and system invariants.
    - Methods: `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`
- Top-level functions: `_make_ohlcv`

### `tests/test_zero_errors.py`
- Intent: Integration test: common operations must produce zero ERROR-level log entries.
- Classes:
  - `_ErrorCapture`: Handler that records any ERROR or CRITICAL log records.
    - Methods: `emit`
- Top-level functions: `error_capture`, `test_config_validation_no_errors`, `test_regime_detection_no_errors`, `test_health_service_no_errors`, `test_data_loading_graceful_degradation`, `test_data_loading_single_ticker_if_cache_exists`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` (entrypoints and workflows)
