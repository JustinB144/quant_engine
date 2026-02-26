# Test Specification Map

Source-derived behavioral map of the current test suite (current working tree).

## Scope Summary

| Scope | Test Files | Test Classes | Test Methods |
|---|---|---|---|
| `tests/api` | 7 | 0 | 55 |
| `tests` | 59 | 285 | 1006 |
| `kalshi/tests` | 8 | 8 | 48 |

## `tests/api`

| File | LOC | Test Classes | Test Methods | Intent |
|---|---|---|---|---|
| `tests/api/test_compute_routers.py` | 66 | 0 | 6 | Tests for POST compute endpoints — verify job creation. |
| `tests/api/test_envelope.py` | 52 | 0 | 6 | Tests for ApiResponse envelope and ResponseMeta. |
| `tests/api/test_integration.py` | 73 | 0 | 4 | Integration tests — full app startup, envelope consistency, config patch. |
| `tests/api/test_jobs.py` | 136 | 0 | 10 | Tests for the job system: store, runner, lifecycle. |
| `tests/api/test_main.py` | 49 | 0 | 5 | Tests for app factory and basic middleware. |
| `tests/api/test_read_routers.py` | 119 | 0 | 14 | Tests for all GET endpoints — verify ApiResponse envelope. |
| `tests/api/test_services.py` | 81 | 0 | 10 | Tests for service wrappers — verify dict outputs. |

### `tests/api/test_compute_routers.py`
- Intent: Tests for POST compute endpoints — verify job creation.
- LOC: 66
- Top-level tests: `test_train_creates_job`, `test_predict_creates_job`, `test_backtest_creates_job`, `test_autopilot_creates_job`, `test_job_status_queryable`, `test_nonexistent_job_404`

### `tests/api/test_envelope.py`
- Intent: Tests for ApiResponse envelope and ResponseMeta.
- LOC: 52
- Top-level tests: `test_success_response`, `test_fail_response`, `test_from_cached`, `test_meta_defaults`, `test_meta_custom_fields`, `test_serialization_roundtrip`

### `tests/api/test_integration.py`
- Intent: Integration tests — full app startup, envelope consistency, config patch.
- LOC: 73
- Top-level tests: `test_all_gets_return_envelope`, `test_config_patch_and_read`, `test_config_patch_invalid_key`, `test_meta_has_generated_at`

### `tests/api/test_jobs.py`
- Intent: Tests for the job system: store, runner, lifecycle.
- LOC: 136
- Top-level tests: `test_create_and_get_job`, `test_list_jobs`, `test_update_status`, `test_update_progress`, `test_cancel_queued_job`, `test_cancel_completed_job_fails`, `test_get_nonexistent_job`, `test_job_runner_succeeds`, `test_job_runner_failure`, `test_sse_events`

### `tests/api/test_main.py`
- Intent: Tests for app factory and basic middleware.
- LOC: 49
- Top-level tests: `test_create_app`, `test_openapi_schema`, `test_routes_registered`, `test_404_wrapped`, `test_cors_headers`

### `tests/api/test_read_routers.py`
- Intent: Tests for all GET endpoints — verify ApiResponse envelope.
- LOC: 119
- Top-level tests: `test_health`, `test_dashboard_summary`, `test_data_universe`, `test_models_versions`, `test_signals_latest`, `test_backtests_latest`, `test_backtests_trades`, `test_backtests_equity_curve`, `test_autopilot_latest_cycle`, `test_autopilot_strategies`, `test_autopilot_paper_state`, `test_config_get`, `test_logs`, `test_jobs_list_empty`

### `tests/api/test_services.py`
- Intent: Tests for service wrappers — verify dict outputs.
- LOC: 81
- Top-level tests: `test_data_service_universe_info`, `test_data_service_cached_tickers`, `test_backtest_service_no_results`, `test_autopilot_service_latest_cycle`, `test_autopilot_service_strategy_registry`, `test_autopilot_service_paper_state`, `test_health_service_quick`, `test_model_service_list_versions`, `test_model_service_champion_info`, `test_results_service_list`

## `tests`

| File | LOC | Test Classes | Test Methods | Intent |
|---|---|---|---|---|
| `tests/test_ab_testing_framework.py` | 508 | 9 | 21 | Tests for the A/B testing framework (Spec 014). |
| `tests/test_autopilot_predictor_fallback.py` | 59 | 1 | 1 | Test module for autopilot predictor fallback behavior and regressions. |
| `tests/test_backtest_realism.py` | 562 | 7 | 23 | Spec 011 — Backtest Execution Realism & Validation Enforcement tests. |
| `tests/test_bocpd.py` | 345 | 8 | 24 | Unit tests for BOCPD (Bayesian Online Change-Point Detection). |
| `tests/test_cache_metadata_rehydrate.py` | 98 | 1 | 3 | Test module for cache metadata rehydrate behavior and regressions. |
| `tests/test_charting_endpoints.py` | 320 | 6 | 26 | Tests for charting API endpoints and data flow (Spec 004). |
| `tests/test_conceptual_fixes.py` | 325 | 4 | 11 | Tests for Spec 008 — Ensemble & Conceptual Fixes. |
| `tests/test_covariance_estimator.py` | 26 | 1 | 1 | Test module for covariance estimator behavior and regressions. |
| `tests/test_data_diagnostics.py` | 184 | 4 | 9 | Tests for data loading diagnostics (Spec 005). |
| `tests/test_delisting_total_return.py` | 77 | 1 | 2 | Test module for delisting total return behavior and regressions. |
| `tests/test_drawdown_liquidation.py` | 139 | 6 | 1 | Test module for drawdown liquidation behavior and regressions. |
| `tests/test_ensemble_voting.py` | 221 | 2 | 10 | Tests for confidence-weighted ensemble voting (SPEC_10 T2). |
| `tests/test_evaluation_engine.py` | 326 | 4 | 22 | Tests for evaluation/engine.py — EvaluationEngine integration tests. |
| `tests/test_evaluation_fragility.py` | 159 | 5 | 16 | Tests for evaluation/fragility.py — PnL concentration, drawdown, recovery, critical slowing. |
| `tests/test_evaluation_metrics.py` | 125 | 2 | 12 | Tests for evaluation/metrics.py — slice metrics and decile spread. |
| `tests/test_evaluation_ml_diagnostics.py` | 110 | 2 | 9 | Tests for evaluation/ml_diagnostics.py — feature drift and ensemble disagreement. |
| `tests/test_evaluation_slicing.py` | 131 | 2 | 9 | Tests for evaluation/slicing.py — PerformanceSlice and SliceRegistry. |
| `tests/test_execution_dynamic_costs.py` | 49 | 1 | 1 | Test module for execution dynamic costs behavior and regressions. |
| `tests/test_execution_layer.py` | 849 | 17 | 79 | Comprehensive tests for Spec 06: Execution Layer Improvements. |
| `tests/test_feature_fixes.py` | 378 | 7 | 13 | Tests verifying Spec 012 feature engineering fixes: |
| `tests/test_health_checks_rewritten.py` | 417 | 8 | 21 | Tests for the rewritten health check system (Spec 010). |
| `tests/test_health_spec09.py` | 721 | 7 | 51 | Tests for Health System Spec 09 enhancements. |
| `tests/test_health_transparency.py` | 283 | 5 | 14 | Tests for health system transparency and trustworthiness (Spec 002). |
| `tests/test_integration.py` | 557 | 4 | 13 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 39 | 1 | 1 | Test module for iv arbitrage builder behavior and regressions. |
| `tests/test_jump_model_pypi.py` | 379 | 10 | 20 | Unit tests for the PyPI JumpModel wrapper (Spec 001). |
| `tests/test_jump_model_validation.py` | 397 | 6 | 10 | Jump Model Audit and Validation (SPEC_10 T1). |
| `tests/test_kalshi_asof_features.py` | 66 | 1 | 2 | Test module for kalshi asof features behavior and regressions. |
| `tests/test_kalshi_distribution.py` | 115 | 1 | 3 | Test module for kalshi distribution behavior and regressions. |
| `tests/test_kalshi_hardening.py` | 606 | 1 | 16 | Test module for kalshi hardening behavior and regressions. |
| `tests/test_loader_and_predictor.py` | 231 | 3 | 8 | Test module for loader and predictor behavior and regressions. |
| `tests/test_lookahead_detection.py` | 145 | 1 | 2 | Automated lookahead bias detection for the feature pipeline. |
| `tests/test_observation_matrix_expansion.py` | 186 | 3 | 8 | Tests for expanded HMM observation matrix (SPEC_10 T4). |
| `tests/test_online_update.py` | 237 | 6 | 10 | Tests for online regime updating via forward algorithm (SPEC_10 T5). |
| `tests/test_panel_split.py` | 56 | 1 | 2 | Test module for panel split behavior and regressions. |
| `tests/test_paper_trader_integration.py` | 844 | 12 | 13 | Test module for paper trader integration — spec 015. |
| `tests/test_paper_trader_kelly.py` | 129 | 1 | 1 | Test module for paper trader kelly behavior and regressions. |
| `tests/test_portfolio_layer.py` | 1027 | 10 | 54 | Comprehensive test suite for Spec 07: Portfolio Layer + Regime-Conditioned Constraints. |
| `tests/test_position_sizing_overhaul.py` | 425 | 8 | 30 | Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul. |
| `tests/test_promotion_contract.py` | 108 | 1 | 2 | Test module for promotion contract behavior and regressions. |
| `tests/test_provider_registry.py` | 30 | 1 | 3 | Test module for provider registry behavior and regressions. |
| `tests/test_regime_consensus.py` | 205 | 4 | 12 | Tests for cross-sectional regime consensus (SPEC_10 T6). |
| `tests/test_regime_detection_integration.py` | 610 | 8 | 28 | End-to-end integration tests for the upgraded regime detection pipeline (SPEC_10 T7). |
| `tests/test_regime_integration.py` | 312 | 6 | 16 | Integration tests for the full regime detection pipeline with Jump Model (Spec 001). |
| `tests/test_regime_payload.py` | 242 | 4 | 8 | Tests for regime data pipeline end-to-end (Spec 003). |
| `tests/test_regime_uncertainty.py` | 163 | 2 | 13 | Tests for regime uncertainty integration (SPEC_10 T3). |
| `tests/test_research_factors.py` | 134 | 1 | 4 | Test module for research factors behavior and regressions. |
| `tests/test_risk_governor.py` | 713 | 11 | 50 | Comprehensive tests for Spec 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing. |
| `tests/test_risk_improvements.py` | 493 | 9 | 24 | Comprehensive tests for Spec 016: Risk System Improvements. |
| `tests/test_shock_vector.py` | 373 | 6 | 29 | Unit tests for ShockVector and ShockVectorValidator. |
| `tests/test_signal_meta_labeling.py` | 757 | 7 | 42 | Tests for Spec 04: Signal Enhancement + Meta-Labeling + Fold-Level Validation. |
| `tests/test_structural_features.py` | 821 | 8 | 47 | Tests for Spec 02: Structural Feature Expansion. |
| `tests/test_structural_state_integration.py` | 385 | 7 | 23 | Integration tests for the Structural State Layer (SPEC_03). |
| `tests/test_survivorship_pit.py` | 75 | 1 | 1 | Test module for survivorship pit behavior and regressions. |
| `tests/test_system_innovations.py` | 773 | 16 | 46 | Comprehensive tests for Spec 017: System-Level Innovation. |
| `tests/test_training_pipeline_fixes.py` | 568 | 5 | 18 | Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance. |
| `tests/test_truth_layer.py` | 872 | 7 | 60 | Tests for Spec 01: Foundational Hardening — Truth Layer. |
| `tests/test_validation_and_risk_extensions.py` | 108 | 1 | 3 | Test module for validation and risk extensions behavior and regressions. |
| `tests/test_zero_errors.py` | 121 | 1 | 5 | Integration test: common operations must produce zero ERROR-level log entries. |

### `tests/test_ab_testing_framework.py`
- Intent: Tests for the A/B testing framework (Spec 014).
- LOC: 508
- Test classes:
  - `TestBlockBootstrap`: 2 test(s) -> `test_block_bootstrap_wider_ci`, `test_newey_west_returns_valid`
  - `TestTickerAssignment`: 3 test(s) -> `test_ticker_assignment_deterministic`, `test_ticker_assignment_balanced`, `test_no_ticker_contamination`
  - `TestPowerAnalysis`: 3 test(s) -> `test_power_analysis_reasonable`, `test_power_analysis_larger_effect_fewer_samples`, `test_power_analysis_higher_power_more_samples`
  - `TestEarlyStopping`: 3 test(s) -> `test_early_stopping_insufficient_data`, `test_early_stopping_conservative_early`, `test_early_stopping_returns_valid_dict`
  - `TestTradePersistence`: 1 test(s) -> `test_trade_persistence_parquet`
  - `TestComprehensiveReport`: 2 test(s) -> `test_comprehensive_report_metrics`, `test_report_with_regime_breakdown`
  - `TestMaxConcurrentTests`: 2 test(s) -> `test_max_concurrent_tests`, `test_completed_tests_dont_count`
  - `TestRegistryOperations`: 3 test(s) -> `test_cancel_test`, `test_get_active_test`, `test_list_tests_by_status`
  - `TestVariantMetrics`: 2 test(s) -> `test_empty_variant_metrics`, `test_winning_variant_positive_sharpe`

### `tests/test_autopilot_predictor_fallback.py`
- Intent: Test module for autopilot predictor fallback behavior and regressions.
- LOC: 59
- Test classes:
  - `AutopilotPredictorFallbackTests`: 1 test(s) -> `test_ensure_predictor_falls_back_when_model_import_fails`

### `tests/test_backtest_realism.py`
- Intent: Spec 011 — Backtest Execution Realism & Validation Enforcement tests.
- LOC: 562
- Test classes:
  - `TestEntryTimingConsistency`: 2 test(s) -> `test_simple_mode_entry_at_next_bar_open`, `test_risk_managed_mode_entry_at_next_bar_open`
  - `TestAlmgrenChrissCalibration`: 3 test(s) -> `test_config_risk_aversion_not_risk_neutral`, `test_trajectory_default_matches_config`, `test_higher_risk_aversion_frontloads_execution`
  - `TestExitVolumeConstraints`: 3 test(s) -> `test_execution_model_limits_fill_ratio`, `test_force_full_bypasses_volume_constraint`, `test_moderate_order_gets_partial_fill`
  - `TestNegativeSharpeFailsDSR`: 4 test(s) -> `test_negative_sharpe_rejected`, `test_zero_sharpe_rejected`, `test_positive_sharpe_can_pass`, `test_weak_positive_sharpe_with_many_trials_fails`
  - `TestPBOThreshold`: 3 test(s) -> `test_config_pbo_threshold`, `test_pbo_above_045_is_overfit`, `test_pbo_max_combinations_increased`
  - `TestValidationRequiredForPromotion`: 5 test(s) -> `test_no_validation_rejects_promotion`, `test_negative_sharpe_always_rejected`, `test_mc_not_significant_rejects_promotion`, `test_all_validations_pass_allows_promotion`, `test_pbo_above_045_rejects_promotion`
  - `TestPartialExitMultiBar`: 3 test(s) -> `test_backtester_has_residual_tracking`, `test_volume_constrained_exit_records_fill_ratio`, `test_residual_position_exit_concept`

### `tests/test_bocpd.py`
- Intent: Unit tests for BOCPD (Bayesian Online Change-Point Detection).
- LOC: 345
- Test classes:
  - `TestBOCPDMeanShift`: 3 test(s) -> `test_detects_mean_shift`, `test_changepoint_probs_bounded`, `test_run_lengths_track_segments`
  - `TestBOCPDVarianceShift`: 1 test(s) -> `test_detects_variance_shift`
  - `TestBOCPDMultipleChangepoints`: 1 test(s) -> `test_detects_multiple_changepoints`
  - `TestBOCPDBatchOutput`: 4 test(s) -> `test_output_shapes`, `test_output_types`, `test_predicted_stds_positive`, `test_no_nan_in_output`
  - `TestBOCPDSingleUpdate`: 4 test(s) -> `test_first_update`, `test_sequential_updates`, `test_non_finite_observation_handled`, `test_inf_observation_handled`
  - `TestBOCPDEdgeCases`: 4 test(s) -> `test_empty_series`, `test_single_observation`, `test_constant_series`, `test_reset_clears_state`
  - `TestBOCPDParameterValidation`: 6 test(s) -> `test_invalid_hazard_lambda_zero`, `test_invalid_hazard_lambda_one`, `test_invalid_hazard_lambda_negative`, `test_invalid_max_runlength`, `test_invalid_hazard_func`, `test_geometric_hazard`
  - `TestBOCPDConsistency`: 1 test(s) -> `test_batch_matches_sequential`

### `tests/test_cache_metadata_rehydrate.py`
- Intent: Test module for cache metadata rehydrate behavior and regressions.
- LOC: 98
- Test classes:
  - `CacheMetadataRehydrateTests`: 3 test(s) -> `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`

### `tests/test_charting_endpoints.py`
- Intent: Tests for charting API endpoints and data flow (Spec 004).
- LOC: 320
- Test classes:
  - `TestBarsEndpoint`: 6 test(s) -> `test_daily_bars_valid_ohlcv`, `test_intraday_15min_loads`, `test_max_bars_limit`, `test_missing_timeframe_returns_not_found`, `test_missing_ticker_returns_not_found`, `test_range_based_filename`
  - `TestAvailableTimeframes`: 2 test(s) -> `test_discovers_daily_and_15min`, `test_bars_response_includes_available_timeframes`
  - `TestIndicatorEndpoint`: 7 test(s) -> `test_rsi_returns_panel_type`, `test_bollinger_returns_overlay_type`, `test_macd_returns_three_components`, `test_sma_returns_overlay`, `test_unknown_indicator_returns_error`, `test_available_indicators_list`, `test_stochastic_returns_k_and_d`
  - `TestParseIndicatorSpec`: 4 test(s) -> `test_rsi_14`, `test_macd_no_period`, `test_bollinger_20`, `test_sma_200`
  - `TestBatchIndicators`: 2 test(s) -> `test_multiple_indicators_single_call`, `test_batch_with_missing_data`
  - `TestFindCachedParquet`: 5 test(s) -> `test_exact_match`, `test_case_insensitive`, `test_range_based_pattern`, `test_no_match_returns_none`, `test_timeframe_aliases`

### `tests/test_conceptual_fixes.py`
- Intent: Tests for Spec 008 — Ensemble & Conceptual Fixes.
- LOC: 325
- Test classes:
  - `TestEnsembleNoPhantomVote`: 2 test(s) -> `test_two_method_ensemble_does_not_call_jump`, `test_two_method_ensemble_requires_unanimity`
  - `TestEnsembleWithJumpModel`: 2 test(s) -> `test_three_method_ensemble_calls_all_methods`, `test_three_method_probabilities_sum_to_one`
  - `TestRuleDetectMinDuration`: 3 test(s) -> `test_no_short_runs_with_min_duration`, `test_min_duration_1_is_noop`, `test_smoothing_preserves_regime_count`
  - `TestConfigEndpointDefaults`: 4 test(s) -> `test_adjustable_config_includes_backtest_keys`, `test_adjustable_config_values_match_config_py`, `test_config_status_includes_training_section`, `test_config_status_backtest_section_matches`

### `tests/test_covariance_estimator.py`
- Intent: Test module for covariance estimator behavior and regressions.
- LOC: 26
- Test classes:
  - `CovarianceEstimatorTests`: 1 test(s) -> `test_single_asset_covariance_is_2d_and_positive`

### `tests/test_data_diagnostics.py`
- Intent: Tests for data loading diagnostics (Spec 005).
- LOC: 184
- Test classes:
  - `TestOrchestratorImport`: 2 test(s) -> `test_orchestrator_imports_data_cache_dir`, `test_orchestrator_uses_data_cache_dir_in_diagnostics`
  - `TestLoadUniverseSkipReasons`: 2 test(s) -> `test_load_universe_logs_skip_reasons`, `test_skip_reasons_include_reason_text`
  - `TestErrorMessageDiagnostics`: 1 test(s) -> `test_error_message_includes_diagnostics`
  - `TestDataStatusService`: 4 test(s) -> `test_data_status_returns_summary`, `test_data_status_ticker_entries`, `test_data_status_with_missing_cache_dir`, `test_data_status_freshness_categories`

### `tests/test_delisting_total_return.py`
- Intent: Test module for delisting total return behavior and regressions.
- LOC: 77
- Test classes:
  - `DelistingTotalReturnTests`: 2 test(s) -> `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`

### `tests/test_drawdown_liquidation.py`
- Intent: Test module for drawdown liquidation behavior and regressions.
- LOC: 139
- Test classes:
  - `_FakePositionSizer`: 0 test(s)
  - `_FakeDrawdownController`: 0 test(s)
  - `_FakeStopLossManager`: 0 test(s)
  - `_FakePortfolioRisk`: 0 test(s)
  - `_FakeRiskMetrics`: 0 test(s)
  - `DrawdownLiquidationTests`: 1 test(s) -> `test_critical_drawdown_forces_liquidation`

### `tests/test_ensemble_voting.py`
- Intent: Tests for confidence-weighted ensemble voting (SPEC_10 T2).
- LOC: 221
- Test classes:
  - `TestConfidenceCalibrator`: 5 test(s) -> `test_fit_and_calibrate`, `test_component_weights_sum_to_one`, `test_hmm_gets_highest_weight`, `test_uncalibrated_returns_raw`, `test_ece_computed`
  - `TestWeightedEnsembleVoting`: 5 test(s) -> `test_ensemble_returns_valid_output`, `test_ensemble_regime_values_canonical`, `test_ensemble_probabilities_sum_to_one`, `test_ensemble_uncertainty_populated`, `test_calibration_changes_weights`

### `tests/test_evaluation_engine.py`
- Intent: Tests for evaluation/engine.py — EvaluationEngine integration tests.
- LOC: 326
- Test classes:
  - `TestWalkForwardWithEmbargo`: 4 test(s) -> `test_basic_walk_forward`, `test_embargo_prevents_leakage`, `test_insufficient_data`, `test_overfit_detection_on_noise`
  - `TestRollingIC`: 4 test(s) -> `test_rolling_ic_correlated_predictions`, `test_rolling_ic_random_predictions`, `test_detect_ic_decay_decaying_signal`, `test_detect_ic_decay_stable_signal`
  - `TestCalibrationAnalysis`: 3 test(s) -> `test_well_calibrated_model`, `test_insufficient_data`, `test_with_confidence_scores`
  - `TestEvaluationEngine`: 11 test(s) -> `test_full_evaluation`, `test_regime_slicing_populates`, `test_walk_forward_runs`, `test_rolling_ic_runs`, `test_fragility_with_trades`, `test_ml_diagnostics_with_importance`, `test_ensemble_disagreement`, `test_generate_json_report`, `test_generate_html_report`, `test_red_flags_raised_for_overfit`, `test_no_regime_still_works`

### `tests/test_evaluation_fragility.py`
- Intent: Tests for evaluation/fragility.py — PnL concentration, drawdown, recovery, critical slowing.
- LOC: 159
- Test classes:
  - `TestPnlConcentration`: 5 test(s) -> `test_single_trade_dominates`, `test_evenly_distributed_pnl`, `test_fragile_flag`, `test_empty_trades`, `test_herfindahl_index`
  - `TestDrawdownDistribution`: 3 test(s) -> `test_basic_drawdown_stats`, `test_no_drawdown_flat_returns`, `test_single_large_drawdown`
  - `TestRecoveryTimeDistribution`: 3 test(s) -> `test_recovery_times_are_positive`, `test_empty_returns`, `test_no_drawdown_no_recovery`
  - `TestCriticalSlowingDown`: 3 test(s) -> `test_increasing_recovery_time_detected`, `test_stable_recovery_time_no_detection`, `test_insufficient_data`
  - `TestConsecutiveLossFrequency`: 2 test(s) -> `test_loss_streaks_detected`, `test_all_positive_no_streaks`

### `tests/test_evaluation_metrics.py`
- Intent: Tests for evaluation/metrics.py — slice metrics and decile spread.
- LOC: 125
- Test classes:
  - `TestComputeSliceMetrics`: 7 test(s) -> `test_positive_returns_positive_sharpe`, `test_negative_returns_negative_sharpe`, `test_empty_returns`, `test_small_sample_low_confidence`, `test_max_drawdown_computed`, `test_win_rate_range`, `test_ic_with_predictions`
  - `TestDecileSpread`: 5 test(s) -> `test_perfect_predictions_positive_spread`, `test_random_predictions_near_zero_spread`, `test_insufficient_data`, `test_decile_counts_sum_to_total`, `test_per_regime_decomposition`

### `tests/test_evaluation_ml_diagnostics.py`
- Intent: Tests for evaluation/ml_diagnostics.py — feature drift and ensemble disagreement.
- LOC: 110
- Test classes:
  - `TestFeatureImportanceDrift`: 5 test(s) -> `test_stable_features_no_drift`, `test_drifting_features_detected`, `test_single_period_no_drift`, `test_top_k_stability`, `test_feature_names_tracked`
  - `TestEnsembleDisagreement`: 4 test(s) -> `test_identical_predictions_no_disagreement`, `test_random_predictions_low_correlation`, `test_single_model_no_disagreement`, `test_pairwise_correlations_stored`

### `tests/test_evaluation_slicing.py`
- Intent: Tests for evaluation/slicing.py — PerformanceSlice and SliceRegistry.
- LOC: 131
- Test classes:
  - `TestPerformanceSlice`: 3 test(s) -> `test_apply_returns_correct_subset`, `test_apply_empty_slice`, `test_apply_small_slice_flags_low_confidence`
  - `TestSliceRegistry`: 6 test(s) -> `test_create_regime_slices_returns_five`, `test_normal_slice_selects_regimes_0_and_1`, `test_high_vol_slice_selects_regime_3`, `test_build_metadata_has_required_columns`, `test_create_individual_regime_slices`, `test_slices_handle_volatility_input`

### `tests/test_execution_dynamic_costs.py`
- Intent: Test module for execution dynamic costs behavior and regressions.
- LOC: 49
- Test classes:
  - `ExecutionDynamicCostTests`: 1 test(s) -> `test_dynamic_costs_increase_under_stress`

### `tests/test_execution_layer.py`
- Intent: Comprehensive tests for Spec 06: Execution Layer Improvements.
- LOC: 849
- Test classes:
  - `TestBreakProbabilityMultiplier`: 11 test(s) -> `test_none_returns_unity`, `test_nan_returns_unity`, `test_zero_returns_low`, `test_below_low_threshold`, `test_at_medium_boundary`, `test_interpolation_low_to_medium`, `test_at_high_boundary`, `test_interpolation_medium_to_high`, `test_above_high_threshold`, `test_clipped_to_one`, `test_monotonic_increasing`
  - `TestStructureUncertaintyMultiplier`: 5 test(s) -> `test_none_returns_unity`, `test_zero_returns_unity`, `test_half_uncertainty`, `test_full_uncertainty`, `test_clipped_above_one`
  - `TestDriftScoreMultiplier`: 6 test(s) -> `test_none_returns_unity`, `test_zero_returns_unity`, `test_half_drift`, `test_full_drift`, `test_floor_at_0_70`, `test_monotonic_decreasing`
  - `TestSystemicStressMultiplier`: 3 test(s) -> `test_none_returns_unity`, `test_zero_returns_unity`, `test_full_stress`
  - `TestCompositeStructuralMultiplier`: 5 test(s) -> `test_all_none_returns_unity`, `test_disabled_returns_unity`, `test_multiplicative_combination`, `test_clipped_at_three`, `test_clipped_at_floor`
  - `TestStructuralCostsInSimulate`: 6 test(s) -> `test_high_break_prob_increases_cost`, `test_high_uncertainty_increases_cost`, `test_high_drift_decreases_cost`, `test_high_systemic_stress_increases_cost`, `test_structural_mult_in_fill`, `test_sell_side_structural_cost`
  - `TestADVTracker`: 13 test(s) -> `test_initial_adv`, `test_ema_smoothing`, `test_volume_trend_above_average`, `test_volume_trend_below_average`, `test_volume_trend_unknown_symbol`, `test_adjust_participation_limit_high_volume`, `test_adjust_participation_limit_low_volume`, `test_volume_cost_adjustment_high_volume`, `test_volume_cost_adjustment_low_volume`, `test_simple_adv`, `test_update_from_series`, `test_zero_volume_ignored` (+1 more)
  - `TestVolumeTrendInSimulate`: 1 test(s) -> `test_high_volume_trend_allows_larger_fills`
  - `TestNoTradeGate`: 4 test(s) -> `test_entry_blocked_during_extreme_stress`, `test_entry_allowed_below_threshold`, `test_exit_not_blocked_during_extreme_stress`, `test_no_urgency_type_no_gate`
  - `TestUrgencyDifferentiation`: 2 test(s) -> `test_urgency_type_recorded`, `test_exit_has_higher_cost_tolerance`
  - `TestCostCalibratorSegmentation`: 5 test(s) -> `test_micro_cap_classification`, `test_small_cap_classification`, `test_mid_cap_classification`, `test_large_cap_classification`, `test_invalid_marketcap_defaults_to_mid`
  - `TestCostCalibratorCoefficients`: 2 test(s) -> `test_default_coefficients`, `test_get_impact_coeff_by_marketcap`
  - `TestCostCalibratorCalibration`: 5 test(s) -> `test_insufficient_trades_returns_empty`, `test_calibration_updates_coefficients`, `test_smoothing_prevents_wild_swings`, `test_reset_history`, `test_record_trade_validation`
  - `TestCostCalibratorInExecution`: 1 test(s) -> `test_override_changes_impact`
  - `TestBackwardCompatibility`: 5 test(s) -> `test_no_structural_inputs_same_behavior`, `test_original_simulate_signature`, `test_original_fill_fields_present`, `test_zero_desired_returns_zero_fill`, `test_side_validation`
  - `TestExistingDynamicCosts`: 1 test(s) -> `test_dynamic_costs_increase_under_stress`
  - `TestIntegrationExecution`: 4 test(s) -> `test_full_execution_pipeline`, `test_exit_during_stress`, `test_calibrate_cost_model_legacy`, `test_cost_details_populated`

### `tests/test_feature_fixes.py`
- Intent: Tests verifying Spec 012 feature engineering fixes:
- LOC: 378
- Test classes:
  - `TestTSMomBackwardShift`: 2 test(s) -> `test_tsmom_uses_backward_shift`, `test_tsmom_values_dont_change_when_future_removed`
  - `TestIVShockNonFuture`: 2 test(s) -> `test_iv_shock_no_future_shift`, `test_iv_shock_no_lookahead`
  - `TestFeatureMetadata`: 3 test(s) -> `test_feature_metadata_exists`, `test_all_metadata_has_valid_type`, `test_pipeline_features_have_metadata`
  - `TestProductionModeFilter`: 2 test(s) -> `test_production_mode_filters_research`, `test_production_mode_default_false`
  - `TestRollingVWAPIsCausal`: 2 test(s) -> `test_rolling_vwap_is_causal`, `test_rolling_vwap_first_rows_nan`
  - `TestMultiHorizonVRP`: 1 test(s) -> `test_vrp_multi_horizon`
  - `TestFullPipelineLookahead`: 1 test(s) -> `test_automated_lookahead_detection`

### `tests/test_health_checks_rewritten.py`
- Intent: Tests for the rewritten health check system (Spec 010).
- LOC: 417
- Test classes:
  - `TestUnavailableScoring`: 4 test(s) -> `test_unavailable_helper_returns_zero`, `test_unavailable_signal_decay_no_data`, `test_unavailable_tail_risk_no_data`, `test_unavailable_execution_no_data`
  - `TestMethodologyField`: 3 test(s) -> `test_unavailable_has_methodology`, `test_signal_decay_has_methodology`, `test_all_checks_have_methodology`
  - `TestDomainScoring`: 4 test(s) -> `test_excludes_unavailable_from_average`, `test_all_unavailable_returns_none`, `test_domain_status_unavailable_majority`, `test_domain_status_pass`
  - `TestSignalDecayIC`: 2 test(s) -> `test_declining_ic_detected`, `test_stable_ic_passes`
  - `TestSurvivorshipCheck`: 2 test(s) -> `test_with_total_ret_column`, `test_without_total_ret_column`
  - `TestExecutionQualityTCA`: 2 test(s) -> `test_without_tca_uses_correlation_proxy`, `test_with_tca_columns`
  - `TestOverallScoreCalculation`: 2 test(s) -> `test_comprehensive_health_structure`, `test_weights_sum_to_one`
  - `TestHealthCheckResultDataclass`: 2 test(s) -> `test_to_dict`, `test_default_values`

### `tests/test_health_spec09.py`
- Intent: Tests for Health System Spec 09 enhancements.
- LOC: 721
- Test classes:
  - `TestInformationRatio`: 5 test(s) -> `test_ir_computed_on_valid_trades`, `test_ir_unavailable_no_trades`, `test_ir_unavailable_too_few_trades`, `test_ir_high_alpha_scores_well`, `test_ir_methodology_present`
  - `TestQuantifiedSurvivorshipBias`: 2 test(s) -> `test_quantified_pnl_loss`, `test_no_trades_falls_back_to_binary`
  - `TestHealthConfidenceIntervals`: 10 test(s) -> `test_bootstrap_ci`, `test_normal_ci`, `test_t_ci`, `test_binomial_ci`, `test_auto_selects_bootstrap_for_small_n`, `test_auto_selects_normal_for_large_n`, `test_insufficient_samples`, `test_ci_width_decreases_with_n`, `test_propagate_weighted_ci`, `test_to_dict_format`
  - `TestHealthHistoryTrending`: 8 test(s) -> `test_rolling_average_7d`, `test_rolling_average_short_series`, `test_trend_detection_improving`, `test_trend_detection_degrading`, `test_trend_detection_stable`, `test_trend_detection_too_few`, `test_history_with_trends_structure`, `test_save_and_retrieve`
  - `TestHealthRiskGate`: 11 test(s) -> `test_full_health_full_size`, `test_moderate_health_reduces_size`, `test_low_health_severe_reduction`, `test_zero_health_near_halt`, `test_apply_health_gate_scales_position`, `test_apply_health_gate_weights_array`, `test_should_halt_trading`, `test_disabled_gate_identity`, `test_smoothing`, `test_interpolation_between_breakpoints`, `test_get_status`
  - `TestHealthAlerts`: 8 test(s) -> `test_degradation_alert_triggered`, `test_degradation_alert_not_triggered`, `test_domain_failure_alert`, `test_no_domain_failure_above_threshold`, `test_deduplication`, `test_alert_to_dict`, `test_low_confidence_alert`, `test_null_health_no_degradation_alert`
  - `TestUpdatedDomainWeights`: 7 test(s) -> `test_execution_weight_increased`, `test_governance_weight_decreased`, `test_weights_sum_to_one`, `test_ir_check_included_in_signal_quality`, `test_comprehensive_health_has_ci`, `test_comprehensive_health_has_alerts`, `test_updated_methodology_string`

### `tests/test_health_transparency.py`
- Intent: Tests for health system transparency and trustworthiness (Spec 002).
- LOC: 283
- Test classes:
  - `TestUnavailableExclusion`: 4 test(s) -> `test_unavailable_helper_returns_correct_fields`, `test_domain_score_excludes_unavailable`, `test_all_unavailable_returns_none`, `test_no_hardcoded_50_fallback`
  - `TestMethodologyPresence`: 3 test(s) -> `test_healthcheckresult_has_methodology_field`, `test_unavailable_has_methodology`, `test_healthcheckresult_to_dict_includes_methodology`
  - `TestSeverityWeighting`: 3 test(s) -> `test_severity_weights_defined`, `test_critical_failure_impacts_more`, `test_severity_field_on_dataclass`
  - `TestHealthHistory`: 2 test(s) -> `test_save_and_retrieve_history`, `test_history_respects_limit`
  - `TestAPIMethodology`: 2 test(s) -> `test_healthcheckresult_serialization`, `test_overall_methodology_in_comprehensive_health`

### `tests/test_integration.py`
- Intent: End-to-end integration tests for the quant engine pipeline.
- LOC: 557
- Test classes:
  - `TestFullPipelineSynthetic`: 6 test(s) -> `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`
  - `TestCvGapHardBlock`: 1 test(s) -> `test_cv_gap_hard_block`
  - `TestRegime2Suppression`: 2 test(s) -> `test_regime_2_suppression`, `test_regime_0_not_suppressed`
  - `TestCrossSectionalRanking`: 4 test(s) -> `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`

### `tests/test_iv_arbitrage_builder.py`
- Intent: Test module for iv arbitrage builder behavior and regressions.
- LOC: 39
- Test classes:
  - `ArbitrageFreeSVIBuilderTests`: 1 test(s) -> `test_build_surface_has_valid_shape_and_monotone_total_variance`

### `tests/test_jump_model_pypi.py`
- Intent: Unit tests for the PyPI JumpModel wrapper (Spec 001).
- LOC: 379
- Test classes:
  - `TestFitBasic`: 3 test(s) -> `test_fit_returns_jump_model_result`, `test_fit_output_shapes`, `test_fit_output_dtypes`
  - `TestValidRegimes`: 1 test(s) -> `test_fit_returns_valid_regimes`
  - `TestProbsSumToOne`: 3 test(s) -> `test_fit_probs_sum_to_one`, `test_probs_non_negative`, `test_probs_no_nan`
  - `TestCVLambdaSelection`: 2 test(s) -> `test_cv_lambda_within_range`, `test_cv_lambda_is_float`
  - `TestFallbackShortData`: 1 test(s) -> `test_short_data_raises`
  - `TestNaNHandling`: 2 test(s) -> `test_nan_features_handled`, `test_all_nan_column_raises`
  - `TestPredictOnline`: 4 test(s) -> `test_predict_online_shape`, `test_predict_online_valid_labels`, `test_predict_online_not_fitted_raises`, `test_predict_proba_online_shape`
  - `TestRegimePersistence`: 1 test(s) -> `test_higher_lambda_fewer_transitions`
  - `TestContinuousVsDiscrete`: 2 test(s) -> `test_discrete_mode_hard_probs`, `test_wrapper_always_returns_soft_probs`
  - `TestCanonicalMapping`: 1 test(s) -> `test_canonical_mapping`

### `tests/test_jump_model_validation.py`
- Intent: Jump Model Audit and Validation (SPEC_10 T1).
- LOC: 397
- Test classes:
  - `TestSingleLargeJump`: 2 test(s) -> `test_legacy_detects_single_jump`, `test_pypi_detects_single_jump`
  - `TestSmallJumpsNotOverDetected`: 1 test(s) -> `test_small_jumps_few_false_transitions`
  - `TestNoiseFalsePositiveRate`: 1 test(s) -> `test_noise_legacy_low_fp_rate`
  - `TestPrecisionRecall`: 2 test(s) -> `test_precision_recall_legacy`, `test_precision_recall_pypi`
  - `TestComputationTime`: 3 test(s) -> `test_legacy_fit_under_2_seconds`, `test_pypi_fit_under_5_seconds`, `test_legacy_predict_under_100ms`
  - `TestLegacyVsPyPIAgreement`: 1 test(s) -> `test_models_agree_on_structured_data`

### `tests/test_kalshi_asof_features.py`
- Intent: Test module for kalshi asof features behavior and regressions.
- LOC: 66
- Test classes:
  - `KalshiAsofFeatureTests`: 2 test(s) -> `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`

### `tests/test_kalshi_distribution.py`
- Intent: Test module for kalshi distribution behavior and regressions.
- LOC: 115
- Test classes:
  - `KalshiDistributionTests`: 3 test(s) -> `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`

### `tests/test_kalshi_hardening.py`
- Intent: Test module for kalshi hardening behavior and regressions.
- LOC: 606
- Test classes:
  - `KalshiHardeningTests`: 16 test(s) -> `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof` (+4 more)

### `tests/test_loader_and_predictor.py`
- Intent: Test module for loader and predictor behavior and regressions.
- LOC: 231
- Test classes:
  - `_FakeWRDSProvider`: 0 test(s)
  - `_UnavailableWRDSProvider`: 0 test(s)
  - `LoaderAndPredictorTests`: 8 test(s) -> `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`

### `tests/test_lookahead_detection.py`
- Intent: Automated lookahead bias detection for the feature pipeline.
- LOC: 145
- Test classes:
  - `TestLookaheadDetection`: 2 test(s) -> `test_no_feature_uses_future_data`, `test_no_feature_uses_future_data_deep`

### `tests/test_observation_matrix_expansion.py`
- Intent: Tests for expanded HMM observation matrix (SPEC_10 T4).
- LOC: 186
- Test classes:
  - `TestObservationMatrixExpansion`: 5 test(s) -> `test_structural_features_included`, `test_expanded_feature_count`, `test_graceful_fallback_without_structural`, `test_all_features_standardized`, `test_no_nan_or_inf`
  - `TestBICWithExpandedFeatures`: 2 test(s) -> `test_bic_runs_with_structural`, `test_hmm_fits_with_expanded_features`
  - `TestBOCPDInlineComputation`: 1 test(s) -> `test_bocpd_computed_inline`

### `tests/test_online_update.py`
- Intent: Tests for online regime updating via forward algorithm (SPEC_10 T5).
- LOC: 237
- Test classes:
  - `TestForwardStep`: 2 test(s) -> `test_forward_step_valid_probabilities`, `test_forward_step_concentrates_on_correct_state`
  - `TestSecurityTracking`: 3 test(s) -> `test_update_regime_returns_valid`, `test_security_cache_updated`, `test_cache_reset`
  - `TestOnlineVsFullRefit`: 1 test(s) -> `test_online_agrees_with_full_refit`
  - `TestBatchUpdate`: 1 test(s) -> `test_batch_update_returns_all_securities`
  - `TestRefitScheduling`: 2 test(s) -> `test_should_refit_after_interval`, `test_should_not_refit_before_interval`
  - `TestOnlinePerformance`: 1 test(s) -> `test_batch_update_faster_than_full_refit`

### `tests/test_panel_split.py`
- Intent: Test module for panel split behavior and regressions.
- LOC: 56
- Test classes:
  - `PanelSplitTests`: 2 test(s) -> `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`

### `tests/test_paper_trader_integration.py`
- Intent: Test module for paper trader integration — spec 015.
- LOC: 844
- Test classes:
  - `TestDrawdownBlocksEntries`: 1 test(s) -> `test_drawdown_blocks_entries`
  - `TestDrawdownForceLiquidate`: 1 test(s) -> `test_drawdown_force_liquidate`
  - `TestStopLossTriggersExit`: 1 test(s) -> `test_hard_stop_fires`
  - `TestTrailingStopTriggers`: 1 test(s) -> `test_trailing_stop_fires`
  - `TestRiskManagerBlocksCorrelated`: 1 test(s) -> `test_correlated_entry_blocked`
  - `TestRegimeStoredInPosition`: 1 test(s) -> `test_regime_stored_in_position`
  - `TestRegimeStoredInTrade`: 1 test(s) -> `test_regime_stored_in_trade`
  - `TestRegimePassedToSizer`: 1 test(s) -> `test_regime_affects_position_size`
  - `TestEquityCurveTracked`: 1 test(s) -> `test_equity_curve_tracked`
  - `TestRiskMetricsComputed`: 2 test(s) -> `test_risk_metrics_computed`, `test_no_risk_metrics_when_insufficient_data`
  - `TestEquityCurveCapped`: 1 test(s) -> `test_equity_curve_capped`
  - `TestBackwardCompatOldStateFormat`: 1 test(s) -> `test_backward_compat`

### `tests/test_paper_trader_kelly.py`
- Intent: Test module for paper trader kelly behavior and regressions.
- LOC: 129
- Test classes:
  - `PaperTraderKellyTests`: 1 test(s) -> `test_kelly_sizing_changes_position_size_with_bounds`

### `tests/test_portfolio_layer.py`
- Intent: Comprehensive test suite for Spec 07: Portfolio Layer + Regime-Conditioned Constraints.
- LOC: 1027
- Test classes:
  - `_TempUniverseYAML`: 0 test(s)
  - `TestUniverseConfig`: 14 test(s) -> `test_load_valid_yaml`, `test_get_sector_known_ticker`, `test_get_sector_unknown_ticker`, `test_get_sector_case_insensitive`, `test_get_sector_constituents`, `test_liquidity_tier`, `test_borrowability`, `test_constraint_base`, `test_stress_multipliers`, `test_factor_bounds`, `test_missing_file_raises`, `test_invalid_yaml_raises` (+2 more)
  - `TestConstraintMultiplier`: 4 test(s) -> `test_normal_regime_multipliers_are_unity`, `test_stress_regime_multipliers_tighten`, `test_is_stress_regime`, `test_with_universe_config`
  - `TestSmoothTransitions`: 4 test(s) -> `test_smoothed_multipliers_transition_gradually`, `test_smoothed_alpha_1_is_immediate`, `test_reset_clears_smoothing_state`, `test_regime_transition_0_to_3_to_0`
  - `TestPortfolioRiskRegime`: 6 test(s) -> `test_normal_regime_uses_base_constraints`, `test_stress_regime_tightens_sector_cap`, `test_constraint_utilization_returned`, `test_backoff_recommended_when_near_binding`, `test_backward_compatible_without_regime`, `test_compute_constraint_utilization`
  - `TestRegimeConditionalCorrelation`: 4 test(s) -> `test_correlation_check_without_regime_labels`, `test_correlation_check_with_regime_labels`, `test_regime_cov_cache_populated`, `test_invalidate_cache`
  - `TestFactorExposureManager`: 6 test(s) -> `test_compute_exposures_basic`, `test_beta_near_one_for_balanced_portfolio`, `test_check_factor_bounds_normal`, `test_check_factor_bounds_stress_violation`, `test_unconstrained_factors_not_violated`, `test_empty_weights`
  - `TestSizingBackoff`: 6 test(s) -> `test_no_backoff_below_threshold`, `test_backoff_at_90_pct`, `test_backoff_at_95_pct`, `test_multiple_constraints_cumulative`, `test_empty_utilization_no_change`, `test_custom_backoff_policy`
  - `TestConstraintReplay`: 4 test(s) -> `test_replay_basic`, `test_replay_detects_concentration`, `test_replay_empty_history`, `test_robustness_score_perfect`
  - `TestPortfolioLayerIntegration`: 6 test(s) -> `test_full_workflow_normal_regime`, `test_full_workflow_stress_regime`, `test_sizing_backoff_integration`, `test_universe_config_integrated_with_risk_manager`, `test_constraint_replay_integration`, `test_portfolio_summary_uses_universe_config`

### `tests/test_position_sizing_overhaul.py`
- Intent: Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul.
- LOC: 425
- Test classes:
  - `TestKellyNegativeEdge`: 5 test(s) -> `test_kelly_negative_edge_returns_zero`, `test_kelly_positive_edge_returns_positive`, `test_kelly_invalid_inputs_return_zero`, `test_kelly_small_sample_penalty`, `test_kelly_n_trades_passed_through_size_position`
  - `TestDrawdownGovernor`: 6 test(s) -> `test_convex_at_50pct_drawdown`, `test_aggressive_at_90pct_drawdown`, `test_no_drawdown_full_sizing`, `test_beyond_max_dd_returns_zero`, `test_positive_drawdown_returns_full`, `test_convex_curve_is_smooth`
  - `TestPerRegimeBayesian`: 3 test(s) -> `test_bull_kelly_greater_than_hv`, `test_global_fallback_when_few_regime_trades`, `test_per_regime_counters_populated`
  - `TestRegimeStatsUpdate`: 3 test(s) -> `test_update_from_trade_df`, `test_too_few_trades_keeps_defaults`, `test_multiple_regimes_updated`
  - `TestConfidenceScalar`: 3 test(s) -> `test_max_confidence_scalar_is_one`, `test_zero_confidence_scalar_is_half`, `test_confidence_never_amplifies`
  - `TestPortfolioCorrelationPenalty`: 3 test(s) -> `test_high_corr_smaller_than_uncorr`, `test_no_positions_returns_base`, `test_negative_corr_no_penalty`
  - `TestDrawdownControllerIntegration`: 5 test(s) -> `test_normal_state_allows_entries`, `test_large_loss_triggers_caution`, `test_critical_drawdown_forces_liquidation`, `test_paper_trader_has_drawdown_controller`, `test_recovery_allows_gradual_reentry`
  - `TestSizePositionIntegration`: 2 test(s) -> `test_negative_edge_signal_still_gets_sized_via_blend`, `test_drawdown_reduces_kelly_in_blend`

### `tests/test_promotion_contract.py`
- Intent: Test module for promotion contract behavior and regressions.
- LOC: 108
- Test classes:
  - `PromotionContractTests`: 2 test(s) -> `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`

### `tests/test_provider_registry.py`
- Intent: Test module for provider registry behavior and regressions.
- LOC: 30
- Test classes:
  - `ProviderRegistryTests`: 3 test(s) -> `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`

### `tests/test_regime_consensus.py`
- Intent: Tests for cross-sectional regime consensus (SPEC_10 T6).
- LOC: 205
- Test classes:
  - `TestComputeConsensus`: 5 test(s) -> `test_unanimous_consensus`, `test_80_percent_consensus`, `test_low_consensus_early_warning`, `test_empty_securities`, `test_regime_pcts_sum_to_one`
  - `TestDivergenceDetection`: 3 test(s) -> `test_falling_consensus_detected`, `test_stable_consensus_not_diverging`, `test_insufficient_history`
  - `TestEarlyWarning`: 2 test(s) -> `test_warning_below_threshold`, `test_no_warning_above_threshold`
  - `TestConsensusSeries`: 2 test(s) -> `test_consensus_series_shape`, `test_consensus_drops_during_transition`

### `tests/test_regime_detection_integration.py`
- Intent: End-to-end integration tests for the upgraded regime detection pipeline (SPEC_10 T7).
- LOC: 610
- Test classes:
  - `TestEnsembleVotingEndToEnd`: 5 test(s) -> `test_ensemble_produces_valid_output`, `test_ensemble_regimes_canonical`, `test_ensemble_probabilities_valid`, `test_ensemble_uncertainty_populated`, `test_ensemble_confidence_range`
  - `TestUncertaintyGateIntegration`: 4 test(s) -> `test_uncertainty_gate_reduces_sizes`, `test_gate_series_produces_valid_output`, `test_stress_assumption_on_high_uncertainty`, `test_weight_reduction_applied`
  - `TestRegimeTransitions`: 3 test(s) -> `test_detects_multiple_regimes`, `test_regime_features_complete`, `test_regime_duration_computed`
  - `TestConsensusIntegration`: 3 test(s) -> `test_consensus_from_multi_security`, `test_consensus_series_from_batch`, `test_correlated_securities_have_high_consensus`
  - `TestOnlineUpdateIntegration`: 2 test(s) -> `test_online_update_after_hmm_fit`, `test_online_batch_update`
  - `TestConfidenceCalibrationIntegration`: 2 test(s) -> `test_calibrate_and_use_in_ensemble`, `test_calibrator_ecm_populated`
  - `TestMinRegimeSamplesConfig`: 4 test(s) -> `test_min_regime_samples_value`, `test_min_regime_days_value`, `test_short_regime_would_train`, `test_very_short_regime_blocked`
  - `TestFullPipelineEndToEnd`: 5 test(s) -> `test_detect_with_shock_context`, `test_batch_regime_detection`, `test_pipeline_ensemble_uncertainty_consensus`, `test_regime_exports_complete`, `test_config_constants_exist`

### `tests/test_regime_integration.py`
- Intent: Integration tests for the full regime detection pipeline with Jump Model (Spec 001).
- LOC: 312
- Test classes:
  - `TestDetectFullJumpModel`: 7 test(s) -> `test_returns_valid_regime_output`, `test_regime_values_in_canonical_set`, `test_probabilities_nan_free`, `test_probabilities_sum_to_one`, `test_confidence_in_valid_range`, `test_uncertainty_populated`, `test_transition_matrix_none_for_jump`
  - `TestEnsembleIncludesJump`: 2 test(s) -> `test_ensemble_runs_with_jump`, `test_ensemble_probabilities_valid`
  - `TestRegimeFeaturesUnchanged`: 3 test(s) -> `test_regime_features_columns_match`, `test_regime_features_required_columns`, `test_regime_features_no_nan`
  - `TestComputeRegimePayloadJump`: 1 test(s) -> `test_payload_with_jump_model_output`
  - `TestHMMStillWorks`: 2 test(s) -> `test_hmm_detect_full`, `test_rule_detect_full`
  - `TestLegacyFallback`: 1 test(s) -> `test_legacy_jump_model`

### `tests/test_regime_payload.py`
- Intent: Tests for regime data pipeline end-to-end (Spec 003).
- LOC: 242
- Test classes:
  - `TestRegimeNaN`: 2 test(s) -> `test_hmm_observation_matrix_no_inf`, `test_detector_no_nan_probs`
  - `TestRegimeTimeline`: 2 test(s) -> `test_regime_changes_structure`, `test_regime_changes_limited_to_20`
  - `TestRegimeMetadata`: 3 test(s) -> `test_metadata_returns_all_regimes`, `test_metadata_has_detection_method`, `test_metadata_has_matrix_explanation`
  - `TestRegimeFallback`: 1 test(s) -> `test_empty_cache_returns_graceful_fallback`

### `tests/test_regime_uncertainty.py`
- Intent: Tests for regime uncertainty integration (SPEC_10 T3).
- LOC: 163
- Test classes:
  - `TestRegimeUncertaintyEntropy`: 4 test(s) -> `test_uniform_posterior_max_entropy`, `test_concentrated_posterior_low_entropy`, `test_degenerate_posterior_zero_entropy`, `test_entropy_in_valid_range`
  - `TestUncertaintyGate`: 9 test(s) -> `test_zero_entropy_full_size`, `test_moderate_entropy_reduces_size`, `test_high_entropy_reduces_more`, `test_multiplier_never_below_floor`, `test_apply_uncertainty_gate_reduces_weights`, `test_should_assume_stress_high_entropy`, `test_is_uncertain_flag`, `test_gate_series`, `test_custom_sizing_map`

### `tests/test_research_factors.py`
- Intent: Test module for research factors behavior and regressions.
- LOC: 134
- Test classes:
  - `ResearchFactorTests`: 4 test(s) -> `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`

### `tests/test_risk_governor.py`
- Intent: Comprehensive tests for Spec 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing.
- LOC: 713
- Test classes:
  - `TestUncertaintyScaling`: 8 test(s) -> `test_high_uncertainty_reduces_size`, `test_low_uncertainty_near_full_size`, `test_none_inputs_no_scaling`, `test_partial_none_no_scaling`, `test_uncertainty_scale_bounds`, `test_uncertainty_scale_with_invalid_inputs`, `test_drift_score_interpretation`, `test_sizing_details_includes_uncertainty`
  - `TestShockBudget`: 3 test(s) -> `test_position_capped_by_shock_budget`, `test_small_position_passes_through`, `test_shock_budget_applied_in_size_position`
  - `TestTurnoverBudget`: 5 test(s) -> `test_turnover_within_budget_passes`, `test_turnover_budget_exhausted_blocks_increase`, `test_turnover_disabled_passes_through`, `test_turnover_tracking_accumulates`, `test_reset_turnover_tracking`
  - `TestConcentrationLimit`: 3 test(s) -> `test_position_capped_at_limit`, `test_position_below_limit_unchanged`, `test_concentration_limit_in_size_position`
  - `TestRegimeConditionalBlendWeights`: 5 test(s) -> `test_different_regimes_different_blends`, `test_blend_weights_sum_to_one`, `test_static_fallback_when_no_regime_config`, `test_regime_mapping`, `test_blend_size_details_in_result`
  - `TestParameterizedKellyBayesian`: 4 test(s) -> `test_min_samples_threshold`, `test_large_sample_converges_to_empirical`, `test_different_priors_produce_different_estimates`, `test_regime_specific_bayesian_with_min_samples`
  - `TestPaperTraderUnifiedInterface`: 5 test(s) -> `test_paper_trader_method_exists`, `test_basic_sizing_output`, `test_few_trades_conservative`, `test_uncertainty_passed_through`, `test_budget_constraints_applied`
  - `TestBudgetConstraintInteractions`: 2 test(s) -> `test_shock_then_concentration`, `test_without_portfolio_equity_no_constraints`
  - `TestBackwardCompatibility`: 8 test(s) -> `test_size_position_original_signature`, `test_size_position_with_regime_and_drawdown`, `test_kelly_negative_edge_still_zero`, `test_drawdown_governor_unchanged`, `test_portfolio_aware_still_works`, `test_update_regime_stats_unchanged`, `test_update_kelly_bayesian_unchanged`, `test_size_portfolio_unchanged`
  - `TestConfigParameters`: 4 test(s) -> `test_risk_governor_config_loaded`, `test_shock_budget_default`, `test_concentration_limit_default`, `test_blend_weights_by_regime_has_all_states`
  - `TestFullSizingPipeline`: 3 test(s) -> `test_all_features_enabled`, `test_zero_equity_no_budget_constraints`, `test_multiple_positions_turnover_tracking`

### `tests/test_risk_improvements.py`
- Intent: Comprehensive tests for Spec 016: Risk System Improvements.
- LOC: 493
- Test classes:
  - `TestLedoitWolfDataDrivenShrinkage`: 3 test(s) -> `test_shrinkage_comes_from_data`, `test_covariance_is_psd`, `test_covariance_shape`
  - `TestEWMACovariance`: 2 test(s) -> `test_ewma_reflects_recent_vol_increase`, `test_ewma_covariance_is_psd`
  - `TestParametricVaR`: 2 test(s) -> `test_parametric_var_populated`, `test_historical_captures_fat_tails_better`
  - `TestCornishFisherVaR`: 2 test(s) -> `test_cornish_fisher_populated`, `test_cornish_fisher_adjusts_for_negative_skew`
  - `TestCorrelationStress`: 2 test(s) -> `test_stress_vol_exceeds_normal`, `test_single_asset_no_diversification_effect`
  - `TestRollingAttribution`: 1 test(s) -> `test_components_sum_to_total`
  - `TestStopLossSpreadBuffer`: 3 test(s) -> `test_buffer_lowers_initial_stop`, `test_buffer_amount_correct`, `test_zero_buffer_no_effect`
  - `TestRecoveryRampConcave`: 4 test(s) -> `test_early_recovery_more_cautious`, `test_quadratic_vs_linear_values`, `test_full_recovery_reaches_one`, `test_new_entries_blocked_early_recovery`
  - `TestFactorStressScenarios`: 5 test(s) -> `test_crisis_scenarios_defined`, `test_factor_stress_produces_negative_returns`, `test_factor_contributions_sum_to_total`, `test_zero_exposure_no_impact`, `test_custom_scenarios`

### `tests/test_shock_vector.py`
- Intent: Unit tests for ShockVector and ShockVectorValidator.
- LOC: 373
- Test classes:
  - `TestShockVectorConstruction`: 8 test(s) -> `test_default_construction`, `test_full_construction`, `test_invalid_schema_version_raises`, `test_invalid_regime_raises`, `test_negative_regime_raises`, `test_confidence_clipped_to_bounds`, `test_changepoint_prob_clipped`, `test_negative_runlength_clipped`
  - `TestShockVectorValidator`: 9 test(s) -> `test_valid_shock_vector`, `test_validates_all_regimes`, `test_validates_all_model_types`, `test_rejects_invalid_model_type`, `test_validates_structural_features_types`, `test_rejects_non_numeric_structural_feature`, `test_rejects_non_finite_structural_feature`, `test_validates_transition_matrix_shape`, `test_rejects_wrong_transition_matrix_shape`
  - `TestShockVectorBatchValidation`: 3 test(s) -> `test_all_valid`, `test_some_invalid`, `test_empty_batch`
  - `TestShockVectorSerialization`: 3 test(s) -> `test_to_dict`, `test_from_dict`, `test_round_trip`
  - `TestShockEventDetection`: 5 test(s) -> `test_no_shock`, `test_shock_from_jump`, `test_shock_from_changepoint`, `test_shock_from_large_move`, `test_custom_threshold`
  - `TestRegimeName`: 1 test(s) -> `test_all_regime_names`

### `tests/test_signal_meta_labeling.py`
- Intent: Tests for Spec 04: Signal Enhancement + Meta-Labeling + Fold-Level Validation.
- LOC: 757
- Test classes:
  - `TestCrossSectionalTopK`: 4 test(s) -> `test_topk_quantile_selects_correct_fraction`, `test_topk_quantile_extreme_dispersion`, `test_topk_quantile_single_stock_keeps_all`, `test_topk_config_quantile_value`
  - `TestMetaLabelingModel`: 13 test(s) -> `test_build_meta_features_shape`, `test_build_meta_features_no_nans`, `test_build_meta_features_all_zeros_input`, `test_build_meta_features_regime_one_hot`, `test_build_labels_correct_direction`, `test_build_labels_all_below_threshold`, `test_train_produces_model`, `test_train_insufficient_samples_raises`, `test_predict_confidence_range`, `test_predict_untrained_raises`, `test_feature_importance_no_single_dominant`, `test_save_and_load_roundtrip` (+1 more)
  - `TestFoldMetrics`: 5 test(s) -> `test_walk_forward_returns_fold_metrics`, `test_fold_win_rate_range`, `test_fold_profit_factor_nonnegative`, `test_backward_compat_aggregate_still_works`, `test_insufficient_data_returns_empty_folds`
  - `TestFoldConsistency`: 9 test(s) -> `test_perfect_consistency`, `test_zero_consistency_negative_mean`, `test_low_consistency_high_variance`, `test_moderate_consistency`, `test_single_fold_returns_one`, `test_clipping_bounds`, `test_fold_consistency_integrated_into_score`, `test_fold_consistency_key_in_sharpe_format`, `test_config_weight_default`
  - `TestMetaLabelingFiltering`: 3 test(s) -> `test_high_confidence_passes`, `test_all_above_threshold_no_filtering`, `test_all_below_threshold_all_filtered`
  - `TestConfigDefaults`: 3 test(s) -> `test_signal_topk_quantile`, `test_meta_labeling_confidence_threshold`, `test_fold_consistency_weight`
  - `TestIntegration`: 5 test(s) -> `test_full_pipeline_meta_features_through_training`, `test_topk_then_meta_labeling_reduces_signals`, `test_fold_metrics_feed_into_promotion_gate`, `test_backward_compat_meta_labeling_disabled`, `test_backward_compat_no_fold_metrics`

### `tests/test_structural_features.py`
- Intent: Tests for Spec 02: Structural Feature Expansion.
- LOC: 821
- Test classes:
  - `TestSpectralAnalyzer`: 8 test(s) -> `test_detects_weekly_cycle`, `test_hf_lf_energy_non_negative`, `test_spectral_entropy_bounded`, `test_noisy_signal_high_entropy`, `test_compute_all_consistent`, `test_output_length_matches_input`, `test_bandwidth_non_negative`, `test_invalid_window_raises`
  - `TestSSADecomposer`: 7 test(s) -> `test_trending_signal_high_trend_strength`, `test_noisy_signal_high_noise_ratio`, `test_singular_entropy_bounded`, `test_oscillatory_strength_bounded`, `test_trend_noise_osc_sum_to_one`, `test_compute_all_returns_all_keys`, `test_embed_dim_validation`
  - `TestTailRiskAnalyzer`: 6 test(s) -> `test_jump_intensity_detects_jumps`, `test_expected_shortfall_negative`, `test_vol_of_vol_non_negative`, `test_srm_non_negative`, `test_extreme_pct_bounded`, `test_compute_all_returns_all_keys`
  - `TestEigenvalueAnalyzer`: 7 test(s) -> `test_high_concentration_when_correlated`, `test_low_concentration_when_independent`, `test_effective_rank_range`, `test_avg_corr_stress_range`, `test_condition_number_non_negative`, `test_too_few_assets_returns_nan`, `test_compute_all_consistency`
  - `TestOptimalTransportAnalyzer`: 6 test(s) -> `test_wasserstein_detects_shift`, `test_wasserstein_non_negative`, `test_sinkhorn_non_negative`, `test_identical_distributions_near_zero`, `test_compute_all_returns_all_keys`, `test_invalid_params_raise`
  - `TestStructuralFeaturesIntegration`: 5 test(s) -> `test_compute_structural_features_produces_columns`, `test_structural_features_same_index`, `test_structural_features_no_inf`, `test_feature_metadata_completeness`, `test_feature_metadata_categories`
  - `TestFeatureRedundancy`: 4 test(s) -> `test_no_false_positives_on_independent`, `test_detects_correlated_features`, `test_validate_composition_passes_on_normal_data`, `test_report_formatting`
  - `TestStructuralConfig`: 4 test(s) -> `test_config_params_exist`, `test_ssa_embed_dim_less_than_window`, `test_wasserstein_ref_gte_window`, `test_sinkhorn_epsilon_positive`

### `tests/test_structural_state_integration.py`
- Intent: Integration tests for the Structural State Layer (SPEC_03).
- LOC: 385
- Test classes:
  - `TestConfigConstants`: 2 test(s) -> `test_bocpd_config_constants`, `test_shock_vector_config_constants`
  - `TestRegimeDetectorBOCPD`: 4 test(s) -> `test_detector_initializes_with_bocpd`, `test_detector_initializes_without_bocpd`, `test_detect_still_works_with_bocpd`, `test_detect_still_works_without_bocpd`
  - `TestDetectWithShockContext`: 6 test(s) -> `test_returns_shock_vector`, `test_returns_shock_vector_without_bocpd`, `test_shock_vector_validates`, `test_shock_vector_has_model_type`, `test_jump_detection_on_synthetic_data`, `test_bocpd_signals_populated`
  - `TestBatchShockVectorGeneration`: 2 test(s) -> `test_batch_returns_dict`, `test_batch_all_vectors_valid`
  - `TestHMMObservationMatrixValidation`: 4 test(s) -> `test_valid_features`, `test_missing_core_feature`, `test_high_nan_warning`, `test_reports_missing_extended_features`
  - `TestPackageExports`: 3 test(s) -> `test_bocpd_exports`, `test_shock_vector_exports`, `test_validation_export`
  - `TestShockVectorJSON`: 2 test(s) -> `test_to_dict_is_json_serializable`, `test_to_dict_excludes_transition_matrix`

### `tests/test_survivorship_pit.py`
- Intent: Test module for survivorship pit behavior and regressions.
- LOC: 75
- Test classes:
  - `SurvivorshipPointInTimeTests`: 1 test(s) -> `test_filter_panel_by_point_in_time_universe`

### `tests/test_system_innovations.py`
- Intent: Comprehensive tests for Spec 017: System-Level Innovation.
- LOC: 773
- Test classes:
  - `TestCUSUMDetectsMeanShift`: 4 test(s) -> `test_detects_upward_shift`, `test_detects_downward_shift`, `test_no_false_alarm_stationary`, `test_too_few_samples`
  - `TestPSIDetectsDistributionChange`: 4 test(s) -> `test_psi_detects_mean_shift`, `test_psi_stable_distributions`, `test_psi_top_shifted_features`, `test_check_all_combined`
  - `TestConformalCoverage`: 2 test(s) -> `test_90_percent_coverage`, `test_wider_coverage_target`
  - `TestConformalUncertaintyScaling`: 4 test(s) -> `test_wider_interval_smaller_scalar`, `test_scalar_range`, `test_calibration_requires_min_samples`, `test_predict_before_calibration_raises`
  - `TestMultiHorizonBlending`: 4 test(s) -> `test_trending_bull_weights_long_horizons`, `test_high_vol_weights_short_horizons`, `test_different_regimes_produce_different_blends`, `test_single_horizon_fallback`
  - `TestRegimeStrategyProfiles`: 4 test(s) -> `test_all_regimes_have_profiles`, `test_profiles_are_distinct`, `test_high_vol_is_most_conservative`, `test_bull_is_most_aggressive`
  - `TestConfidenceBlending`: 3 test(s) -> `test_full_confidence_uses_regime_params`, `test_zero_confidence_uses_defaults`, `test_partial_confidence_blends`
  - `TestFactorExposureLimits`: 4 test(s) -> `test_alert_on_extreme_beta`, `test_pass_on_normal_exposures`, `test_empty_portfolio_no_crash`, `test_multiple_violations`
  - `TestCostBudgetFullRebalance`: 1 test(s) -> `test_within_budget_full_execution`
  - `TestCostBudgetPartialRebalance`: 2 test(s) -> `test_partial_execution_over_budget`, `test_no_trades_needed`
  - `TestDiagnosticsIdentifyAlphaDecay`: 3 test(s) -> `test_alpha_decay_detected`, `test_stale_data_detected`, `test_model_staleness_detected`
  - `TestDiagnosticsPerforming`: 3 test(s) -> `test_positive_returns_performing`, `test_no_equity_curve_unknown`, `test_unfavorable_regime_high_vol`
  - `TestSurvivorshipBiasComparison`: 3 test(s) -> `test_survivors_higher_sharpe`, `test_comparison_summary`, `test_quick_survivorship_check`
  - `TestDiverseEnsembleExists`: 2 test(s) -> `test_ensemble_diversify_config_exists`, `test_diverse_ensemble_class_exists`
  - `TestShiftDetectionRetrainIntegration`: 2 test(s) -> `test_retrain_trigger_has_shift_detector`, `test_check_shift_returns_results`
  - `TestConformalSerialization`: 1 test(s) -> `test_round_trip`

### `tests/test_training_pipeline_fixes.py`
- Intent: Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance.
- LOC: 568
- Test classes:
  - `TestPerFoldFeatureSelection`: 3 test(s) -> `test_feature_selection_per_fold`, `test_stable_features_selected_most_folds`, `test_compute_stable_features_fallback`
  - `TestCalibrationValidationSplit`: 5 test(s) -> `test_calibration_uses_separate_split`, `test_ece_computed`, `test_ece_perfect_calibration`, `test_ece_worst_calibration`, `test_reliability_curve`
  - `TestRegimeMinSamples`: 3 test(s) -> `test_regime_model_min_samples`, `test_regime_model_skipped_for_low_samples`, `test_regime_fallback_to_global`
  - `TestCorrelationPruning`: 3 test(s) -> `test_correlation_threshold_080`, `test_old_threshold_would_keep_correlated`, `test_default_threshold_is_080`
  - `TestGovernanceScoring`: 4 test(s) -> `test_governance_score_includes_validation`, `test_governance_score_backward_compatible`, `test_governance_promotion_with_validation`, `test_dsr_penalty_reduces_score`

### `tests/test_truth_layer.py`
- Intent: Tests for Spec 01: Foundational Hardening — Truth Layer.
- LOC: 872
- Test classes:
  - `TestPreconditionsContract`: 14 test(s) -> `test_validate_execution_contract_passes_default_config`, `test_preconditions_config_rejects_zero_label_h`, `test_preconditions_config_rejects_negative_label_h`, `test_preconditions_config_rejects_excessive_label_h`, `test_preconditions_config_accepts_valid_values`, `test_preconditions_config_coerces_strings`, `test_preconditions_config_rejects_invalid_ret_type`, `test_preconditions_config_rejects_invalid_px_type`, `test_preconditions_config_rejects_invalid_entry_type`, `test_enforce_preconditions_respects_feature_flag`, `test_enforce_preconditions_raises_on_invalid_config`, `test_return_type_enum_values` (+2 more)
  - `TestDataIntegrity`: 7 test(s) -> `test_assess_quality_fail_on_error_raises`, `test_assess_quality_fail_on_error_false_returns_report`, `test_assess_quality_passes_clean_data`, `test_data_integrity_validator_passes_clean_universe`, `test_data_integrity_validator_blocks_corrupt_data_fail_fast`, `test_data_integrity_validator_no_fail_fast_collects_all`, `test_data_integrity_empty_universe`
  - `TestLeakageTripwires`: 8 test(s) -> `test_leakage_detector_catches_forward_shift`, `test_leakage_detector_passes_clean_features`, `test_leakage_detector_multiple_shifts`, `test_run_leakage_checks_raises_on_leakage`, `test_run_leakage_checks_passes_clean`, `test_causality_enforcement_rejects_research_only`, `test_causality_enforcement_end_of_day_features`, `test_leakage_detector_handles_empty_index`
  - `TestNullModels`: 10 test(s) -> `test_random_baseline_generates_signals`, `test_random_baseline_is_reproducible`, `test_random_baseline_sharpe_near_zero`, `test_zero_baseline_returns_zero`, `test_zero_baseline_generates_zero_signals`, `test_momentum_baseline_generates_signals`, `test_momentum_baseline_computes_metrics`, `test_compute_null_baselines_returns_all_three`, `test_null_model_results_summary`, `test_backtest_result_summarize_vs_null`
  - `TestCostStress`: 7 test(s) -> `test_cost_stress_basic_sweep`, `test_cost_stress_1x_matches_base`, `test_cost_stress_higher_cost_reduces_sharpe`, `test_cost_stress_breakeven_estimation`, `test_cost_stress_report_format`, `test_cost_stress_to_dict`, `test_cost_stress_empty_returns`
  - `TestCacheStaleness`: 8 test(s) -> `test_get_last_trading_day_returns_timestamp`, `test_get_last_trading_day_weekday`, `test_get_last_trading_day_weekend`, `test_trading_days_between_same_day`, `test_trading_days_between_weekdays`, `test_trading_days_between_across_weekend`, `test_cache_usable_respects_trading_calendar`, `test_cache_stale_after_many_trading_days`
  - `TestConfigIntegration`: 6 test(s) -> `test_config_has_execution_contract_constants`, `test_config_has_truth_layer_flags`, `test_config_has_cost_stress_multipliers`, `test_config_structured_has_preconditions`, `test_config_structured_has_cost_stress`, `test_validate_config_still_works`

### `tests/test_validation_and_risk_extensions.py`
- Intent: Test module for validation and risk extensions behavior and regressions.
- LOC: 108
- Test classes:
  - `ValidationAndRiskExtensionTests`: 3 test(s) -> `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`

### `tests/test_zero_errors.py`
- Intent: Integration test: common operations must produce zero ERROR-level log entries.
- LOC: 121
- Top-level tests: `test_config_validation_no_errors`, `test_regime_detection_no_errors`, `test_health_service_no_errors`, `test_data_loading_graceful_degradation`, `test_data_loading_single_ticker_if_cache_exists`
- Test classes:
  - `_ErrorCapture`: 0 test(s)

## `kalshi/tests`

| File | LOC | Test Classes | Test Methods | Intent |
|---|---|---|---|---|
| `kalshi/tests/test_bin_validity.py` | 106 | 1 | 9 | Bin overlap/gap detection test (Instructions I.3). |
| `kalshi/tests/test_distribution.py` | 42 | 1 | 1 | Kalshi test module for distribution behavior and regressions. |
| `kalshi/tests/test_leakage.py` | 47 | 1 | 1 | Kalshi test module for leakage behavior and regressions. |
| `kalshi/tests/test_no_leakage.py` | 118 | 1 | 2 | No-leakage test at panel level (Instructions I.4). |
| `kalshi/tests/test_signature_kat.py` | 142 | 1 | 4 | Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1). |
| `kalshi/tests/test_stale_quotes.py` | 153 | 1 | 10 | Stale quote cutoff test (Instructions I.5). |
| `kalshi/tests/test_threshold_direction.py` | 127 | 1 | 17 | Threshold direction correctness test (Instructions I.2). |
| `kalshi/tests/test_walkforward_purge.py` | 160 | 1 | 4 | Walk-forward purge/embargo test (Instructions I.6). |

### `kalshi/tests/test_bin_validity.py`
- Intent: Bin overlap/gap detection test (Instructions I.3).
- LOC: 106
- Test classes:
  - `BinValidityTests`: 9 test(s) -> `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`

### `kalshi/tests/test_distribution.py`
- Intent: Kalshi test module for distribution behavior and regressions.
- LOC: 42
- Test classes:
  - `DistributionLocalTests`: 1 test(s) -> `test_bin_distribution_probability_mass_is_normalized`

### `kalshi/tests/test_leakage.py`
- Intent: Kalshi test module for leakage behavior and regressions.
- LOC: 47
- Test classes:
  - `LeakageLocalTests`: 1 test(s) -> `test_feature_rows_strictly_pre_release`

### `kalshi/tests/test_no_leakage.py`
- Intent: No-leakage test at panel level (Instructions I.4).
- LOC: 118
- Test classes:
  - `NoLeakageTests`: 2 test(s) -> `test_all_asof_before_release`, `test_single_event_no_leakage`

### `kalshi/tests/test_signature_kat.py`
- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- LOC: 142
- Test classes:
  - `SignatureKATTests`: 4 test(s) -> `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`

### `kalshi/tests/test_stale_quotes.py`
- Intent: Stale quote cutoff test (Instructions I.5).
- LOC: 153
- Test classes:
  - `StaleQuoteCutoffTests`: 10 test(s) -> `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`

### `kalshi/tests/test_threshold_direction.py`
- Intent: Threshold direction correctness test (Instructions I.2).
- LOC: 127
- Test classes:
  - `ThresholdDirectionTests`: 17 test(s) -> `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below` (+5 more)

### `kalshi/tests/test_walkforward_purge.py`
- Intent: Walk-forward purge/embargo test (Instructions I.6).
- LOC: 160
- Test classes:
  - `WalkForwardPurgeTests`: 4 test(s) -> `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`
