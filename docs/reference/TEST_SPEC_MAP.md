# Test Specification Map

Source-derived behavioral map of the current test suite. Treat these tests as executable contracts for expected system behavior.

## Scope Summary

| Scope | Test Files | Test Classes | Test Methods |
|---|---:|---:|---:|
| `api` | 8 | 0 | 55 |
| `autopilot` | 1 | 1 | 1 |
| `data` | 5 | 10 | 24 |
| `features` | 2 | 8 | 17 |
| `integration` | 1 | 4 | 13 |
| `kalshi` | 12 | 11 | 69 |
| `misc` | 11 | 24 | 73 |
| `risk/backtest/validation` | 6 | 17 | 30 |

## `api`

### `tests/api/__init__.py`

- Intent: No module docstring.
- LOC: 0
- Test classes: none

### `tests/api/test_compute_routers.py`

- Intent: Tests for POST compute endpoints — verify job creation.
- LOC: 65
- Test classes: none
- Top-level tests: `test_train_creates_job`, `test_predict_creates_job`, `test_backtest_creates_job`, `test_autopilot_creates_job`, `test_job_status_queryable`, `test_nonexistent_job_404`

### `tests/api/test_envelope.py`

- Intent: Tests for ApiResponse envelope and ResponseMeta.
- LOC: 51
- Test classes: none
- Top-level tests: `test_success_response`, `test_fail_response`, `test_from_cached`, `test_meta_defaults`, `test_meta_custom_fields`, `test_serialization_roundtrip`

### `tests/api/test_integration.py`

- Intent: Integration tests — full app startup, envelope consistency, config patch.
- LOC: 72
- Test classes: none
- Top-level tests: `test_all_gets_return_envelope`, `test_config_patch_and_read`, `test_config_patch_invalid_key`, `test_meta_has_generated_at`

### `tests/api/test_jobs.py`

- Intent: Tests for the job system: store, runner, lifecycle.
- LOC: 135
- Test classes: none
- Top-level tests: `test_create_and_get_job`, `test_list_jobs`, `test_update_status`, `test_update_progress`, `test_cancel_queued_job`, `test_cancel_completed_job_fails`, `test_get_nonexistent_job`, `test_job_runner_succeeds`, `test_job_runner_failure`, `test_sse_events`
- Top-level helpers: `store`, `runner`

### `tests/api/test_main.py`

- Intent: Tests for app factory and basic middleware.
- LOC: 48
- Test classes: none
- Top-level tests: `test_create_app`, `test_openapi_schema`, `test_routes_registered`, `test_404_wrapped`, `test_cors_headers`

### `tests/api/test_read_routers.py`

- Intent: Tests for all GET endpoints — verify ApiResponse envelope.
- LOC: 118
- Test classes: none
- Top-level tests: `test_health`, `test_dashboard_summary`, `test_data_universe`, `test_models_versions`, `test_signals_latest`, `test_backtests_latest`, `test_backtests_trades`, `test_backtests_equity_curve`, `test_autopilot_latest_cycle`, `test_autopilot_strategies`, `test_autopilot_paper_state`, `test_config_get`, `test_logs`, `test_jobs_list_empty`

### `tests/api/test_services.py`

- Intent: Tests for service wrappers — verify dict outputs.
- LOC: 80
- Test classes: none
- Top-level tests: `test_data_service_universe_info`, `test_data_service_cached_tickers`, `test_backtest_service_no_results`, `test_autopilot_service_latest_cycle`, `test_autopilot_service_strategy_registry`, `test_autopilot_service_paper_state`, `test_health_service_quick`, `test_model_service_list_versions`, `test_model_service_champion_info`, `test_results_service_list`

## `autopilot`

### `tests/test_autopilot_predictor_fallback.py`

- Intent: Test module for autopilot predictor fallback behavior and regressions.
- LOC: 58
- Test classes and methods:
  - `AutopilotPredictorFallbackTests`: 1 test method(s) -> `test_ensure_predictor_falls_back_when_model_import_fails`

## `data`

### `tests/test_cache_metadata_rehydrate.py`

- Intent: Test module for cache metadata rehydrate behavior and regressions.
- LOC: 97
- Test classes and methods:
  - `CacheMetadataRehydrateTests`: 3 test method(s) -> `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`
- Top-level helpers: `_write_daily_csv`

### `tests/test_data_diagnostics.py`

- Intent: Tests for data loading diagnostics (Spec 005).
- LOC: 183
- Test classes and methods:
  - `TestOrchestratorImport`: 2 test method(s) -> `test_orchestrator_imports_data_cache_dir`, `test_orchestrator_uses_data_cache_dir_in_diagnostics`
  - `TestLoadUniverseSkipReasons`: 2 test method(s) -> `test_load_universe_logs_skip_reasons`, `test_skip_reasons_include_reason_text`
  - `TestErrorMessageDiagnostics`: 1 test method(s) -> `test_error_message_includes_diagnostics`
  - `TestDataStatusService`: 4 test method(s) -> `test_data_status_returns_summary`, `test_data_status_ticker_entries`, `test_data_status_with_missing_cache_dir`, `test_data_status_freshness_categories`

### `tests/test_loader_and_predictor.py`

- Intent: Test module for loader and predictor behavior and regressions.
- LOC: 230
- Test classes and methods:
  - `_FakeWRDSProvider`: helper/support class (methods: `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_option_surface_features`, `resolve_permno`)
  - `_UnavailableWRDSProvider`: helper/support class (methods: `available`)
  - `LoaderAndPredictorTests`: 8 test method(s) -> `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`

### `tests/test_provider_registry.py`

- Intent: Test module for provider registry behavior and regressions.
- LOC: 29
- Test classes and methods:
  - `ProviderRegistryTests`: 3 test method(s) -> `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`

### `tests/test_survivorship_pit.py`

- Intent: Test module for survivorship pit behavior and regressions.
- LOC: 74
- Test classes and methods:
  - `SurvivorshipPointInTimeTests`: 1 test method(s) -> `test_filter_panel_by_point_in_time_universe`

## `features`

### `tests/test_feature_fixes.py`

- Intent: Tests verifying Spec 012 feature engineering fixes:
- LOC: 377
- Test classes and methods:
  - `TestTSMomBackwardShift`: 2 test method(s) -> `test_tsmom_uses_backward_shift`, `test_tsmom_values_dont_change_when_future_removed`
  - `TestIVShockNonFuture`: 2 test method(s) -> `test_iv_shock_no_future_shift`, `test_iv_shock_no_lookahead`
  - `TestFeatureMetadata`: 3 test method(s) -> `test_feature_metadata_exists`, `test_all_metadata_has_valid_type`, `test_pipeline_features_have_metadata`
  - `TestProductionModeFilter`: 2 test method(s) -> `test_production_mode_filters_research`, `test_production_mode_default_false`
  - `TestRollingVWAPIsCausal`: 2 test method(s) -> `test_rolling_vwap_is_causal`, `test_rolling_vwap_first_rows_nan`
  - `TestMultiHorizonVRP`: 1 test method(s) -> `test_vrp_multi_horizon`
  - `TestFullPipelineLookahead`: 1 test method(s) -> `test_automated_lookahead_detection`
- Top-level helpers: `_make_ohlcv`, `_make_ohlcv_with_iv`

### `tests/test_research_factors.py`

- Intent: Test module for research factors behavior and regressions.
- LOC: 133
- Test classes and methods:
  - `ResearchFactorTests`: 4 test method(s) -> `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`
- Top-level helpers: `_make_ohlcv`

## `integration`

### `tests/test_integration.py`

- Intent: End-to-end integration tests for the quant engine pipeline.
- LOC: 556
- Test classes and methods:
  - `TestFullPipelineSynthetic`: 6 test method(s) -> `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`
  - `TestCvGapHardBlock`: 1 test method(s) -> `test_cv_gap_hard_block`
  - `TestRegime2Suppression`: 2 test method(s) -> `test_regime_2_suppression`, `test_regime_0_not_suppressed`
  - `TestCrossSectionalRanking`: 4 test method(s) -> `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`
- Top-level helpers: `_generate_synthetic_ohlcv`

## `kalshi`

### `kalshi/tests/__init__.py`

- Intent: Kalshi package-local tests.
- LOC: 1
- Test classes: none

### `kalshi/tests/test_bin_validity.py`

- Intent: Bin overlap/gap detection test (Instructions I.3).
- LOC: 105
- Test classes and methods:
  - `BinValidityTests`: 9 test method(s) -> `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`

### `kalshi/tests/test_distribution.py`

- Intent: Kalshi test module for distribution behavior and regressions.
- LOC: 41
- Test classes and methods:
  - `DistributionLocalTests`: 1 test method(s) -> `test_bin_distribution_probability_mass_is_normalized`

### `kalshi/tests/test_leakage.py`

- Intent: Kalshi test module for leakage behavior and regressions.
- LOC: 46
- Test classes and methods:
  - `LeakageLocalTests`: 1 test method(s) -> `test_feature_rows_strictly_pre_release`

### `kalshi/tests/test_no_leakage.py`

- Intent: No-leakage test at panel level (Instructions I.4).
- LOC: 117
- Test classes and methods:
  - `NoLeakageTests`: 2 test method(s) -> `test_all_asof_before_release`, `test_single_event_no_leakage`

### `kalshi/tests/test_signature_kat.py`

- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- LOC: 141
- Test classes and methods:
  - `SignatureKATTests`: 4 test method(s) -> `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`

### `kalshi/tests/test_stale_quotes.py`

- Intent: Stale quote cutoff test (Instructions I.5).
- LOC: 152
- Test classes and methods:
  - `StaleQuoteCutoffTests`: 10 test method(s) -> `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`

### `kalshi/tests/test_threshold_direction.py`

- Intent: Threshold direction correctness test (Instructions I.2).
- LOC: 126
- Test classes and methods:
  - `ThresholdDirectionTests`: 17 test method(s) -> `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below`, `test_title_guess_or_higher`, `test_title_guess_or_lower`, `test_no_direction_signal`, `test_empty_row`, `test_legacy_resolve_returns_string`

### `kalshi/tests/test_walkforward_purge.py`

- Intent: Walk-forward purge/embargo test (Instructions I.6).
- LOC: 159
- Test classes and methods:
  - `WalkForwardPurgeTests`: 4 test method(s) -> `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`

### `tests/test_kalshi_asof_features.py`

- Intent: Test module for kalshi asof features behavior and regressions.
- LOC: 65
- Test classes and methods:
  - `KalshiAsofFeatureTests`: 2 test method(s) -> `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`

### `tests/test_kalshi_distribution.py`

- Intent: Test module for kalshi distribution behavior and regressions.
- LOC: 114
- Test classes and methods:
  - `KalshiDistributionTests`: 3 test method(s) -> `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`

### `tests/test_kalshi_hardening.py`

- Intent: Test module for kalshi hardening behavior and regressions.
- LOC: 605
- Test classes and methods:
  - `KalshiHardeningTests`: 16 test method(s) -> `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof`, `test_mapping_store_asof`, `test_store_ingestion_and_health_tables`, `test_provider_materializes_daily_health_report`, `test_signer_canonical_payload_and_header_fields`

## `misc`

### `tests/__init__.py`

- Intent: Project-level test suite package.
- LOC: 6
- Test classes: none

### `tests/conftest.py`

- Intent: Shared test fixtures for the quant_engine test suite.
- LOC: 240
- Test classes and methods:
  - `_InMemoryJobStore`: helper/support class (methods: `__init__`, `initialize`, `close`, `create_job`, `get_job`, `list_jobs`, `update_status`, `update_progress`, `cancel_job`)
- Top-level helpers: `pytest_sessionfinish`, `synthetic_ohlcv_data`, `synthetic_trades_csv`, `synthetic_model_meta`, `tmp_results_dir`, `tmp_model_dir`, `tmp_data_cache_dir`, `app`, `client`

### `tests/test_conceptual_fixes.py`

- Intent: Tests for Spec 008 — Ensemble & Conceptual Fixes.
- LOC: 324
- Test classes and methods:
  - `TestEnsembleNoPhantomVote`: 2 test method(s) -> `test_two_method_ensemble_does_not_call_jump`, `test_two_method_ensemble_requires_unanimity`
  - `TestEnsembleWithJumpModel`: 2 test method(s) -> `test_three_method_ensemble_calls_all_methods`, `test_three_method_probabilities_sum_to_one`
  - `TestRuleDetectMinDuration`: 3 test method(s) -> `test_no_short_runs_with_min_duration`, `test_min_duration_1_is_noop`, `test_smoothing_preserves_regime_count`
  - `TestConfigEndpointDefaults`: 4 test method(s) -> `test_adjustable_config_includes_backtest_keys`, `test_adjustable_config_values_match_config_py`, `test_config_status_includes_training_section`, `test_config_status_backtest_section_matches`
- Top-level helpers: `_make_synthetic_features`, `_make_features_with_regime_flickers`

### `tests/test_delisting_total_return.py`

- Intent: Test module for delisting total return behavior and regressions.
- LOC: 76
- Test classes and methods:
  - `DelistingTotalReturnTests`: 2 test method(s) -> `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`

### `tests/test_lookahead_detection.py`

- Intent: Automated lookahead bias detection for the feature pipeline.
- LOC: 144
- Test classes and methods:
  - `TestLookaheadDetection`: 2 test method(s) -> `test_no_feature_uses_future_data`, `test_no_feature_uses_future_data_deep`
- Top-level helpers: `_make_synthetic_ohlcv`

### `tests/test_panel_split.py`

- Intent: Test module for panel split behavior and regressions.
- LOC: 55
- Test classes and methods:
  - `PanelSplitTests`: 2 test method(s) -> `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`

### `tests/test_paper_trader_kelly.py`

- Intent: Test module for paper trader kelly behavior and regressions.
- LOC: 128
- Test classes and methods:
  - `PaperTraderKellyTests`: 1 test method(s) -> `test_kelly_sizing_changes_position_size_with_bounds`
- Top-level helpers: `_mock_price_data`, `_seed_state`, `_run_cycle`

### `tests/test_position_sizing_overhaul.py`

- Intent: Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul.
- LOC: 414
- Test classes and methods:
  - `TestKellyNegativeEdge`: 5 test method(s) -> `test_kelly_negative_edge_returns_zero`, `test_kelly_positive_edge_returns_positive`, `test_kelly_invalid_inputs_return_zero`, `test_kelly_small_sample_penalty`, `test_kelly_n_trades_passed_through_size_position`
  - `TestDrawdownGovernor`: 6 test method(s) -> `test_convex_at_50pct_drawdown`, `test_aggressive_at_90pct_drawdown`, `test_no_drawdown_full_sizing`, `test_beyond_max_dd_returns_zero`, `test_positive_drawdown_returns_full`, `test_convex_curve_is_smooth`
  - `TestPerRegimeBayesian`: 3 test method(s) -> `test_bull_kelly_greater_than_hv`, `test_global_fallback_when_few_regime_trades`, `test_per_regime_counters_populated`
  - `TestRegimeStatsUpdate`: 3 test method(s) -> `test_update_from_trade_df`, `test_too_few_trades_keeps_defaults`, `test_multiple_regimes_updated`
  - `TestConfidenceScalar`: 3 test method(s) -> `test_max_confidence_scalar_is_one`, `test_zero_confidence_scalar_is_half`, `test_confidence_never_amplifies`
  - `TestPortfolioCorrelationPenalty`: 3 test method(s) -> `test_high_corr_smaller_than_uncorr`, `test_no_positions_returns_base`, `test_negative_corr_no_penalty`
  - `TestDrawdownControllerIntegration`: 5 test method(s) -> `test_normal_state_allows_entries`, `test_large_loss_triggers_caution`, `test_critical_drawdown_forces_liquidation`, `test_paper_trader_has_drawdown_controller`, `test_recovery_allows_gradual_reentry`
  - `TestSizePositionIntegration`: 2 test method(s) -> `test_negative_edge_signal_still_gets_sized_via_blend`, `test_drawdown_reduces_kelly_in_blend`

### `tests/test_promotion_contract.py`

- Intent: Test module for promotion contract behavior and regressions.
- LOC: 105
- Test classes and methods:
  - `PromotionContractTests`: 2 test method(s) -> `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`
- Top-level helpers: `_candidate`, `_result`

### `tests/test_training_pipeline_fixes.py`

- Intent: Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance.
- LOC: 567
- Test classes and methods:
  - `TestPerFoldFeatureSelection`: 3 test method(s) -> `test_feature_selection_per_fold`, `test_stable_features_selected_most_folds`, `test_compute_stable_features_fallback`
  - `TestCalibrationValidationSplit`: 5 test method(s) -> `test_calibration_uses_separate_split`, `test_ece_computed`, `test_ece_perfect_calibration`, `test_ece_worst_calibration`, `test_reliability_curve`
  - `TestRegimeMinSamples`: 3 test method(s) -> `test_regime_model_min_samples`, `test_regime_model_skipped_for_low_samples`, `test_regime_fallback_to_global`
  - `TestCorrelationPruning`: 3 test method(s) -> `test_correlation_threshold_080`, `test_old_threshold_would_keep_correlated`, `test_default_threshold_is_080`
  - `TestGovernanceScoring`: 4 test method(s) -> `test_governance_score_includes_validation`, `test_governance_score_backward_compatible`, `test_governance_promotion_with_validation`, `test_dsr_penalty_reduces_score`
- Top-level helpers: `_make_feature_matrix`, `_make_targets`, `_make_regimes`

### `tests/test_zero_errors.py`

- Intent: Integration test: common operations must produce zero ERROR-level log entries.
- LOC: 120
- Test classes and methods:
  - `_ErrorCapture`: helper/support class (methods: `__init__`, `emit`)
- Top-level tests: `test_config_validation_no_errors`, `test_regime_detection_no_errors`, `test_health_service_no_errors`, `test_data_loading_graceful_degradation`, `test_data_loading_single_ticker_if_cache_exists`
- Top-level helpers: `error_capture`

## `risk/backtest/validation`

### `tests/test_backtest_realism.py`

- Intent: Spec 011 — Backtest Execution Realism & Validation Enforcement tests.
- LOC: 561
- Test classes and methods:
  - `TestEntryTimingConsistency`: 2 test method(s) -> `test_simple_mode_entry_at_next_bar_open`, `test_risk_managed_mode_entry_at_next_bar_open`
  - `TestAlmgrenChrissCalibration`: 3 test method(s) -> `test_config_risk_aversion_not_risk_neutral`, `test_trajectory_default_matches_config`, `test_higher_risk_aversion_frontloads_execution`
  - `TestExitVolumeConstraints`: 3 test method(s) -> `test_execution_model_limits_fill_ratio`, `test_force_full_bypasses_volume_constraint`, `test_moderate_order_gets_partial_fill`
  - `TestNegativeSharpeFailsDSR`: 4 test method(s) -> `test_negative_sharpe_rejected`, `test_zero_sharpe_rejected`, `test_positive_sharpe_can_pass`, `test_weak_positive_sharpe_with_many_trials_fails`
  - `TestPBOThreshold`: 3 test method(s) -> `test_config_pbo_threshold`, `test_pbo_above_045_is_overfit`, `test_pbo_max_combinations_increased`
  - `TestValidationRequiredForPromotion`: 5 test method(s) -> `test_no_validation_rejects_promotion`, `test_negative_sharpe_always_rejected`, `test_mc_not_significant_rejects_promotion`, `test_all_validations_pass_allows_promotion`, `test_pbo_above_045_rejects_promotion`
  - `TestPartialExitMultiBar`: 3 test method(s) -> `test_backtester_has_residual_tracking`, `test_volume_constrained_exit_records_fill_ratio`, `test_residual_position_exit_concept`
- Top-level helpers: `_make_ohlcv`, `_make_predictions`, `_candidate`, `_passing_result`, `_all_pass_metrics`

### `tests/test_covariance_estimator.py`

- Intent: Test module for covariance estimator behavior and regressions.
- LOC: 25
- Test classes and methods:
  - `CovarianceEstimatorTests`: 1 test method(s) -> `test_single_asset_covariance_is_2d_and_positive`

### `tests/test_drawdown_liquidation.py`

- Intent: Test module for drawdown liquidation behavior and regressions.
- LOC: 138
- Test classes and methods:
  - `_FakePositionSizer`: helper/support class (methods: `size_position`)
  - `_FakeDrawdownController`: helper/support class (methods: `__init__`, `update`, `get_summary`)
  - `_FakeStopLossManager`: helper/support class (methods: `evaluate`)
  - `_FakePortfolioRisk`: helper/support class (methods: `check_new_position`)
  - `_FakeRiskMetrics`: helper/support class (methods: `compute_full_report`)
  - `DrawdownLiquidationTests`: 1 test method(s) -> `test_critical_drawdown_forces_liquidation`

### `tests/test_execution_dynamic_costs.py`

- Intent: Test module for execution dynamic costs behavior and regressions.
- LOC: 48
- Test classes and methods:
  - `ExecutionDynamicCostTests`: 1 test method(s) -> `test_dynamic_costs_increase_under_stress`

### `tests/test_iv_arbitrage_builder.py`

- Intent: Test module for iv arbitrage builder behavior and regressions.
- LOC: 38
- Test classes and methods:
  - `ArbitrageFreeSVIBuilderTests`: 1 test method(s) -> `test_build_surface_has_valid_shape_and_monotone_total_variance`

### `tests/test_validation_and_risk_extensions.py`

- Intent: Test module for validation and risk extensions behavior and regressions.
- LOC: 107
- Test classes and methods:
  - `ValidationAndRiskExtensionTests`: 3 test method(s) -> `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`
- Top-level helpers: `_make_ohlcv`
