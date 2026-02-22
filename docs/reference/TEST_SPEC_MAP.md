# Test Specification Map

This document treats the test suite as a behavioral specification map. It highlights what system behaviors are intentionally protected by tests.

## Scope Summary

| Scope | Test Files | Test Classes | Test Methods |
|---|---:|---:|---:|
| `autopilot` | 1 | 1 | 1 |
| `autopilot/promotion` | 1 | 1 | 2 |
| `data` | 3 | 5 | 14 |
| `end-to-end` | 1 | 4 | 13 |
| `features` | 1 | 1 | 4 |
| `kalshi` | 12 | 11 | 69 |
| `misc` | 4 | 3 | 4 |
| `models/iv` | 2 | 2 | 2 |
| `risk/backtest` | 3 | 8 | 5 |
| `training/validation` | 1 | 1 | 2 |

## `autopilot`

### `tests/test_autopilot_predictor_fallback.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `unittest.mock`, `pandas`, `quant_engine.autopilot.engine`
- Test classes and methods:
  - `AutopilotPredictorFallbackTests`: 1 test method(s) -> `test_ensure_predictor_falls_back_when_model_import_fails`

## `autopilot/promotion`

### `tests/test_promotion_contract.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.autopilot.promotion_gate`, `quant_engine.autopilot.strategy_discovery`, `quant_engine.backtest.engine`
- Test classes and methods:
  - `PromotionContractTests`: 2 test method(s) -> `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`
- Top-level helpers: `_candidate`, `_result`

## `data`

### `tests/test_cache_metadata_rehydrate.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `json`, `tempfile`, `unittest`, `pathlib`, `pandas`, `quant_engine.data.local_cache`
- Test classes and methods:
  - `CacheMetadataRehydrateTests`: 3 test method(s) -> `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`
- Top-level helpers: `_write_daily_csv`

### `tests/test_loader_and_predictor.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `tempfile`, `unittest`, `unittest.mock`, `pandas`, `quant_engine.data.loader`, `quant_engine.data.local_cache`, `quant_engine.data.local_cache`, `quant_engine.data.local_cache`, `quant_engine.models.predictor`
- Test classes and methods:
  - `_FakeWRDSProvider`: 0 test method(s)
    - helper methods: `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `resolve_permno`
  - `_UnavailableWRDSProvider`: 0 test method(s)
    - helper methods: `available`
  - `LoaderAndPredictorTests`: 8 test method(s) -> `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`

### `tests/test_provider_registry.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `quant_engine.data.provider_registry`, `quant_engine.kalshi.client`
- Test classes and methods:
  - `ProviderRegistryTests`: 3 test method(s) -> `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`

## `end-to-end`

### `tests/test_integration.py`
- Intent: End-to-end integration tests for the quant engine pipeline.
- Imports (first): `sys`, `pathlib`, `numpy`, `pandas`, `pytest`
- Test classes and methods:
  - `TestFullPipelineSynthetic`: 6 test method(s) -> `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`
    - helper methods: `synthetic_data`, `pipeline_outputs`
  - `TestCvGapHardBlock`: 1 test method(s) -> `test_cv_gap_hard_block`
  - `TestRegime2Suppression`: 2 test method(s) -> `test_regime_2_suppression`, `test_regime_0_not_suppressed`
  - `TestCrossSectionalRanking`: 4 test method(s) -> `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`
- Top-level helpers: `_generate_synthetic_ohlcv`

## `features`

### `tests/test_research_factors.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.features.pipeline`, `quant_engine.features.research_factors`
- Test classes and methods:
  - `ResearchFactorTests`: 4 test method(s) -> `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`
- Top-level helpers: `_make_ohlcv`

## `kalshi`

### `kalshi/tests/__init__.py`
- Intent: Kalshi package-local tests.
- Test classes: none

### `kalshi/tests/test_bin_validity.py`
- Intent: Bin overlap/gap detection test (Instructions I.3).
- Imports (first): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Test classes and methods:
  - `BinValidityTests`: 9 test method(s) -> `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`

### `kalshi/tests/test_distribution.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Test classes and methods:
  - `DistributionLocalTests`: 1 test method(s) -> `test_bin_distribution_probability_mass_is_normalized`

### `kalshi/tests/test_leakage.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.kalshi.events`
- Test classes and methods:
  - `LeakageLocalTests`: 1 test method(s) -> `test_feature_rows_strictly_pre_release`

### `kalshi/tests/test_no_leakage.py`
- Intent: No-leakage test at panel level (Instructions I.4).
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.kalshi.events`
- Test classes and methods:
  - `NoLeakageTests`: 2 test method(s) -> `test_all_asof_before_release`, `test_single_event_no_leakage`
    - helper methods: `_build_synthetic_panel`

### `kalshi/tests/test_signature_kat.py`
- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Imports (first): `base64`, `unittest`, `quant_engine.kalshi.client`
- Test classes and methods:
  - `SignatureKATTests`: 4 test method(s) -> `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`
    - helper methods: `_skip_if_no_crypto`

### `kalshi/tests/test_stale_quotes.py`
- Intent: Stale quote cutoff test (Instructions I.5).
- Imports (first): `unittest`, `quant_engine.kalshi.quality`
- Test classes and methods:
  - `StaleQuoteCutoffTests`: 10 test method(s) -> `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`

### `kalshi/tests/test_threshold_direction.py`
- Intent: Threshold direction correctness test (Instructions I.2).
- Imports (first): `unittest`, `quant_engine.kalshi.distribution`
- Test classes and methods:
  - `ThresholdDirectionTests`: 17 test method(s) -> `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below`, `test_title_guess_or_higher`, `test_title_guess_or_lower`, `test_no_direction_signal`, `test_empty_row`, `test_legacy_resolve_returns_string`

### `kalshi/tests/test_walkforward_purge.py`
- Intent: Walk-forward purge/embargo test (Instructions I.6).
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.kalshi.walkforward`
- Test classes and methods:
  - `WalkForwardPurgeTests`: 4 test method(s) -> `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`
    - helper methods: `_build_synthetic_data`

### `tests/test_kalshi_asof_features.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.kalshi.events`
- Test classes and methods:
  - `KalshiAsofFeatureTests`: 2 test method(s) -> `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`

### `tests/test_kalshi_distribution.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Test classes and methods:
  - `KalshiDistributionTests`: 3 test method(s) -> `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`

### `tests/test_kalshi_hardening.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `base64`, `tempfile`, `unittest`, `pathlib`, `numpy`, `pandas`, `quant_engine.kalshi.client`, `quant_engine.kalshi.distribution`, `quant_engine.kalshi.events`, `quant_engine.kalshi.mapping_store`, `quant_engine.kalshi.options`, `quant_engine.kalshi.promotion`
- Test classes and methods:
  - `KalshiHardeningTests`: 16 test method(s) -> `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof`, `test_mapping_store_asof`, `test_store_ingestion_and_health_tables`, `test_provider_materializes_daily_health_report`, `test_signer_canonical_payload_and_header_fields`

## `misc`

### `tests/__init__.py`
- Intent: No module docstring; infer from test names below.
- Test classes: none

### `tests/test_delisting_total_return.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.features.pipeline`
- Test classes and methods:
  - `DelistingTotalReturnTests`: 2 test method(s) -> `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`

### `tests/test_execution_dynamic_costs.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `quant_engine.backtest.execution`
- Test classes and methods:
  - `ExecutionDynamicCostTests`: 1 test method(s) -> `test_dynamic_costs_increase_under_stress`

### `tests/test_paper_trader_kelly.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `json`, `tempfile`, `unittest`, `pathlib`, `numpy`, `pandas`, `quant_engine.autopilot.paper_trader`, `quant_engine.autopilot.registry`
- Test classes and methods:
  - `PaperTraderKellyTests`: 1 test method(s) -> `test_kelly_sizing_changes_position_size_with_bounds`
- Top-level helpers: `_mock_price_data`, `_seed_state`, `_run_cycle`

## `models/iv`

### `tests/test_iv_arbitrage_builder.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `numpy`, `quant_engine.models.iv.models`
- Test classes and methods:
  - `ArbitrageFreeSVIBuilderTests`: 1 test method(s) -> `test_build_surface_has_valid_shape_and_monotone_total_variance`

### `tests/test_survivorship_pit.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `tempfile`, `unittest`, `pathlib`, `pandas`, `quant_engine.data.survivorship`
- Test classes and methods:
  - `SurvivorshipPointInTimeTests`: 1 test method(s) -> `test_filter_panel_by_point_in_time_universe`

## `risk/backtest`

### `tests/test_covariance_estimator.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.risk.covariance`
- Test classes and methods:
  - `CovarianceEstimatorTests`: 1 test method(s) -> `test_single_asset_covariance_is_2d_and_positive`

### `tests/test_drawdown_liquidation.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `types`, `pandas`, `quant_engine.backtest.engine`, `quant_engine.risk.drawdown`, `quant_engine.risk.stop_loss`
- Test classes and methods:
  - `_FakePositionSizer`: 0 test method(s)
    - helper methods: `size_position`
  - `_FakeDrawdownController`: 0 test method(s)
    - helper methods: `__init__`, `update`, `get_summary`
  - `_FakeStopLossManager`: 0 test method(s)
    - helper methods: `evaluate`
  - `_FakePortfolioRisk`: 0 test method(s)
    - helper methods: `check_new_position`
  - `_FakeRiskMetrics`: 0 test method(s)
    - helper methods: `compute_full_report`
  - `DrawdownLiquidationTests`: 1 test method(s) -> `test_critical_drawdown_forces_liquidation`

### `tests/test_validation_and_risk_extensions.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `numpy`, `pandas`, `quant_engine.backtest.validation`, `quant_engine.risk.portfolio_risk`
- Test classes and methods:
  - `ValidationAndRiskExtensionTests`: 3 test method(s) -> `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`
- Top-level helpers: `_make_ohlcv`

## `training/validation`

### `tests/test_panel_split.py`
- Intent: No module docstring; infer from test names below.
- Imports (first): `unittest`, `pandas`, `quant_engine.models.trainer`
- Test classes and methods:
  - `PanelSplitTests`: 2 test method(s) -> `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`
