# `tests` Package Guide

## Purpose

Behavioral specification suite covering core pipeline correctness, leakage guards, execution realism, and subsystem contracts.

## Package Summary

- Modules: 20
- Classes: 29
- Top-level functions: 9
- LOC: 2,572

## How This Package Fits Into the System

- Protects behavioral contracts across all core subsystems
- Should be updated when intentional behavior changes are made

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `tests/__init__.py` | 2 | 0 | 0 | No module docstring. |
| `tests/test_autopilot_predictor_fallback.py` | 53 | 1 | 0 | No module docstring. |
| `tests/test_cache_metadata_rehydrate.py` | 91 | 1 | 1 | No module docstring. |
| `tests/test_covariance_estimator.py` | 20 | 1 | 0 | No module docstring. |
| `tests/test_delisting_total_return.py` | 71 | 1 | 0 | No module docstring. |
| `tests/test_drawdown_liquidation.py` | 128 | 6 | 0 | No module docstring. |
| `tests/test_execution_dynamic_costs.py` | 43 | 1 | 0 | No module docstring. |
| `tests/test_integration.py` | 556 | 4 | 1 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 33 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_asof_features.py` | 60 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_distribution.py` | 109 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_hardening.py` | 600 | 1 | 0 | No module docstring. |
| `tests/test_loader_and_predictor.py` | 220 | 3 | 0 | No module docstring. |
| `tests/test_panel_split.py` | 50 | 1 | 0 | No module docstring. |
| `tests/test_paper_trader_kelly.py` | 120 | 1 | 3 | No module docstring. |
| `tests/test_promotion_contract.py` | 95 | 1 | 2 | No module docstring. |
| `tests/test_provider_registry.py` | 24 | 1 | 0 | No module docstring. |
| `tests/test_research_factors.py` | 127 | 1 | 1 | No module docstring. |
| `tests/test_survivorship_pit.py` | 69 | 1 | 0 | No module docstring. |
| `tests/test_validation_and_risk_extensions.py` | 101 | 1 | 1 | No module docstring. |

## Module Details

### `tests/__init__.py`
- Intent: No module docstring; infer from symbol names below.
- Classes: none
- Top-level functions: none

### `tests/test_autopilot_predictor_fallback.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `AutopilotPredictorFallbackTests`: No class docstring.
    - Methods: `test_ensure_predictor_falls_back_when_model_import_fails`
- Top-level functions: none

### `tests/test_cache_metadata_rehydrate.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `CacheMetadataRehydrateTests`: No class docstring.
    - Methods: `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`
- Top-level functions: `_write_daily_csv`

### `tests/test_covariance_estimator.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `CovarianceEstimatorTests`: No class docstring.
    - Methods: `test_single_asset_covariance_is_2d_and_positive`
- Top-level functions: none

### `tests/test_delisting_total_return.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `DelistingTotalReturnTests`: No class docstring.
    - Methods: `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`
- Top-level functions: none

### `tests/test_drawdown_liquidation.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `_FakePositionSizer`: No class docstring.
    - Methods: `size_position`
  - `_FakeDrawdownController`: No class docstring.
    - Methods: `__init__`, `update`, `get_summary`
  - `_FakeStopLossManager`: No class docstring.
    - Methods: `evaluate`
  - `_FakePortfolioRisk`: No class docstring.
    - Methods: `check_new_position`
  - `_FakeRiskMetrics`: No class docstring.
    - Methods: `compute_full_report`
  - `DrawdownLiquidationTests`: No class docstring.
    - Methods: `test_critical_drawdown_forces_liquidation`
- Top-level functions: none

### `tests/test_execution_dynamic_costs.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `ExecutionDynamicCostTests`: No class docstring.
    - Methods: `test_dynamic_costs_increase_under_stress`
- Top-level functions: none

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
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `ArbitrageFreeSVIBuilderTests`: No class docstring.
    - Methods: `test_build_surface_has_valid_shape_and_monotone_total_variance`
- Top-level functions: none

### `tests/test_kalshi_asof_features.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `KalshiAsofFeatureTests`: No class docstring.
    - Methods: `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`
- Top-level functions: none

### `tests/test_kalshi_distribution.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `KalshiDistributionTests`: No class docstring.
    - Methods: `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`
- Top-level functions: none

### `tests/test_kalshi_hardening.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `KalshiHardeningTests`: No class docstring.
    - Methods: `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof`, `test_mapping_store_asof`, `test_store_ingestion_and_health_tables`, `test_provider_materializes_daily_health_report`, `test_signer_canonical_payload_and_header_fields`
- Top-level functions: none

### `tests/test_loader_and_predictor.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `_FakeWRDSProvider`: No class docstring.
    - Methods: `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `resolve_permno`
  - `_UnavailableWRDSProvider`: No class docstring.
    - Methods: `available`
  - `LoaderAndPredictorTests`: No class docstring.
    - Methods: `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`
- Top-level functions: none

### `tests/test_panel_split.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `PanelSplitTests`: No class docstring.
    - Methods: `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`
- Top-level functions: none

### `tests/test_paper_trader_kelly.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `PaperTraderKellyTests`: No class docstring.
    - Methods: `test_kelly_sizing_changes_position_size_with_bounds`
- Top-level functions: `_mock_price_data`, `_seed_state`, `_run_cycle`

### `tests/test_promotion_contract.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `PromotionContractTests`: No class docstring.
    - Methods: `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`
- Top-level functions: `_candidate`, `_result`

### `tests/test_provider_registry.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `ProviderRegistryTests`: No class docstring.
    - Methods: `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`
- Top-level functions: none

### `tests/test_research_factors.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `ResearchFactorTests`: No class docstring.
    - Methods: `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`
- Top-level functions: `_make_ohlcv`

### `tests/test_survivorship_pit.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `SurvivorshipPointInTimeTests`: No class docstring.
    - Methods: `test_filter_panel_by_point_in_time_universe`
- Top-level functions: none

### `tests/test_validation_and_risk_extensions.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `ValidationAndRiskExtensionTests`: No class docstring.
    - Methods: `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`
- Top-level functions: `_make_ohlcv`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
