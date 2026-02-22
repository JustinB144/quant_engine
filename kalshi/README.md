# `kalshi` Package Guide

## Purpose

Event-market vertical: signed client, event-time storage, distribution reconstruction, event features, walk-forward evaluation, promotion helpers.

## Package Summary

- Modules: 25
- Classes: 34
- Top-level functions: 66
- LOC: 5,947

## How This Package Fits Into the System

- Uses its own provider/store pipeline and can reuse promotion logic from `autopilot` and validation from `backtest`
- Feeds Dash autopilot/events views and event research outputs

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `kalshi/__init__.py` | 58 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 630 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 112 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 917 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 511 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 70 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 126 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 144 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 158 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 176 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 628 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 203 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 141 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 95 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 623 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/tests/__init__.py` | 1 | 0 | 0 | Kalshi package-local tests. |
| `kalshi/tests/test_bin_validity.py` | 105 | 1 | 0 | Bin overlap/gap detection test (Instructions I.3). |
| `kalshi/tests/test_distribution.py` | 36 | 1 | 0 | No module docstring. |
| `kalshi/tests/test_leakage.py` | 41 | 1 | 0 | No module docstring. |
| `kalshi/tests/test_no_leakage.py` | 117 | 1 | 0 | No-leakage test at panel level (Instructions I.4). |
| `kalshi/tests/test_signature_kat.py` | 141 | 1 | 0 | Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1). |
| `kalshi/tests/test_stale_quotes.py` | 152 | 1 | 0 | Stale quote cutoff test (Instructions I.5). |
| `kalshi/tests/test_threshold_direction.py` | 126 | 1 | 0 | Threshold direction correctness test (Instructions I.2). |
| `kalshi/tests/test_walkforward_purge.py` | 159 | 1 | 0 | Walk-forward purge/embargo test (Instructions I.6). |
| `kalshi/walkforward.py` | 477 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

## Module Details

### `kalshi/__init__.py`
- Intent: Kalshi vertical for intraday event-market research.
- Classes: none
- Top-level functions: none

### `kalshi/client.py`
- Intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Classes:
  - `RetryPolicy`: No class docstring.
  - `RateLimitPolicy`: No class docstring.
  - `RequestLimiter`: Lightweight token-bucket limiter with runtime limit updates.
    - Methods: `__init__`, `_refill`, `acquire`, `update_rate`, `update_from_account_limits`
  - `KalshiSigner`: Signs Kalshi requests using RSA-PSS SHA256.
    - Methods: `__init__`, `available`, `_canonical_path`, `_load_private_key`, `_sign_with_cryptography`, `_sign_with_openssl`, `sign`
  - `KalshiClient`: Kalshi HTTP wrapper with:
    - Methods: `__init__`, `available`, `_join_url`, `_auth_headers`, `_request_with_retries`, `_request`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`
- Top-level functions: `_normalize_env`

### `kalshi/disagreement.py`
- Intent: Cross-market disagreement engine for Kalshi event features.
- Classes:
  - `DisagreementSignals`: No class docstring.
- Top-level functions: `compute_disagreement`, `disagreement_as_feature_dict`

### `kalshi/distribution.py`
- Intent: Contract -> probability distribution builder for Kalshi markets.
- Classes:
  - `DistributionConfig`: No class docstring.
  - `DirectionResult`: Result of threshold direction resolution with confidence metadata.
  - `BinValidationResult`: Result of bin overlap/gap/ordering validation.
- Top-level functions: `_is_tz_aware_datetime`, `_to_utc_timestamp`, `_prob_from_mid`, `_entropy`, `_isotonic_nonincreasing`, `_isotonic_nondecreasing`, `_resolve_threshold_direction`, `_resolve_threshold_direction_with_confidence`, `_validate_bins`, `_tail_thresholds`, `_latest_quotes_asof`, `_normalize_mass`, `_moments`, `_cdf_from_pmf`, `_pmf_on_grid`, `_distribution_distances`, `_tail_probs_from_mass`, `_tail_probs_from_threshold_curve`, `_estimate_liquidity_proxy`, `build_distribution_snapshot`, `_lag_slug`, `_add_distance_features`, `build_distribution_panel`

### `kalshi/events.py`
- Intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Classes:
  - `EventTimestampMeta`: Authoritative event timestamp metadata (D2).
  - `EventFeatureConfig`: No class docstring.
- Top-level functions: `_to_utc_ts`, `_ensure_asof_before_release`, `asof_join`, `build_event_snapshot_grid`, `_merge_event_market_map`, `_add_revision_speed_features`, `add_reference_disagreement_features`, `build_event_feature_panel`, `build_asset_time_feature_panel`, `build_event_labels`, `build_asset_response_labels`

### `kalshi/mapping_store.py`
- Intent: Versioned event-to-market mapping persistence.
- Classes:
  - `EventMarketMappingRecord`: No class docstring.
  - `EventMarketMappingStore`: No class docstring.
    - Methods: `__init__`, `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`
- Top-level functions: none

### `kalshi/microstructure.py`
- Intent: Market microstructure diagnostics for Kalshi event markets.
- Classes:
  - `MicrostructureDiagnostics`: No class docstring.
- Top-level functions: `compute_microstructure`, `microstructure_as_feature_dict`

### `kalshi/options.py`
- Intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Classes: none
- Top-level functions: `_to_utc_ts`, `build_options_reference_panel`, `add_options_disagreement_features`

### `kalshi/pipeline.py`
- Intent: Orchestration helpers for the Kalshi event-market vertical.
- Classes:
  - `KalshiPipeline`: No class docstring.
    - Methods: `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`
- Top-level functions: none

### `kalshi/promotion.py`
- Intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Classes:
  - `EventPromotionConfig`: No class docstring.
- Top-level functions: `_to_backtest_result`, `evaluate_event_promotion`

### `kalshi/provider.py`
- Intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Classes:
  - `KalshiProvider`: Provider interface similar to WRDSProvider, but for event-market data.
    - Methods: `__init__`, `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes`, `compute_and_store_distributions`, `materialize_daily_health_report`, `get_daily_health_report`, `store_clock_check`
- Top-level functions: `_to_iso_utc`, `_safe_hash_text`, `_asof_date`

### `kalshi/quality.py`
- Intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Classes:
  - `QualityDimensions`: No class docstring.
  - `StalePolicy`: No class docstring.
- Top-level functions: `_finite`, `dynamic_stale_cutoff_minutes`, `compute_quality_dimensions`, `passes_hard_gates`, `quality_as_feature_dict`

### `kalshi/regimes.py`
- Intent: Regime tagging for Kalshi event strategies.
- Classes:
  - `EventRegimeTag`: No class docstring.
- Top-level functions: `classify_inflation_regime`, `classify_policy_regime`, `classify_vol_regime`, `tag_event_regime`, `evaluate_strategy_by_regime`, `regime_stability_score`

### `kalshi/router.py`
- Intent: Routing helpers for live vs historical Kalshi endpoints.
- Classes:
  - `RouteDecision`: No class docstring.
  - `KalshiDataRouter`: Chooses live vs historical endpoint roots by cutoff timestamp.
    - Methods: `__init__`, `_to_utc_ts`, `update_cutoff`, `_extract_end_ts`, `_clean_path`, `resolve`
- Top-level functions: none

### `kalshi/storage.py`
- Intent: Event-time storage layer for Kalshi + macro event research.
- Classes:
  - `EventTimeStore`: Intraday/event-time storage with a stable schema.
    - Methods: `__init__`, `_execute`, `_executemany`, `_table_columns`, `_norm_ts`, `_clean_value`, `_insert_or_replace`, `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs`, `upsert_data_provenance`, `upsert_coverage_diagnostics`, `upsert_ingestion_logs`, `upsert_daily_health_reports`, `upsert_ingestion_checkpoints`, `get_ingestion_checkpoint`, `get_event_market_map_asof`, `query_df`
- Top-level functions: none

### `kalshi/tests/__init__.py`
- Intent: Kalshi package-local tests.
- Classes: none
- Top-level functions: none

### `kalshi/tests/test_bin_validity.py`
- Intent: Bin overlap/gap detection test (Instructions I.3).
- Classes:
  - `BinValidityTests`: Tests for bin overlap/gap/ordering validation.
    - Methods: `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`
- Top-level functions: none

### `kalshi/tests/test_distribution.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `DistributionLocalTests`: No class docstring.
    - Methods: `test_bin_distribution_probability_mass_is_normalized`
- Top-level functions: none

### `kalshi/tests/test_leakage.py`
- Intent: No module docstring; infer from symbol names below.
- Classes:
  - `LeakageLocalTests`: No class docstring.
    - Methods: `test_feature_rows_strictly_pre_release`
- Top-level functions: none

### `kalshi/tests/test_no_leakage.py`
- Intent: No-leakage test at panel level (Instructions I.4).
- Classes:
  - `NoLeakageTests`: Panel-level look-ahead bias detection.
    - Methods: `_build_synthetic_panel`, `test_all_asof_before_release`, `test_single_event_no_leakage`
- Top-level functions: none

### `kalshi/tests/test_signature_kat.py`
- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Classes:
  - `SignatureKATTests`: Known-answer tests for Kalshi request signing.
    - Methods: `_skip_if_no_crypto`, `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`
- Top-level functions: none

### `kalshi/tests/test_stale_quotes.py`
- Intent: Stale quote cutoff test (Instructions I.5).
- Classes:
  - `StaleQuoteCutoffTests`: Tests for dynamic stale-cutoff schedule.
    - Methods: `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`
- Top-level functions: none

### `kalshi/tests/test_threshold_direction.py`
- Intent: Threshold direction correctness test (Instructions I.2).
- Classes:
  - `ThresholdDirectionTests`: Tests for threshold direction resolution.
    - Methods: `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below`, `test_title_guess_or_higher`, `test_title_guess_or_lower`, `test_no_direction_signal`, `test_empty_row`, `test_legacy_resolve_returns_string`
- Top-level functions: none

### `kalshi/tests/test_walkforward_purge.py`
- Intent: Walk-forward purge/embargo test (Instructions I.6).
- Classes:
  - `WalkForwardPurgeTests`: Tests that walk-forward purge/embargo prevents data leakage.
    - Methods: `_build_synthetic_data`, `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`
- Top-level functions: none

### `kalshi/walkforward.py`
- Intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Classes:
  - `EventWalkForwardConfig`: No class docstring.
  - `EventWalkForwardFold`: No class docstring.
  - `EventWalkForwardResult`: No class docstring.
    - Methods: `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`
- Top-level functions: `_bootstrap_mean_ci`, `_event_regime_stability`, `evaluate_event_contract_metrics`, `_corr`, `_fit_ridge`, `_predict`, `_prepare_panel`, `run_event_walkforward`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
