# `kalshi` Package Guide

## Purpose

Kalshi vertical for intraday event-market research.

## Package Summary

- Modules: 25
- Classes: 34
- Top-level functions: 66
- LOC: 6,096

## How This Package Fits Into The System

- Implements the event-market vertical (client, storage, distributions, event features, walk-forward, promotion helpers).
- `run_kalshi_event_pipeline.py` is the main orchestration entrypoint for ingestion and event research outputs.
- No public FastAPI Kalshi router is currently mounted; `api/services/kalshi_service.py` exists but is not routed.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `kalshi/__init__.py` | 58 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 655 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 113 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 935 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 517 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 75 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 127 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 145 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 167 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 178 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 647 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 206 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 142 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 102 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 649 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/tests/__init__.py` | 1 | 0 | 0 | Kalshi package-local tests. |
| `kalshi/tests/test_bin_validity.py` | 105 | 1 | 0 | Bin overlap/gap detection test (Instructions I.3). |
| `kalshi/tests/test_distribution.py` | 41 | 1 | 0 | Kalshi test module for distribution behavior and regressions. |
| `kalshi/tests/test_leakage.py` | 46 | 1 | 0 | Kalshi test module for leakage behavior and regressions. |
| `kalshi/tests/test_no_leakage.py` | 117 | 1 | 0 | No-leakage test at panel level (Instructions I.4). |
| `kalshi/tests/test_signature_kat.py` | 141 | 1 | 0 | Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1). |
| `kalshi/tests/test_stale_quotes.py` | 152 | 1 | 0 | Stale quote cutoff test (Instructions I.5). |
| `kalshi/tests/test_threshold_direction.py` | 126 | 1 | 0 | Threshold direction correctness test (Instructions I.2). |
| `kalshi/tests/test_walkforward_purge.py` | 159 | 1 | 0 | Walk-forward purge/embargo test (Instructions I.6). |
| `kalshi/walkforward.py` | 492 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

## Module Details

### `kalshi/__init__.py`
- Intent: Kalshi vertical for intraday event-market research.
- Classes: none
- Top-level functions: none

### `kalshi/client.py`
- Intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Classes:
  - `RetryPolicy`: HTTP retry settings for Kalshi API requests.
  - `RateLimitPolicy`: Token-bucket rate-limit settings for Kalshi API access.
  - `RequestLimiter`: Lightweight token-bucket limiter with runtime limit updates.
    - Methods: `acquire`, `update_rate`, `update_from_account_limits`
  - `KalshiSigner`: Signs Kalshi requests using RSA-PSS SHA256.
    - Methods: `available`, `sign`
  - `KalshiClient`: Kalshi HTTP wrapper with:
    - Methods: `available`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`
- Top-level functions: `_normalize_env`

### `kalshi/disagreement.py`
- Intent: Cross-market disagreement engine for Kalshi event features.
- Classes:
  - `DisagreementSignals`: Container for cross-market disagreement features derived from Kalshi and options references.
- Top-level functions: `compute_disagreement`, `disagreement_as_feature_dict`

### `kalshi/distribution.py`
- Intent: Contract -> probability distribution builder for Kalshi markets.
- Classes:
  - `DistributionConfig`: Configuration for Kalshi contract-to-distribution reconstruction and snapshot feature extraction.
  - `DirectionResult`: Result of threshold direction resolution with confidence metadata.
  - `BinValidationResult`: Result of bin overlap/gap/ordering validation.
- Top-level functions: `_is_tz_aware_datetime`, `_to_utc_timestamp`, `_prob_from_mid`, `_entropy`, `_isotonic_nonincreasing`, `_isotonic_nondecreasing`, `_resolve_threshold_direction`, `_resolve_threshold_direction_with_confidence`, `_validate_bins`, `_tail_thresholds`, `_latest_quotes_asof`, `_normalize_mass`, `_moments`, `_cdf_from_pmf`, `_pmf_on_grid`, `_distribution_distances` (+7 more)

### `kalshi/events.py`
- Intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Classes:
  - `EventTimestampMeta`: Authoritative event timestamp metadata (D2).
  - `EventFeatureConfig`: Configuration for event snapshot horizons and event-panel quality filtering.
- Top-level functions: `_to_utc_ts`, `_ensure_asof_before_release`, `asof_join`, `build_event_snapshot_grid`, `_merge_event_market_map`, `_add_revision_speed_features`, `add_reference_disagreement_features`, `build_event_feature_panel`, `build_asset_time_feature_panel`, `build_event_labels`, `build_asset_response_labels`

### `kalshi/mapping_store.py`
- Intent: Versioned event-to-market mapping persistence.
- Classes:
  - `EventMarketMappingRecord`: Versioned mapping row linking a macro event to a Kalshi market over an effective time window.
  - `EventMarketMappingStore`: Persistence helper for versioned event-to-market mappings stored in EventTimeStore.
    - Methods: `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`
- Top-level functions: none

### `kalshi/microstructure.py`
- Intent: Market microstructure diagnostics for Kalshi event markets.
- Classes:
  - `MicrostructureDiagnostics`: Microstructure summary metrics computed from a quote panel window.
- Top-level functions: `compute_microstructure`, `microstructure_as_feature_dict`

### `kalshi/options.py`
- Intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Classes: none
- Top-level functions: `_to_utc_ts`, `build_options_reference_panel`, `add_options_disagreement_features`

### `kalshi/pipeline.py`
- Intent: Orchestration helpers for the Kalshi event-market vertical.
- Classes:
  - `KalshiPipeline`: High-level orchestration wrapper for Kalshi sync, feature, walk-forward, and promotion workflows.
    - Methods: `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`
- Top-level functions: none

### `kalshi/promotion.py`
- Intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Classes:
  - `EventPromotionConfig`: Strategy metadata used to evaluate an event strategy through the shared promotion gate.
- Top-level functions: `_to_backtest_result`, `evaluate_event_promotion`

### `kalshi/provider.py`
- Intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Classes:
  - `KalshiProvider`: Provider interface similar to WRDSProvider, but for event-market data.
    - Methods: `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes` (+4 more)
- Top-level functions: `_to_iso_utc`, `_safe_hash_text`, `_asof_date`

### `kalshi/quality.py`
- Intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Classes:
  - `QualityDimensions`: Component-level quality metrics for a Kalshi distribution snapshot.
  - `StalePolicy`: Parameters controlling dynamic stale-quote cutoff schedules for Kalshi snapshots.
- Top-level functions: `_finite`, `dynamic_stale_cutoff_minutes`, `compute_quality_dimensions`, `passes_hard_gates`, `quality_as_feature_dict`

### `kalshi/regimes.py`
- Intent: Regime tagging for Kalshi event strategies.
- Classes:
  - `EventRegimeTag`: Macro regime labels attached to an event for regime-stability analysis.
- Top-level functions: `classify_inflation_regime`, `classify_policy_regime`, `classify_vol_regime`, `tag_event_regime`, `evaluate_strategy_by_regime`, `regime_stability_score`

### `kalshi/router.py`
- Intent: Routing helpers for live vs historical Kalshi endpoints.
- Classes:
  - `RouteDecision`: Resolved endpoint route decision (base URL, path, and historical/live choice).
  - `KalshiDataRouter`: Chooses live vs historical endpoint roots by cutoff timestamp.
    - Methods: `update_cutoff`, `resolve`
- Top-level functions: none

### `kalshi/storage.py`
- Intent: Event-time storage layer for Kalshi + macro event research.
- Classes:
  - `EventTimeStore`: Intraday/event-time storage with a stable schema.
    - Methods: `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs` (+8 more)
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
- Intent: Kalshi test module for distribution behavior and regressions.
- Classes:
  - `DistributionLocalTests`: Test cases covering Kalshi subsystem behavior and safety constraints.
    - Methods: `test_bin_distribution_probability_mass_is_normalized`
- Top-level functions: none

### `kalshi/tests/test_leakage.py`
- Intent: Kalshi test module for leakage behavior and regressions.
- Classes:
  - `LeakageLocalTests`: Test cases covering Kalshi subsystem behavior and safety constraints.
    - Methods: `test_feature_rows_strictly_pre_release`
- Top-level functions: none

### `kalshi/tests/test_no_leakage.py`
- Intent: No-leakage test at panel level (Instructions I.4).
- Classes:
  - `NoLeakageTests`: Panel-level look-ahead bias detection.
    - Methods: `test_all_asof_before_release`, `test_single_event_no_leakage`
- Top-level functions: none

### `kalshi/tests/test_signature_kat.py`
- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Classes:
  - `SignatureKATTests`: Known-answer tests for Kalshi request signing.
    - Methods: `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`
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
    - Methods: `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below` (+5 more)
- Top-level functions: none

### `kalshi/tests/test_walkforward_purge.py`
- Intent: Walk-forward purge/embargo test (Instructions I.6).
- Classes:
  - `WalkForwardPurgeTests`: Tests that walk-forward purge/embargo prevents data leakage.
    - Methods: `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`
- Top-level functions: none

### `kalshi/walkforward.py`
- Intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Classes:
  - `EventWalkForwardConfig`: Configuration for event-level walk-forward splits, purge/embargo, and trial accounting.
  - `EventWalkForwardFold`: Per-fold event walk-forward metrics for fit quality and event-return diagnostics.
  - `EventWalkForwardResult`: Aggregate event walk-forward outputs and OOS traces used in promotion checks.
    - Methods: `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`
- Top-level functions: `_bootstrap_mean_ci`, `_event_regime_stability`, `evaluate_event_contract_metrics`, `_corr`, `_fit_ridge`, `_predict`, `_prepare_panel`, `run_event_walkforward`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
