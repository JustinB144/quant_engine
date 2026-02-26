# `kalshi` Package Guide

## Purpose

Kalshi event-market ingestion, storage, feature engineering, and evaluation helpers.

## Package Summary

- Modules: 16
- Classes: 26
- Top-level functions: 66
- LOC: 5,224

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `kalshi/__init__.py` | 59 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 656 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 114 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 936 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 518 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 76 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 128 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 146 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 168 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 179 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 648 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 207 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 143 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 103 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 650 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/walkforward.py` | 493 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

## Module Details

### `kalshi/__init__.py`
- Intent: Kalshi vertical for intraday event-market research.
- Classes: none
- Top-level functions: none

### `kalshi/client.py`
- Intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Classes:
  - `RetryPolicy` (methods: none)
  - `RateLimitPolicy` (methods: none)
  - `RequestLimiter` (methods: `acquire`, `update_rate`, `update_from_account_limits`)
  - `KalshiSigner` (methods: `available`, `sign`)
  - `KalshiClient` (methods: `available`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`)
- Top-level functions: `_normalize_env`

### `kalshi/disagreement.py`
- Intent: Cross-market disagreement engine for Kalshi event features.
- Classes:
  - `DisagreementSignals` (methods: none)
- Top-level functions: `compute_disagreement`, `disagreement_as_feature_dict`

### `kalshi/distribution.py`
- Intent: Contract -> probability distribution builder for Kalshi markets.
- Classes:
  - `DistributionConfig` (methods: none)
  - `DirectionResult` (methods: none)
  - `BinValidationResult` (methods: none)
- Top-level functions: `_is_tz_aware_datetime`, `_to_utc_timestamp`, `_prob_from_mid`, `_entropy`, `_isotonic_nonincreasing`, `_isotonic_nondecreasing`, `_resolve_threshold_direction`, `_resolve_threshold_direction_with_confidence`, `_validate_bins`, `_tail_thresholds`, `_latest_quotes_asof`, `_normalize_mass`, `_moments`, `_cdf_from_pmf`, `_pmf_on_grid`, `_distribution_distances`, `_tail_probs_from_mass`, `_tail_probs_from_threshold_curve`, `_estimate_liquidity_proxy`, `build_distribution_snapshot`, `_lag_slug`, `_add_distance_features`, `build_distribution_panel`

### `kalshi/events.py`
- Intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Classes:
  - `EventTimestampMeta` (methods: none)
  - `EventFeatureConfig` (methods: none)
- Top-level functions: `_to_utc_ts`, `_ensure_asof_before_release`, `asof_join`, `build_event_snapshot_grid`, `_merge_event_market_map`, `_add_revision_speed_features`, `add_reference_disagreement_features`, `build_event_feature_panel`, `build_asset_time_feature_panel`, `build_event_labels`, `build_asset_response_labels`

### `kalshi/mapping_store.py`
- Intent: Versioned event-to-market mapping persistence.
- Classes:
  - `EventMarketMappingRecord` (methods: none)
  - `EventMarketMappingStore` (methods: `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`)
- Top-level functions: none

### `kalshi/microstructure.py`
- Intent: Market microstructure diagnostics for Kalshi event markets.
- Classes:
  - `MicrostructureDiagnostics` (methods: none)
- Top-level functions: `compute_microstructure`, `microstructure_as_feature_dict`

### `kalshi/options.py`
- Intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Classes: none
- Top-level functions: `_to_utc_ts`, `build_options_reference_panel`, `add_options_disagreement_features`

### `kalshi/pipeline.py`
- Intent: Orchestration helpers for the Kalshi event-market vertical.
- Classes:
  - `KalshiPipeline` (methods: `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`)
- Top-level functions: none

### `kalshi/promotion.py`
- Intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Classes:
  - `EventPromotionConfig` (methods: none)
- Top-level functions: `_to_backtest_result`, `evaluate_event_promotion`

### `kalshi/provider.py`
- Intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Classes:
  - `KalshiProvider` (methods: `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes` (+4 more))
- Top-level functions: `_to_iso_utc`, `_safe_hash_text`, `_asof_date`

### `kalshi/quality.py`
- Intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Classes:
  - `QualityDimensions` (methods: none)
  - `StalePolicy` (methods: none)
- Top-level functions: `_finite`, `dynamic_stale_cutoff_minutes`, `compute_quality_dimensions`, `passes_hard_gates`, `quality_as_feature_dict`

### `kalshi/regimes.py`
- Intent: Regime tagging for Kalshi event strategies.
- Classes:
  - `EventRegimeTag` (methods: none)
- Top-level functions: `classify_inflation_regime`, `classify_policy_regime`, `classify_vol_regime`, `tag_event_regime`, `evaluate_strategy_by_regime`, `regime_stability_score`

### `kalshi/router.py`
- Intent: Routing helpers for live vs historical Kalshi endpoints.
- Classes:
  - `RouteDecision` (methods: none)
  - `KalshiDataRouter` (methods: `update_cutoff`, `resolve`)
- Top-level functions: none

### `kalshi/storage.py`
- Intent: Event-time storage layer for Kalshi + macro event research.
- Classes:
  - `EventTimeStore` (methods: `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs` (+8 more))
- Top-level functions: none

### `kalshi/walkforward.py`
- Intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Classes:
  - `EventWalkForwardConfig` (methods: none)
  - `EventWalkForwardFold` (methods: none)
  - `EventWalkForwardResult` (methods: `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`)
- Top-level functions: `_bootstrap_mean_ci`, `_event_regime_stability`, `evaluate_event_contract_metrics`, `_corr`, `_fit_ridge`, `_predict`, `_prepare_panel`, `run_event_walkforward`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
- `kalshi/tests/` (package-local Kalshi tests)
