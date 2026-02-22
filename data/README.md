# `data` Package Guide

## Purpose

Provider abstraction, caching, quality controls, and survivorship-safe historical loading.

## Package Summary

- Modules: 10
- Classes: 13
- Top-level functions: 55
- LOC: 5,164

## How This Package Fits Into the System

- Feeds `features`, `models`, `backtest`, `autopilot`, and `dash_ui` data explorers
- Integrates WRDS/local cache/fallback providers
- Enforces quality/provenance and survivorship behavior

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `data/__init__.py` | 13 | 0 | 0 | Data subpackage — self-contained data loading, caching, WRDS, and survivorship. |
| `data/alternative.py` | 652 | 1 | 2 | Alternative data framework — WRDS-backed implementation. |
| `data/feature_store.py` | 308 | 1 | 0 | Point-in-time feature store for backtest acceleration. |
| `data/loader.py` | 707 | 0 | 13 | Data loader — self-contained data loading with multiple sources. |
| `data/local_cache.py` | 674 | 0 | 21 | Local data cache for daily OHLCV data. |
| `data/provider_base.py` | 12 | 1 | 0 | Shared provider protocol for pluggable data connectors. |
| `data/provider_registry.py` | 49 | 0 | 5 | Provider registry for unified data-provider access (WRDS, Kalshi, ...). |
| `data/quality.py` | 237 | 1 | 3 | Data quality checks for OHLCV time series. |
| `data/survivorship.py` | 928 | 8 | 5 | Survivorship Bias Controls (Tasks 112-117) |
| `data/wrds_provider.py` | 1584 | 1 | 6 | wrds_provider.py |

## Module Details

### `data/__init__.py`
- Intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- Classes: none
- Top-level functions: none

### `data/alternative.py`
- Intent: Alternative data framework — WRDS-backed implementation.
- Classes:
  - `AlternativeDataProvider`: WRDS-backed alternative data provider.
    - Methods: `__init__`, `_resolve_permno`, `get_earnings_surprise`, `get_options_flow`, `get_short_interest`, `get_insider_transactions`, `get_institutional_ownership`
- Top-level functions: `_get_wrds`, `compute_alternative_features`

### `data/feature_store.py`
- Intent: Point-in-time feature store for backtest acceleration.
- Classes:
  - `FeatureStore`: Point-in-time feature store for backtest acceleration.
    - Methods: `__init__`, `_version_dir`, `_ts_tag`, `_parquet_path`, `_meta_path`, `save_features`, `load_features`, `list_available`, `invalidate`
- Top-level functions: none

### `data/loader.py`
- Intent: Data loader — self-contained data loading with multiple sources.
- Classes: none
- Top-level functions: `_permno_from_meta`, `_ticker_from_meta`, `_attach_id_attrs`, `_cache_source`, `_cache_is_usable`, `_cached_universe_subset`, `_normalize_ohlcv`, `_harmonize_return_columns`, `_merge_option_surface_from_prefetch`, `load_ohlcv`, `load_universe`, `load_survivorship_universe`, `load_with_delistings`

### `data/local_cache.py`
- Intent: Local data cache for daily OHLCV data.
- Classes: none
- Top-level functions: `_ensure_cache_dir`, `_normalize_ohlcv_columns`, `_to_daily_ohlcv`, `_read_csv_ohlcv`, `_candidate_csv_paths`, `_cache_meta_path`, `_read_cache_meta`, `_write_cache_meta`, `save_ohlcv`, `load_ohlcv_with_meta`, `load_ohlcv`, `load_intraday_ohlcv`, `list_intraday_timeframes`, `list_cached_tickers`, `_daily_cache_files`, `_ticker_from_cache_path`, `_timeframe_from_cache_path`, `_all_cache_files`, `rehydrate_cache_metadata`, `load_ibkr_data`, `cache_universe`

### `data/provider_base.py`
- Intent: Shared provider protocol for pluggable data connectors.
- Classes:
  - `DataProvider`: No class docstring.
    - Methods: `available`
- Top-level functions: none

### `data/provider_registry.py`
- Intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- Classes: none
- Top-level functions: `_wrds_factory`, `_kalshi_factory`, `get_provider`, `list_providers`, `register_provider`

### `data/quality.py`
- Intent: Data quality checks for OHLCV time series.
- Classes:
  - `DataQualityReport`: No class docstring.
    - Methods: `to_dict`
- Top-level functions: `assess_ohlcv_quality`, `generate_quality_report`, `flag_degraded_stocks`

### `data/survivorship.py`
- Intent: Survivorship Bias Controls (Tasks 112-117)
- Classes:
  - `DelistingReason`: Reason for stock delisting.
  - `UniverseMember`: Task 112: Track a symbol's membership in a universe.
    - Methods: `is_active_on`, `to_dict`
  - `UniverseChange`: Task 114: Track a change to universe membership.
    - Methods: `to_dict`
  - `DelistingEvent`: Task 113: Track delisting event with proper returns.
    - Methods: `to_dict`
  - `SurvivorshipReport`: Task 117: Report comparing returns with/without survivorship adjustment.
    - Methods: `to_dict`
  - `UniverseHistoryTracker`: Task 112, 114, 115: Track historical universe membership.
    - Methods: `__init__`, `_init_db`, `add_member`, `record_change`, `get_universe_on_date`, `get_changes_in_period`, `bulk_load_universe`, `clear_universe`
  - `DelistingHandler`: Task 113, 116: Handle delisting events properly.
    - Methods: `__init__`, `_init_db`, `record_delisting`, `preserve_price_history`, `get_dead_company_prices`, `get_delisting_event`, `get_delisting_return`, `is_delisted`, `get_all_delisted_symbols`
  - `SurvivorshipBiasController`: Task 117: Main controller for survivorship bias analysis.
    - Methods: `__init__`, `get_survivorship_free_universe`, `calculate_bias_impact`, `format_report`
- Top-level functions: `hydrate_universe_history_from_snapshots`, `hydrate_sp500_history_from_wrds`, `filter_panel_by_point_in_time_universe`, `reconstruct_historical_universe`, `calculate_survivorship_bias_impact`

### `data/wrds_provider.py`
- Intent: wrds_provider.py
- Classes:
  - `WRDSProvider`: WRDS data provider for the auto-discovery pipeline.
    - Methods: `__init__`, `available`, `_query`, `_query_silent`, `get_sp500_universe`, `get_sp500_history`, `resolve_permno`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_optionmetrics_link`, `_nearest_iv`, `get_option_surface_features`, `get_fundamentals`, `get_earnings_surprises`, `get_institutional_ownership`, `get_taqmsec_ohlcv`, `query_options_volume`, `query_short_interest`, `query_insider_transactions`, `_permno_to_ticker`, `get_ohlcv`
- Top-level functions: `_sanitize_ticker_list`, `_sanitize_permno_list`, `_read_pgpass_password`, `_get_connection`, `get_wrds_provider`, `wrds_available`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
