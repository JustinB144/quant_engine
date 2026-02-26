# `data` Package Guide

## Purpose

Data loading, provider integration, caching, quality, and survivorship controls.

## Package Summary

- Modules: 12
- Classes: 17
- Top-level functions: 80
- LOC: 7,099

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `data/__init__.py` | 33 | 0 | 0 | Data subpackage — self-contained data loading, caching, WRDS, and survivorship. |
| `data/alternative.py` | 653 | 1 | 2 | Alternative data framework — WRDS-backed implementation. |
| `data/cross_source_validator.py` | 680 | 2 | 0 | Cross-source validation system comparing Alpaca/Alpha Vantage against IBKR. |
| `data/feature_store.py` | 313 | 1 | 0 | Point-in-time feature store for backtest acceleration. |
| `data/intraday_quality.py` | 967 | 2 | 20 | Comprehensive quality gate for intraday OHLCV data. |
| `data/loader.py` | 832 | 0 | 17 | Data loader — self-contained data loading with multiple sources. |
| `data/local_cache.py` | 703 | 0 | 21 | Local data cache for daily OHLCV data. |
| `data/provider_base.py` | 15 | 1 | 0 | Shared provider protocol for pluggable data connectors. |
| `data/provider_registry.py` | 54 | 0 | 5 | Provider registry for unified data-provider access (WRDS, Kalshi, ...). |
| `data/quality.py` | 297 | 1 | 4 | Data quality checks for OHLCV time series. |
| `data/survivorship.py` | 936 | 8 | 5 | Survivorship Bias Controls (Tasks 112-117) |
| `data/wrds_provider.py` | 1616 | 1 | 6 | wrds_provider.py |

## Module Details

### `data/__init__.py`
- Intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- Classes: none
- Top-level functions: none

### `data/alternative.py`
- Intent: Alternative data framework — WRDS-backed implementation.
- Classes:
  - `AlternativeDataProvider` (methods: `get_earnings_surprise`, `get_options_flow`, `get_short_interest`, `get_insider_transactions`, `get_institutional_ownership`)
- Top-level functions: `_get_wrds`, `compute_alternative_features`

### `data/cross_source_validator.py`
- Intent: Cross-source validation system comparing Alpaca/Alpha Vantage against IBKR.
- Classes:
  - `CrossValidationReport` (methods: none)
  - `CrossSourceValidator` (methods: `validate_ticker`)
- Top-level functions: none

### `data/feature_store.py`
- Intent: Point-in-time feature store for backtest acceleration.
- Classes:
  - `FeatureStore` (methods: `save_features`, `load_features`, `list_available`, `invalidate`)
- Top-level functions: none

### `data/intraday_quality.py`
- Intent: Comprehensive quality gate for intraday OHLCV data.
- Classes:
  - `CheckResult` (methods: none)
  - `IntradayQualityReport` (methods: `add_check`, `compute_quality_score`)
- Top-level functions: `_get_trading_days`, `_is_in_rth`, `_get_expected_bar_count`, `_check_ohlc_consistency`, `_check_non_negative_volume`, `_check_non_negative_prices`, `_check_timestamp_in_rth`, `_check_extreme_bar_return`, `_check_stale_price`, `_check_zero_volume_liquid`, `_check_missing_bar_ratio`, `_check_duplicate_timestamps`, `_check_monotonic_index`, `_check_overnight_gap`, `_check_volume_distribution`, `_check_split_detection`, `quarantine_ticker`, `write_quality_report`, `read_quality_report`, `validate_intraday_bars`

### `data/loader.py`
- Intent: Data loader — self-contained data loading with multiple sources.
- Classes: none
- Top-level functions: `_permno_from_meta`, `_ticker_from_meta`, `_attach_id_attrs`, `_cache_source`, `_get_last_trading_day`, `_trading_days_between`, `_cache_is_usable`, `_cached_universe_subset`, `_normalize_ohlcv`, `_harmonize_return_columns`, `_merge_option_surface_from_prefetch`, `load_ohlcv`, `get_data_provenance`, `get_skip_reasons`, `load_universe`, `load_survivorship_universe`, `load_with_delistings`

### `data/local_cache.py`
- Intent: Local data cache for daily OHLCV data.
- Classes: none
- Top-level functions: `_ensure_cache_dir`, `_normalize_ohlcv_columns`, `_to_daily_ohlcv`, `_read_csv_ohlcv`, `_candidate_csv_paths`, `_cache_meta_path`, `_read_cache_meta`, `_write_cache_meta`, `save_ohlcv`, `load_ohlcv_with_meta`, `load_ohlcv`, `load_intraday_ohlcv`, `list_intraday_timeframes`, `list_cached_tickers`, `_daily_cache_files`, `_ticker_from_cache_path`, `_timeframe_from_cache_path`, `_all_cache_files`, `rehydrate_cache_metadata`, `load_ibkr_data`, `cache_universe`

### `data/provider_base.py`
- Intent: Shared provider protocol for pluggable data connectors.
- Classes:
  - `DataProvider` (methods: `available`)
- Top-level functions: none

### `data/provider_registry.py`
- Intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- Classes: none
- Top-level functions: `_wrds_factory`, `_kalshi_factory`, `get_provider`, `list_providers`, `register_provider`

### `data/quality.py`
- Intent: Data quality checks for OHLCV time series.
- Classes:
  - `DataQualityReport` (methods: `to_dict`)
- Top-level functions: `_expected_trading_days`, `assess_ohlcv_quality`, `generate_quality_report`, `flag_degraded_stocks`

### `data/survivorship.py`
- Intent: Survivorship Bias Controls (Tasks 112-117)
- Classes:
  - `DelistingReason` (methods: none)
  - `UniverseMember` (methods: `is_active_on`, `to_dict`)
  - `UniverseChange` (methods: `to_dict`)
  - `DelistingEvent` (methods: `to_dict`)
  - `SurvivorshipReport` (methods: `to_dict`)
  - `UniverseHistoryTracker` (methods: `add_member`, `record_change`, `get_universe_on_date`, `get_changes_in_period`, `bulk_load_universe`, `clear_universe`)
  - `DelistingHandler` (methods: `record_delisting`, `preserve_price_history`, `get_dead_company_prices`, `get_delisting_event`, `get_delisting_return`, `is_delisted`, `get_all_delisted_symbols`)
  - `SurvivorshipBiasController` (methods: `get_survivorship_free_universe`, `calculate_bias_impact`, `format_report`)
- Top-level functions: `hydrate_universe_history_from_snapshots`, `hydrate_sp500_history_from_wrds`, `filter_panel_by_point_in_time_universe`, `reconstruct_historical_universe`, `calculate_survivorship_bias_impact`

### `data/wrds_provider.py`
- Intent: wrds_provider.py
- Classes:
  - `WRDSProvider` (methods: `available`, `get_sp500_universe`, `get_sp500_history`, `resolve_permno`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_optionmetrics_link`, `get_option_surface_features`, `get_fundamentals`, `get_earnings_surprises`, `get_institutional_ownership`, `get_taqmsec_ohlcv` (+4 more))
- Top-level functions: `_sanitize_ticker_list`, `_sanitize_permno_list`, `_read_pgpass_password`, `_get_connection`, `get_wrds_provider`, `wrds_available`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
