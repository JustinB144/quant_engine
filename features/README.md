# `features` Package Guide

## Purpose

Feature engineering pipeline and supporting factor families.

## Package Summary

- Modules: 10
- Classes: 5
- Top-level functions: 48
- LOC: 4,026

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `features/__init__.py` | 5 | 0 | 0 | Feature engineering package namespace. |
| `features/harx_spillovers.py` | 243 | 0 | 3 | HARX Volatility Spillover features (Tier 6.1). |
| `features/intraday.py` | 244 | 0 | 2 | Intraday microstructure features from WRDS TAQmsec tick data. |
| `features/lob_features.py` | 312 | 0 | 5 | Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2). |
| `features/macro.py` | 245 | 1 | 1 | FRED macro indicator features for quant_engine. |
| `features/options_factors.py` | 135 | 0 | 4 | Option surface factor construction from OptionMetrics-enriched daily panels. |
| `features/pipeline.py` | 1542 | 1 | 13 | Feature Pipeline — computes model features from OHLCV data. |
| `features/research_factors.py` | 986 | 1 | 19 | Research-derived factor construction for quant_engine. |
| `features/version.py` | 169 | 2 | 0 | Feature versioning system. |
| `features/wave_flow.py` | 145 | 0 | 1 | Wave-Flow Decomposition for quant_engine. |

## Module Details

### `features/__init__.py`
- Intent: Feature engineering package namespace.
- Classes: none
- Top-level functions: none

### `features/harx_spillovers.py`
- Intent: HARX Volatility Spillover features (Tier 6.1).
- Classes: none
- Top-level functions: `_realized_volatility`, `_ols_lstsq`, `compute_harx_spillovers`

### `features/intraday.py`
- Intent: Intraday microstructure features from WRDS TAQmsec tick data.
- Classes: none
- Top-level functions: `compute_intraday_features`, `compute_rolling_vwap`

### `features/lob_features.py`
- Intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- Classes: none
- Top-level functions: `_inter_bar_durations`, `_estimate_poisson_lambda`, `_signed_volume`, `compute_lob_features`, `compute_lob_features_batch`

### `features/macro.py`
- Intent: FRED macro indicator features for quant_engine.
- Classes:
  - `MacroFeatureProvider` (methods: `get_macro_features`)
- Top-level functions: `_cache_key`

### `features/options_factors.py`
- Intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- Classes: none
- Top-level functions: `_pick_numeric`, `_rolling_percentile_rank`, `compute_option_surface_factors`, `compute_iv_shock_features`

### `features/pipeline.py`
- Intent: Feature Pipeline — computes model features from OHLCV data.
- Classes:
  - `FeaturePipeline` (methods: `compute`, `compute_universe`)
- Top-level functions: `get_feature_type`, `_filter_causal_features`, `_build_indicator_set`, `_build_minimal_indicator_set`, `_get_indicators`, `compute_indicator_features`, `compute_raw_features`, `compute_har_volatility_features`, `compute_multiscale_features`, `compute_structural_features`, `compute_interaction_features`, `compute_targets`, `_winsorize_expanding`

### `features/research_factors.py`
- Intent: Research-derived factor construction for quant_engine.
- Classes:
  - `ResearchFactorConfig` (methods: none)
- Top-level functions: `_rolling_zscore`, `_safe_pct_change`, `_required_ohlcv`, `compute_order_flow_impact_factors`, `compute_markov_queue_features`, `compute_time_series_momentum_factors`, `compute_vol_scaled_momentum`, `_rolling_levy_area`, `compute_signature_path_features`, `compute_vol_surface_factors`, `compute_single_asset_research_factors`, `_standardize_block`, `_lagged_weight_matrix`, `compute_cross_asset_research_factors`, `_dtw_distance_numpy`, `_dtw_avg_lag_from_path`, `compute_dtw_lead_lag`, `_numpy_order2_signature`, `compute_path_signatures`

### `features/version.py`
- Intent: Feature versioning system.
- Classes:
  - `FeatureVersion` (methods: `n_features`, `compute_hash`, `to_dict`, `diff`, `is_compatible`)
  - `FeatureRegistry` (methods: `register`, `get_version`, `get_latest`, `list_versions`, `check_compatibility`)
- Top-level functions: none

### `features/wave_flow.py`
- Intent: Wave-Flow Decomposition for quant_engine.
- Classes: none
- Top-level functions: `compute_wave_flow_decomposition`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
