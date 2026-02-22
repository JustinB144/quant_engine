# `features` Package Guide

## Purpose

Feature engineering pipeline and factor families used for training/inference.

## Package Summary

- Modules: 9
- Classes: 3
- Top-level functions: 43
- LOC: 3,047

## How This Package Fits Into the System

- Consumes normalized data from `data` layer
- Feeds `regime`, `models.trainer`, and `models.predictor`
- Provides options/intraday/macro/research factor expansions

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `features/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `features/harx_spillovers.py` | 242 | 0 | 3 | HARX Volatility Spillover features (Tier 6.1). |
| `features/intraday.py` | 193 | 0 | 1 | Intraday microstructure features from WRDS TAQmsec tick data. |
| `features/lob_features.py` | 311 | 0 | 5 | Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2). |
| `features/macro.py` | 243 | 1 | 1 | FRED macro indicator features for quant_engine. |
| `features/options_factors.py` | 119 | 0 | 4 | Option surface factor construction from OptionMetrics-enriched daily panels. |
| `features/pipeline.py` | 819 | 1 | 9 | Feature Pipeline — computes model features from OHLCV data. |
| `features/research_factors.py` | 976 | 1 | 19 | Research-derived factor construction for quant_engine. |
| `features/wave_flow.py` | 144 | 0 | 1 | Wave-Flow Decomposition for quant_engine. |

## Module Details

### `features/__init__.py`
- Intent: No module docstring; infer from symbol names below.
- Classes: none
- Top-level functions: none

### `features/harx_spillovers.py`
- Intent: HARX Volatility Spillover features (Tier 6.1).
- Classes: none
- Top-level functions: `_realized_volatility`, `_ols_lstsq`, `compute_harx_spillovers`

### `features/intraday.py`
- Intent: Intraday microstructure features from WRDS TAQmsec tick data.
- Classes: none
- Top-level functions: `compute_intraday_features`

### `features/lob_features.py`
- Intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- Classes: none
- Top-level functions: `_inter_bar_durations`, `_estimate_poisson_lambda`, `_signed_volume`, `compute_lob_features`, `compute_lob_features_batch`

### `features/macro.py`
- Intent: FRED macro indicator features for quant_engine.
- Classes:
  - `MacroFeatureProvider`: FRED API integration for macro indicator features.
    - Methods: `__init__`, `_fetch_series_fredapi`, `_fetch_series_requests`, `_fetch_series`, `get_macro_features`
- Top-level functions: `_cache_key`

### `features/options_factors.py`
- Intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- Classes: none
- Top-level functions: `_pick_numeric`, `_rolling_percentile_rank`, `compute_option_surface_factors`, `compute_iv_shock_features`

### `features/pipeline.py`
- Intent: Feature Pipeline — computes model features from OHLCV data.
- Classes:
  - `FeaturePipeline`: End-to-end feature computation pipeline.
    - Methods: `__init__`, `compute`, `compute_universe`, `_load_benchmark_close`
- Top-level functions: `_build_indicator_set`, `_get_indicators`, `compute_indicator_features`, `compute_raw_features`, `compute_har_volatility_features`, `compute_multiscale_features`, `compute_interaction_features`, `compute_targets`, `_winsorize_expanding`

### `features/research_factors.py`
- Intent: Research-derived factor construction for quant_engine.
- Classes:
  - `ResearchFactorConfig`: Configuration for research-derived factor generation.
- Top-level functions: `_rolling_zscore`, `_safe_pct_change`, `_required_ohlcv`, `compute_order_flow_impact_factors`, `compute_markov_queue_features`, `compute_time_series_momentum_factors`, `compute_vol_scaled_momentum`, `_rolling_levy_area`, `compute_signature_path_features`, `compute_vol_surface_factors`, `compute_single_asset_research_factors`, `_standardize_block`, `_lagged_weight_matrix`, `compute_cross_asset_research_factors`, `_dtw_distance_numpy`, `_dtw_avg_lag_from_path`, `compute_dtw_lead_lag`, `_numpy_order2_signature`, `compute_path_signatures`

### `features/wave_flow.py`
- Intent: Wave-Flow Decomposition for quant_engine.
- Classes: none
- Top-level functions: `compute_wave_flow_decomposition`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
