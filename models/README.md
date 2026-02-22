# `models` Package Guide

## Purpose

Training, prediction, versioning, governance, retraining triggers, and IV modeling utilities.

## Package Summary

- Modules: 13
- Classes: 26
- Top-level functions: 7
- LOC: 4,397

## How This Package Fits Into the System

- Consumes feature/regime panels
- Feeds predictions to `backtest`, `autopilot`, and UI
- Persists artifacts in `trained_models/` via versioning

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `models/__init__.py` | 20 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 216 | 2 | 0 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/cross_sectional.py` | 136 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 311 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 108 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/iv/__init__.py` | 31 | 0 | 0 | Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface. |
| `models/iv/models.py` | 928 | 10 | 1 | Implied Volatility Surface Models. |
| `models/neural_net.py` | 197 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/predictor.py` | 375 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 296 | 1 | 0 | ML Retraining Trigger Logic |
| `models/trainer.py` | 1340 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 204 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
| `models/walk_forward.py` | 235 | 0 | 4 | Walk-Forward Model Selection — expanding-window hyperparameter search |

## Module Details

### `models/__init__.py`
- Intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- Classes: none
- Top-level functions: none

### `models/calibration.py`
- Intent: Confidence Calibration --- Platt scaling and isotonic regression.
- Classes:
  - `_LinearRescaler`: Maps raw scores to [0, 1] via min-max linear rescaling.
    - Methods: `__init__`, `fit`, `transform`
  - `ConfidenceCalibrator`: Post-hoc confidence calibration via Platt scaling or isotonic regression.
    - Methods: `__init__`, `fit`, `_fit_sklearn`, `transform`, `fit_transform`, `is_fitted`, `backend`, `__repr__`
- Top-level functions: none

### `models/cross_sectional.py`
- Intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- Classes: none
- Top-level functions: `cross_sectional_rank`

### `models/feature_stability.py`
- Intent: Feature Stability Monitoring — tracks feature importance rankings across
- Classes:
  - `StabilityReport`: Summary returned by :meth:`FeatureStabilityTracker.check_stability`.
    - Methods: `to_dict`
  - `FeatureStabilityTracker`: Record and compare feature importance rankings over training cycles.
    - Methods: `__init__`, `_load`, `_save`, `record_importance`, `_spearman_rank_correlation`, `check_stability`
- Top-level functions: none

### `models/governance.py`
- Intent: Champion/challenger governance for model versions.
- Classes:
  - `ChampionRecord`: No class docstring.
    - Methods: `to_dict`
  - `ModelGovernance`: Maintains champion model per horizon and promotes challengers if better.
    - Methods: `__init__`, `_load`, `_save`, `_score`, `get_champion_version`, `evaluate_and_update`
- Top-level functions: none

### `models/iv/__init__.py`
- Intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- Classes: none
- Top-level functions: none

### `models/iv/models.py`
- Intent: Implied Volatility Surface Models.
- Classes:
  - `OptionType`: No class docstring.
  - `Greeks`: Option Greeks container.
  - `HestonParams`: Heston model parameters.
    - Methods: `validate`
  - `SVIParams`: Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).
  - `BlackScholes`: Black-Scholes option pricing and analytics.
    - Methods: `price`, `greeks`, `implied_vol`, `iv_surface`
  - `HestonModel`: Heston (1993) stochastic volatility model.
    - Methods: `__init__`, `characteristic_function`, `price`, `implied_vol`, `iv_surface`, `calibrate`
  - `SVIModel`: SVI (Stochastic Volatility Inspired) implied variance parameterization.
    - Methods: `__init__`, `total_variance`, `implied_vol`, `iv_surface`, `smile`, `calibrate`, `check_no_butterfly_arbitrage`
  - `ArbitrageFreeSVIBuilder`: Arbitrage-aware SVI surface builder.
    - Methods: `__init__`, `_svi_total_variance`, `_initial_guess`, `_vega_spread_weights`, `_slice_objective`, `fit_slice`, `enforce_calendar_monotonicity`, `interpolate_total_variance`, `build_surface`
  - `IVPoint`: Single implied-volatility observation.
  - `IVSurface`: Store and interpolate an implied-volatility surface.
    - Methods: `__init__`, `add_point`, `add_slice`, `add_surface`, `n_points`, `_log_moneyness`, `_build_interpolator`, `get_iv`, `get_smile`, `decompose`, `decompose_surface`
- Top-level functions: `generate_synthetic_market_surface`

### `models/neural_net.py`
- Intent: Tabular Neural Network — feedforward network for tabular financial data.
- Classes:
  - `TabularNet`: Feedforward network for tabular financial data.
    - Methods: `__init__`, `_build_model`, `fit`, `predict`, `feature_importances_`
- Top-level functions: none

### `models/predictor.py`
- Intent: Model Predictor — loads trained ensemble and generates predictions.
- Classes:
  - `EnsemblePredictor`: Loads a trained regime-conditional ensemble and generates predictions.
    - Methods: `__init__`, `_resolve_model_dir`, `_load`, `predict`, `predict_single`
- Top-level functions: `_prepare_features`

### `models/retrain_trigger.py`
- Intent: ML Retraining Trigger Logic
- Classes:
  - `RetrainTrigger`: Determines when ML model should be retrained.
    - Methods: `__init__`, `_load_metadata`, `_save_metadata`, `add_trade_result`, `check`, `record_retraining`, `status`
- Top-level functions: none

### `models/trainer.py`
- Intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Classes:
  - `IdentityScaler`: No-op scaler that passes data through unchanged.
    - Methods: `fit`, `transform`, `fit_transform`, `inverse_transform`
  - `DiverseEnsemble`: Lightweight ensemble wrapper that combines predictions from multiple
    - Methods: `__init__`, `_aggregate_feature_importances`, `predict`
  - `TrainResult`: Result of training a single model.
  - `EnsembleResult`: Result of training the full regime-conditional ensemble.
  - `ModelTrainer`: Trains a regime-conditional gradient boosting ensemble for
    - Methods: `__init__`, `_spearmanr`, `_require_sklearn`, `_extract_dates`, `_sort_panel_by_time`, `_temporal_holdout_masks`, `_date_purged_folds`, `_prune_correlated_features`, `_select_features`, `train_ensemble`, `_train_single`, `_train_diverse_ensemble`, `_optimize_ensemble_weights`, `_clone_model`, `_make_model`, `_save`, `_print_summary`
- Top-level functions: none

### `models/versioning.py`
- Intent: Model Versioning — timestamped model directories with registry.
- Classes:
  - `ModelVersion`: Metadata for a single model version.
    - Methods: `to_dict`, `from_dict`
  - `ModelRegistry`: Manages versioned model storage and retrieval.
    - Methods: `__init__`, `_load_registry`, `_save_registry`, `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`
- Top-level functions: none

### `models/walk_forward.py`
- Intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Classes: none
- Top-level functions: `_spearmanr`, `_expanding_walk_forward_folds`, `_extract_dates`, `walk_forward_select`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
