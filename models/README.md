# `models` Package Guide

## Purpose

Models subpackage — training, prediction, versioning, and retraining triggers.

## Package Summary

- Modules: 14
- Classes: 28
- Top-level functions: 9
- LOC: 5,130

## How This Package Fits Into The System

- Contains training, prediction, versioning, governance, retrain triggers, and IV surface models.
- Model artifacts are persisted under `trained_models/` and read by CLI, API services, and autopilot.
- `models/iv/models.py` powers the IV Surface API demo endpoint and frontend IV Surface page visualizations.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `models/__init__.py` | 20 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 327 | 2 | 2 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/cross_sectional.py` | 136 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 313 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 155 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/iv/__init__.py` | 31 | 0 | 0 | Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface. |
| `models/iv/models.py` | 937 | 10 | 1 | Implied Volatility Surface Models. |
| `models/neural_net.py` | 198 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/online_learning.py` | 273 | 2 | 0 | Online learning module for incremental model updates between full retrains. |
| `models/predictor.py` | 404 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 296 | 1 | 0 | ML Retraining Trigger Logic |
| `models/trainer.py` | 1598 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 207 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
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
    - Methods: `fit`, `transform`
  - `ConfidenceCalibrator`: Post-hoc confidence calibration via Platt scaling or isotonic regression.
    - Methods: `fit`, `transform`, `fit_transform`, `is_fitted`, `backend`
- Top-level functions: `compute_ece`, `compute_reliability_curve`

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
    - Methods: `record_importance`, `check_stability`
- Top-level functions: none

### `models/governance.py`
- Intent: Champion/challenger governance for model versions.
- Classes:
  - `ChampionRecord`: Persisted champion model record for a prediction horizon.
    - Methods: `to_dict`
  - `ModelGovernance`: Maintains champion model per horizon and promotes challengers if better.
    - Methods: `get_champion_version`, `evaluate_and_update`
- Top-level functions: none

### `models/iv/__init__.py`
- Intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- Classes: none
- Top-level functions: none

### `models/iv/models.py`
- Intent: Implied Volatility Surface Models.
- Classes:
  - `OptionType`: Supported option contract types for pricing and volatility surface models.
  - `Greeks`: Option Greeks container.
  - `HestonParams`: Heston model parameters.
    - Methods: `validate`
  - `SVIParams`: Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).
  - `BlackScholes`: Black-Scholes option pricing and analytics.
    - Methods: `price`, `greeks`, `implied_vol`, `iv_surface`
  - `HestonModel`: Heston (1993) stochastic volatility model.
    - Methods: `characteristic_function`, `price`, `implied_vol`, `iv_surface`, `calibrate`
  - `SVIModel`: SVI (Stochastic Volatility Inspired) implied variance parameterization.
    - Methods: `total_variance`, `implied_vol`, `iv_surface`, `smile`, `calibrate`, `check_no_butterfly_arbitrage`
  - `ArbitrageFreeSVIBuilder`: Arbitrage-aware SVI surface builder.
    - Methods: `fit_slice`, `enforce_calendar_monotonicity`, `interpolate_total_variance`, `build_surface`
  - `IVPoint`: Single implied-volatility observation.
  - `IVSurface`: Store and interpolate an implied-volatility surface.
    - Methods: `add_point`, `add_slice`, `add_surface`, `n_points`, `get_iv`, `get_smile`, `decompose`, `decompose_surface`
- Top-level functions: `generate_synthetic_market_surface`

### `models/neural_net.py`
- Intent: Tabular Neural Network — feedforward network for tabular financial data.
- Classes:
  - `TabularNet`: Feedforward network for tabular financial data.
    - Methods: `fit`, `predict`, `feature_importances_`
- Top-level functions: none

### `models/online_learning.py`
- Intent: Online learning module for incremental model updates between full retrains.
- Classes:
  - `OnlineUpdate`: Record of a single online update step.
  - `OnlineLearner`: Incremental model updater between full retrains.
    - Methods: `add_sample`, `update`, `adjust_prediction`, `should_retrain`, `get_status`, `load_state`
- Top-level functions: none

### `models/predictor.py`
- Intent: Model Predictor — loads trained ensemble and generates predictions.
- Classes:
  - `EnsemblePredictor`: Loads a trained regime-conditional ensemble and generates predictions.
    - Methods: `predict`, `predict_single`
- Top-level functions: `_prepare_features`

### `models/retrain_trigger.py`
- Intent: ML Retraining Trigger Logic
- Classes:
  - `RetrainTrigger`: Determines when ML model should be retrained.
    - Methods: `add_trade_result`, `check`, `record_retraining`, `status`
- Top-level functions: none

### `models/trainer.py`
- Intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Classes:
  - `IdentityScaler`: No-op scaler that passes data through unchanged.
    - Methods: `fit`, `transform`, `fit_transform`, `inverse_transform`
  - `DiverseEnsemble`: Lightweight ensemble wrapper that combines predictions from multiple
    - Methods: `predict`
  - `TrainResult`: Result of training a single model.
  - `EnsembleResult`: Result of training the full regime-conditional ensemble.
  - `ModelTrainer`: Trains a regime-conditional gradient boosting ensemble for
    - Methods: `train_ensemble`
- Top-level functions: none

### `models/versioning.py`
- Intent: Model Versioning — timestamped model directories with registry.
- Classes:
  - `ModelVersion`: Metadata for a single model version.
    - Methods: `to_dict`, `from_dict`
  - `ModelRegistry`: Manages versioned model storage and retrieval.
    - Methods: `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`
- Top-level functions: none

### `models/walk_forward.py`
- Intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Classes: none
- Top-level functions: `_spearmanr`, `_expanding_walk_forward_folds`, `_extract_dates`, `walk_forward_select`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` (entrypoints and workflows)
