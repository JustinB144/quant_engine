# `models` Package Guide

## Purpose

Model training, prediction, versioning, governance, and supporting ML utilities.

## Package Summary

- Modules: 14
- Classes: 24
- Top-level functions: 8
- LOC: 5,001

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `models/__init__.py` | 25 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 328 | 2 | 2 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/conformal.py` | 296 | 3 | 0 | Conformal Prediction — distribution-free prediction intervals. |
| `models/cross_sectional.py` | 137 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 314 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 156 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/neural_net.py` | 199 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/online_learning.py` | 274 | 2 | 0 | Online learning module for incremental model updates between full retrains. |
| `models/predictor.py` | 485 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 345 | 1 | 0 | ML Retraining Trigger Logic |
| `models/shift_detection.py` | 323 | 3 | 0 | Distribution Shift Detection — CUSUM and PSI methods. |
| `models/trainer.py` | 1675 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 208 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
| `models/walk_forward.py` | 236 | 0 | 4 | Walk-Forward Model Selection — expanding-window hyperparameter search |

## Module Details

### `models/__init__.py`
- Intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- Classes: none
- Top-level functions: none

### `models/calibration.py`
- Intent: Confidence Calibration --- Platt scaling and isotonic regression.
- Classes:
  - `_LinearRescaler` (methods: `fit`, `transform`)
  - `ConfidenceCalibrator` (methods: `fit`, `transform`, `fit_transform`, `is_fitted`, `backend`)
- Top-level functions: `compute_ece`, `compute_reliability_curve`

### `models/conformal.py`
- Intent: Conformal Prediction — distribution-free prediction intervals.
- Classes:
  - `ConformalInterval` (methods: none)
  - `ConformalCalibrationResult` (methods: none)
  - `ConformalPredictor` (methods: `is_calibrated`, `calibrate`, `predict_interval`, `predict_intervals_batch`, `uncertainty_scalars`, `evaluate_coverage`, `to_dict`, `from_dict`)
- Top-level functions: none

### `models/cross_sectional.py`
- Intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- Classes: none
- Top-level functions: `cross_sectional_rank`

### `models/feature_stability.py`
- Intent: Feature Stability Monitoring — tracks feature importance rankings across
- Classes:
  - `StabilityReport` (methods: `to_dict`)
  - `FeatureStabilityTracker` (methods: `record_importance`, `check_stability`)
- Top-level functions: none

### `models/governance.py`
- Intent: Champion/challenger governance for model versions.
- Classes:
  - `ChampionRecord` (methods: `to_dict`)
  - `ModelGovernance` (methods: `get_champion_version`, `evaluate_and_update`)
- Top-level functions: none

### `models/neural_net.py`
- Intent: Tabular Neural Network — feedforward network for tabular financial data.
- Classes:
  - `TabularNet` (methods: `fit`, `predict`, `feature_importances_`)
- Top-level functions: none

### `models/online_learning.py`
- Intent: Online learning module for incremental model updates between full retrains.
- Classes:
  - `OnlineUpdate` (methods: none)
  - `OnlineLearner` (methods: `add_sample`, `update`, `adjust_prediction`, `should_retrain`, `get_status`, `load_state`)
- Top-level functions: none

### `models/predictor.py`
- Intent: Model Predictor — loads trained ensemble and generates predictions.
- Classes:
  - `EnsemblePredictor` (methods: `predict`, `blend_multi_horizon`, `predict_single`)
- Top-level functions: `_prepare_features`

### `models/retrain_trigger.py`
- Intent: ML Retraining Trigger Logic
- Classes:
  - `RetrainTrigger` (methods: `add_trade_result`, `check_shift`, `check`, `record_retraining`, `status`)
- Top-level functions: none

### `models/shift_detection.py`
- Intent: Distribution Shift Detection — CUSUM and PSI methods.
- Classes:
  - `CUSUMResult` (methods: none)
  - `PSIResult` (methods: none)
  - `DistributionShiftDetector` (methods: `set_reference`, `check_cusum`, `check_psi`, `check_all`)
- Top-level functions: none

### `models/trainer.py`
- Intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Classes:
  - `IdentityScaler` (methods: `fit`, `transform`, `fit_transform`, `inverse_transform`)
  - `DiverseEnsemble` (methods: `predict`)
  - `TrainResult` (methods: none)
  - `EnsembleResult` (methods: none)
  - `ModelTrainer` (methods: `train_ensemble`, `compute_shared_features`)
- Top-level functions: none

### `models/versioning.py`
- Intent: Model Versioning — timestamped model directories with registry.
- Classes:
  - `ModelVersion` (methods: `to_dict`, `from_dict`)
  - `ModelRegistry` (methods: `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`)
- Top-level functions: none

### `models/walk_forward.py`
- Intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Classes: none
- Top-level functions: `_spearmanr`, `_expanding_walk_forward_folds`, `_extract_dates`, `walk_forward_select`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
