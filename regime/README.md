# `regime` Package Guide

## Purpose

Regime detection, structural state, uncertainty, and online-update components.

## Package Summary

- Modules: 13
- Classes: 17
- Top-level functions: 6
- LOC: 4,243

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `regime/__init__.py` | 41 | 0 | 0 | Regime modeling components. |
| `regime/bocpd.py` | 452 | 3 | 0 | Bayesian Online Change-Point Detection (BOCPD) with Gaussian likelihood. |
| `regime/confidence_calibrator.py` | 252 | 1 | 0 | Confidence Calibrator for regime ensemble voting (SPEC_10 T2). |
| `regime/consensus.py` | 274 | 1 | 0 | Cross-Sectional Regime Consensus (SPEC_10 T6). |
| `regime/correlation.py` | 214 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 941 | 2 | 2 | Regime detector with multiple engines and structural state layer. |
| `regime/hmm.py` | 662 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |
| `regime/jump_model.py` | 10 | 0 | 0 | Backward-compatible re-export of legacy Statistical Jump Model. |
| `regime/jump_model_legacy.py` | 243 | 2 | 0 | Statistical Jump Model for regime detection. |
| `regime/jump_model_pypi.py` | 421 | 1 | 0 | PyPI jumpmodels package wrapper for regime detection. |
| `regime/online_update.py` | 246 | 1 | 0 | Online Regime Updating via Forward Algorithm (SPEC_10 T5). |
| `regime/shock_vector.py` | 306 | 2 | 0 | Unified Shock/Structure Vector — version-locked market state representation. |
| `regime/uncertainty_gate.py` | 181 | 1 | 0 | Regime Uncertainty Gate — entropy-based position sizing modifier (SPEC_10 T3). |

## Module Details

### `regime/__init__.py`
- Intent: Regime modeling components.
- Classes: none
- Top-level functions: none

### `regime/bocpd.py`
- Intent: Bayesian Online Change-Point Detection (BOCPD) with Gaussian likelihood.
- Classes:
  - `BOCPDResult` (methods: none)
  - `BOCPDBatchResult` (methods: none)
  - `BOCPDDetector` (methods: `reset`, `update`, `batch_update`)
- Top-level functions: none

### `regime/confidence_calibrator.py`
- Intent: Confidence Calibrator for regime ensemble voting (SPEC_10 T2).
- Classes:
  - `ConfidenceCalibrator` (methods: `fitted`, `component_weights`, `fit`, `calibrate`, `get_component_weight`, `expected_calibration_error`)
- Top-level functions: none

### `regime/consensus.py`
- Intent: Cross-Sectional Regime Consensus (SPEC_10 T6).
- Classes:
  - `RegimeConsensus` (methods: `compute_consensus`, `detect_divergence`, `early_warning`, `compute_consensus_series`, `reset_history`)
- Top-level functions: none

### `regime/correlation.py`
- Intent: Correlation Regime Detection (NEW 11).
- Classes:
  - `CorrelationRegimeDetector` (methods: `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`)
- Top-level functions: none

### `regime/detector.py`
- Intent: Regime detector with multiple engines and structural state layer.
- Classes:
  - `RegimeOutput` (methods: none)
  - `RegimeDetector` (methods: `detect`, `detect_ensemble`, `calibrate_confidence_weights`, `detect_with_confidence`, `detect_full`, `regime_features`, `get_regime_uncertainty`, `map_raw_states_to_regimes_stable`, `detect_with_shock_context`, `detect_batch_with_shock_context`)
- Top-level functions: `validate_hmm_observation_features`, `detect_regimes_batch`

### `regime/hmm.py`
- Intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Classes:
  - `HMMFitResult` (methods: none)
  - `GaussianHMM` (methods: `viterbi`, `fit`, `predict_proba`)
- Top-level functions: `_logsumexp`, `select_hmm_states_bic`, `build_hmm_observation_matrix`, `map_raw_states_to_regimes`

### `regime/jump_model.py`
- Intent: Backward-compatible re-export of legacy Statistical Jump Model.
- Classes: none
- Top-level functions: none

### `regime/jump_model_legacy.py`
- Intent: Statistical Jump Model for regime detection.
- Classes:
  - `JumpModelResult` (methods: none)
  - `StatisticalJumpModel` (methods: `fit`, `compute_jump_penalty_from_data`, `predict`)
- Top-level functions: none

### `regime/jump_model_pypi.py`
- Intent: PyPI jumpmodels package wrapper for regime detection.
- Classes:
  - `PyPIJumpModel` (methods: `fit`, `predict_online`, `predict_proba_online`)
- Top-level functions: none

### `regime/online_update.py`
- Intent: Online Regime Updating via Forward Algorithm (SPEC_10 T5).
- Classes:
  - `OnlineRegimeUpdater` (methods: `forward_step`, `update_regime_for_security`, `update_batch`, `should_refit`, `reset_security_cache`, `cached_securities`, `get_state_probabilities`)
- Top-level functions: none

### `regime/shock_vector.py`
- Intent: Unified Shock/Structure Vector — version-locked market state representation.
- Classes:
  - `ShockVector` (methods: `to_dict`, `from_dict`, `is_shock_event`, `regime_name`)
  - `ShockVectorValidator` (methods: `validate`, `batch_validate`)
- Top-level functions: none

### `regime/uncertainty_gate.py`
- Intent: Regime Uncertainty Gate — entropy-based position sizing modifier (SPEC_10 T3).
- Classes:
  - `UncertaintyGate` (methods: `compute_size_multiplier`, `apply_uncertainty_gate`, `should_assume_stress`, `is_uncertain`, `gate_series`)
- Top-level functions: none

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
