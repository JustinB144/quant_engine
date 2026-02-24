# `regime` Package Guide

## Purpose

Regime modeling components.

## Package Summary

- Modules: 5
- Classes: 7
- Top-level functions: 5
- LOC: 1,624

## How This Package Fits Into The System

- Provides rule-based, HMM, jump-model, and correlation regime detection plus regime feature outputs.
- Regime labels/probabilities feed training, prediction, backtesting, dashboard regime views, and autopilot gating.
- Canonical regime semantics are configured in `config.py` (`REGIME_NAMES`, gating/suppression flags).

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `regime/__init__.py` | 17 | 0 | 0 | Regime modeling components. |
| `regime/correlation.py` | 213 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 586 | 2 | 1 | Regime detector with two engines: |
| `regime/hmm.py` | 566 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |
| `regime/jump_model.py` | 242 | 2 | 0 | Statistical Jump Model for regime detection. |

## Module Details

### `regime/__init__.py`
- Intent: Regime modeling components.
- Classes: none
- Top-level functions: none

### `regime/correlation.py`
- Intent: Correlation Regime Detection (NEW 11).
- Classes:
  - `CorrelationRegimeDetector`: Detect regime changes in pairwise correlation structure.
    - Methods: `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`
- Top-level functions: none

### `regime/detector.py`
- Intent: Regime detector with two engines:
- Classes:
  - `RegimeOutput`: Unified regime detection output consumed by modeling, backtesting, and UI layers.
  - `RegimeDetector`: Classifies market regime at each bar using either rules or HMM.
    - Methods: `detect`, `detect_ensemble`, `detect_with_confidence`, `detect_full`, `regime_features`, `get_regime_uncertainty`, `map_raw_states_to_regimes_stable`
- Top-level functions: `detect_regimes_batch`

### `regime/hmm.py`
- Intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Classes:
  - `HMMFitResult`: Fitted HMM outputs including decoded states, posteriors, transitions, and log-likelihood.
  - `GaussianHMM`: Gaussian HMM using EM (Baum-Welch).
    - Methods: `viterbi`, `fit`, `predict_proba`
- Top-level functions: `_logsumexp`, `select_hmm_states_bic`, `build_hmm_observation_matrix`, `map_raw_states_to_regimes`

### `regime/jump_model.py`
- Intent: Statistical Jump Model for regime detection.
- Classes:
  - `JumpModelResult`: Result from fitting a Statistical Jump Model.
  - `StatisticalJumpModel`: Statistical Jump Model for regime detection.
    - Methods: `fit`, `compute_jump_penalty_from_data`, `predict`
- Top-level functions: none

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
