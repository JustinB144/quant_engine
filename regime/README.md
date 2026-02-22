# `regime` Package Guide

## Purpose

Market regime detection and regime-derived features (rule-based + HMM + correlation regime).

## Package Summary

- Modules: 4
- Classes: 5
- Top-level functions: 5
- LOC: 1,019

## How This Package Fits Into the System

- Consumes feature panels
- Feeds `models`, `backtest`, `autopilot`, and UI diagnostics
- Provides canonical regime labels, confidence, and probabilities

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `regime/__init__.py` | 14 | 0 | 0 | Regime modeling components. |
| `regime/correlation.py` | 212 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 292 | 2 | 1 | Regime detector with two engines: |
| `regime/hmm.py` | 501 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |

## Module Details

### `regime/__init__.py`
- Intent: Regime modeling components.
- Classes: none
- Top-level functions: none

### `regime/correlation.py`
- Intent: Correlation Regime Detection (NEW 11).
- Classes:
  - `CorrelationRegimeDetector`: Detect regime changes in pairwise correlation structure.
    - Methods: `__init__`, `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`
- Top-level functions: none

### `regime/detector.py`
- Intent: Regime detector with two engines:
- Classes:
  - `RegimeOutput`: No class docstring.
  - `RegimeDetector`: Classifies market regime at each bar using either rules or HMM.
    - Methods: `__init__`, `detect`, `_rule_detect`, `_hmm_detect`, `detect_with_confidence`, `detect_full`, `regime_features`, `_get_col`
- Top-level functions: `detect_regimes_batch`

### `regime/hmm.py`
- Intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Classes:
  - `HMMFitResult`: No class docstring.
  - `GaussianHMM`: Gaussian HMM using EM (Baum-Welch).
    - Methods: `__init__`, `_ensure_positive_definite`, `_init_params`, `_log_emission`, `_forward_backward`, `viterbi`, `_smooth_duration`, `fit`, `predict_proba`
- Top-level functions: `_logsumexp`, `select_hmm_states_bic`, `build_hmm_observation_matrix`, `map_raw_states_to_regimes`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
