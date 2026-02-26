# `tests` Package Guide

## Purpose

Project test suite (engine + API tests under `tests/`).

## Package Summary

- Modules: 69
- Classes: 286
- Top-level functions: 164
- LOC: 20,539

## How This Package Fits Into The System

- Executable behavioral specification for engine, API, and integration workflows.
- Includes API route/envelope/job tests under `tests/api/` and engine/regression tests under `tests/`.
- Prefer test assertions over stale narrative docs when they conflict.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `tests/__init__.py` | 7 | 0 | 0 | Project-level test suite package. |
| `tests/api/__init__.py` | 1 | 0 | 0 | No module docstring. |
| `tests/api/test_compute_routers.py` | 66 | 0 | 6 | Tests for POST compute endpoints — verify job creation. |
| `tests/api/test_envelope.py` | 52 | 0 | 6 | Tests for ApiResponse envelope and ResponseMeta. |
| `tests/api/test_integration.py` | 73 | 0 | 4 | Integration tests — full app startup, envelope consistency, config patch. |
| `tests/api/test_jobs.py` | 136 | 0 | 12 | Tests for the job system: store, runner, lifecycle. |
| `tests/api/test_main.py` | 49 | 0 | 5 | Tests for app factory and basic middleware. |
| `tests/api/test_read_routers.py` | 119 | 0 | 14 | Tests for all GET endpoints — verify ApiResponse envelope. |
| `tests/api/test_services.py` | 81 | 0 | 10 | Tests for service wrappers — verify dict outputs. |
| `tests/conftest.py` | 241 | 1 | 9 | Shared test fixtures for the quant_engine test suite. |
| `tests/test_ab_testing_framework.py` | 508 | 9 | 2 | Tests for the A/B testing framework (Spec 014). |
| `tests/test_autopilot_predictor_fallback.py` | 59 | 1 | 0 | Test module for autopilot predictor fallback behavior and regressions. |
| `tests/test_backtest_realism.py` | 562 | 7 | 5 | Spec 011 — Backtest Execution Realism & Validation Enforcement tests. |
| `tests/test_bocpd.py` | 345 | 8 | 4 | Unit tests for BOCPD (Bayesian Online Change-Point Detection). |
| `tests/test_cache_metadata_rehydrate.py` | 98 | 1 | 1 | Test module for cache metadata rehydrate behavior and regressions. |
| `tests/test_charting_endpoints.py` | 320 | 6 | 1 | Tests for charting API endpoints and data flow (Spec 004). |
| `tests/test_conceptual_fixes.py` | 325 | 4 | 2 | Tests for Spec 008 — Ensemble & Conceptual Fixes. |
| `tests/test_covariance_estimator.py` | 26 | 1 | 0 | Test module for covariance estimator behavior and regressions. |
| `tests/test_data_diagnostics.py` | 184 | 4 | 0 | Tests for data loading diagnostics (Spec 005). |
| `tests/test_delisting_total_return.py` | 77 | 1 | 0 | Test module for delisting total return behavior and regressions. |
| `tests/test_drawdown_liquidation.py` | 139 | 6 | 0 | Test module for drawdown liquidation behavior and regressions. |
| `tests/test_ensemble_voting.py` | 221 | 2 | 2 | Tests for confidence-weighted ensemble voting (SPEC_10 T2). |
| `tests/test_evaluation_engine.py` | 326 | 4 | 1 | Tests for evaluation/engine.py — EvaluationEngine integration tests. |
| `tests/test_evaluation_fragility.py` | 159 | 5 | 0 | Tests for evaluation/fragility.py — PnL concentration, drawdown, recovery, critical slowing. |
| `tests/test_evaluation_metrics.py` | 125 | 2 | 2 | Tests for evaluation/metrics.py — slice metrics and decile spread. |
| `tests/test_evaluation_ml_diagnostics.py` | 110 | 2 | 0 | Tests for evaluation/ml_diagnostics.py — feature drift and ensemble disagreement. |
| `tests/test_evaluation_slicing.py` | 131 | 2 | 2 | Tests for evaluation/slicing.py — PerformanceSlice and SliceRegistry. |
| `tests/test_execution_dynamic_costs.py` | 49 | 1 | 0 | Test module for execution dynamic costs behavior and regressions. |
| `tests/test_execution_layer.py` | 849 | 17 | 2 | Comprehensive tests for Spec 06: Execution Layer Improvements. |
| `tests/test_feature_fixes.py` | 378 | 7 | 2 | Tests verifying Spec 012 feature engineering fixes: |
| `tests/test_health_checks_rewritten.py` | 417 | 8 | 4 | Tests for the rewritten health check system (Spec 010). |
| `tests/test_health_spec09.py` | 721 | 7 | 4 | Tests for Health System Spec 09 enhancements. |
| `tests/test_health_transparency.py` | 283 | 5 | 2 | Tests for health system transparency and trustworthiness (Spec 002). |
| `tests/test_integration.py` | 557 | 4 | 1 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 39 | 1 | 0 | Test module for iv arbitrage builder behavior and regressions. |
| `tests/test_jump_model_pypi.py` | 379 | 10 | 3 | Unit tests for the PyPI JumpModel wrapper (Spec 001). |
| `tests/test_jump_model_validation.py` | 397 | 6 | 8 | Jump Model Audit and Validation (SPEC_10 T1). |
| `tests/test_kalshi_asof_features.py` | 66 | 1 | 0 | Test module for kalshi asof features behavior and regressions. |
| `tests/test_kalshi_distribution.py` | 115 | 1 | 0 | Test module for kalshi distribution behavior and regressions. |
| `tests/test_kalshi_hardening.py` | 606 | 1 | 0 | Test module for kalshi hardening behavior and regressions. |
| `tests/test_loader_and_predictor.py` | 231 | 3 | 0 | Test module for loader and predictor behavior and regressions. |
| `tests/test_lookahead_detection.py` | 145 | 1 | 1 | Automated lookahead bias detection for the feature pipeline. |
| `tests/test_observation_matrix_expansion.py` | 186 | 3 | 2 | Tests for expanded HMM observation matrix (SPEC_10 T4). |
| `tests/test_online_update.py` | 237 | 6 | 1 | Tests for online regime updating via forward algorithm (SPEC_10 T5). |
| `tests/test_panel_split.py` | 56 | 1 | 0 | Test module for panel split behavior and regressions. |
| `tests/test_paper_trader_integration.py` | 844 | 12 | 7 | Test module for paper trader integration — spec 015. |
| `tests/test_paper_trader_kelly.py` | 129 | 1 | 3 | Test module for paper trader kelly behavior and regressions. |
| `tests/test_portfolio_layer.py` | 1027 | 10 | 3 | Comprehensive test suite for Spec 07: Portfolio Layer + Regime-Conditioned Constraints. |
| `tests/test_position_sizing_overhaul.py` | 425 | 8 | 0 | Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul. |
| `tests/test_promotion_contract.py` | 108 | 1 | 2 | Test module for promotion contract behavior and regressions. |
| `tests/test_provider_registry.py` | 30 | 1 | 0 | Test module for provider registry behavior and regressions. |
| `tests/test_regime_consensus.py` | 205 | 4 | 0 | Tests for cross-sectional regime consensus (SPEC_10 T6). |
| `tests/test_regime_detection_integration.py` | 610 | 8 | 2 | End-to-end integration tests for the upgraded regime detection pipeline (SPEC_10 T7). |
| `tests/test_regime_integration.py` | 312 | 6 | 1 | Integration tests for the full regime detection pipeline with Jump Model (Spec 001). |
| `tests/test_regime_payload.py` | 242 | 4 | 2 | Tests for regime data pipeline end-to-end (Spec 003). |
| `tests/test_regime_uncertainty.py` | 163 | 2 | 0 | Tests for regime uncertainty integration (SPEC_10 T3). |
| `tests/test_research_factors.py` | 134 | 1 | 1 | Test module for research factors behavior and regressions. |
| `tests/test_risk_governor.py` | 713 | 11 | 0 | Comprehensive tests for Spec 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing. |
| `tests/test_risk_improvements.py` | 493 | 9 | 0 | Comprehensive tests for Spec 016: Risk System Improvements. |
| `tests/test_shock_vector.py` | 373 | 6 | 0 | Unit tests for ShockVector and ShockVectorValidator. |
| `tests/test_signal_meta_labeling.py` | 757 | 7 | 6 | Tests for Spec 04: Signal Enhancement + Meta-Labeling + Fold-Level Validation. |
| `tests/test_structural_features.py` | 821 | 8 | 5 | Tests for Spec 02: Structural Feature Expansion. |
| `tests/test_structural_state_integration.py` | 385 | 7 | 2 | Integration tests for the Structural State Layer (SPEC_03). |
| `tests/test_survivorship_pit.py` | 75 | 1 | 0 | Test module for survivorship pit behavior and regressions. |
| `tests/test_system_innovations.py` | 773 | 16 | 0 | Comprehensive tests for Spec 017: System-Level Innovation. |
| `tests/test_training_pipeline_fixes.py` | 568 | 5 | 3 | Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance. |
| `tests/test_truth_layer.py` | 872 | 7 | 2 | Tests for Spec 01: Foundational Hardening — Truth Layer. |
| `tests/test_validation_and_risk_extensions.py` | 108 | 1 | 1 | Test module for validation and risk extensions behavior and regressions. |
| `tests/test_zero_errors.py` | 121 | 1 | 6 | Integration test: common operations must produce zero ERROR-level log entries. |

## Related Docs

- `../docs/reference/TEST_SPEC_MAP.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
