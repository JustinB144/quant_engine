# Audit Report: Subsystem 07 — Evaluation & Diagnostics

> **Status:** Complete
> **Auditor:** Codex (GPT-5)
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_07_EVALUATION_DIAGNOSTICS.md`

---

## Findings (Severity-Ranked)

### F-01 — Net-zero PnL is misclassified as "no trades" in fragility diagnostics [P1 HIGH]

- **File:** `evaluation/fragility.py:69-73`
- **Proof:** `pnl_concentration()` returns an "empty" result when `abs(total_pnl) < 1e-12`, setting `n_trades` to `0` even when trades exist.
- **Observed repro:**
  - Input trades: `[+10, -10, +5, -5]`
  - Output: `{'total_pnl': 0.0, 'n_trades': 0, ...}`
- **Impact:** Fragility reporting can silently suppress concentration risk for strategies with offsetting wins/losses (net ~0 but non-trivial gross activity).
- **Recommendation:** Treat `n_trades == 0` separately from `total_pnl == 0`; compute concentration on `abs(pnl)` whenever trades exist.

### F-02 — NaN handling in slice metrics can zero-out IC and misreport date ranges [P1 HIGH]

- **File:** `evaluation/metrics.py:55-58, 104-111, 121-122`
- **Proof:**
  1. Returns are filtered for finites (`ret_arr = ret_arr[np.isfinite(ret_arr)]`) and `n` is recomputed.
  2. Predictions are not filtered by the same mask; IC is computed only when `len(pred_arr) == n`.
  3. `start_date`/`end_date` are taken from original `returns.index`, not filtered observations.
- **Observed repro:**
  - With one NaN in returns and aligned predictions, `ic` becomes `0.0` due to length mismatch.
  - With NaNs at start/end, report dates still show full range including NaN endpoints.
- **Impact:** Report correctness drift for slices with missing data (IC and reported date span can be incorrect).
- **Recommendation:** Build a single validity mask over returns/predictions and propagate filtered index for date fields.

### F-03 — Multiple evaluation thresholds are configuration-declared but not actually wired [P2 MEDIUM]

- **Files:** `evaluation/engine.py:34, 37-41, 327`; `backtest/validation.py:928-930`; `evaluation/fragility.py:98-101`
- **Proof:**
  - `EVAL_IC_DECAY_LOOKBACK`, `EVAL_OVERFIT_GAP_THRESHOLD`, `EVAL_PNL_CONCENTRATION_THRESHOLD` are imported in engine but not used in decision logic.
  - Overfit threshold is hardcoded in `walk_forward_with_embargo()` (`mean_gap > 0.10`).
  - Fragility threshold is hardcoded (`top_N_pct > 0.70`).
- **Impact:** Changing these config values may have no effect on red-flag behavior.
- **Recommendation:** Pass thresholds explicitly from engine into downstream functions and remove hardcoded literals.

### F-04 — Public function parameters are unused in fragility module [P2 MEDIUM]

- **Files:** `evaluation/fragility.py:108` (`window`), `evaluation/fragility.py:189` (`lookback`)
- **Proof:** Neither parameter is referenced in its function body.
- **Observed repro:** Calling with very different parameter values returns identical outputs.
- **Impact:** API contract is misleading; callers cannot control analysis horizon as advertised.
- **Recommendation:** Either implement parameter usage or remove/deprecate parameters and update docs/tests.

### F-05 — Underwater visualization does not handle NaNs, producing NaN-tailed series [P2 MEDIUM]

- **File:** `evaluation/visualization.py:190-193`
- **Proof:** `plot_underwater()` computes cumulative equity directly from raw returns without finite filtering.
- **Observed repro:** Returns `[0.01, NaN, 0.02]` produce drawdowns `[0.0, nan, nan]`.
- **Impact:** Chart integrity can break despite metric functions handling finite filtering.
- **Recommendation:** Drop/forward-fill NaNs consistently (or explicitly gap them) before cumulative computations.

### F-06 — Decile spread minimum threshold constant is unused [P3 LOW]

- **File:** `evaluation/metrics.py:20`
- **Proof:** `EVAL_DECILE_SPREAD_MIN` is imported but never referenced in `metrics.py` or engine red-flag logic.
- **Impact:** Config surface includes a threshold that does not influence diagnostics.
- **Recommendation:** Either apply threshold in red-flag logic or remove constant/import.

### F-07 — Boundary documentation drift: evaluation no longer reads `backtest_*_summary.json` [P3 LOW]

- **Files:** `docs/audit/data/INTERFACE_CONTRACTS.yaml:1069-1093`; `evaluation/engine.py`
- **Proof:** Interface contract states summary JSON is read by `evaluation/engine.py`; code has no summary-artifact reader path.
- **Impact:** Audit boundary metadata is stale and may mislead downstream audits.
- **Recommendation:** Update boundary contract/shared artifact metadata to match current implementation.

---

## Executive Summary

All **8 subsystem files** and **2,816 lines** were reviewed line-by-line, with read-only contract verification against `backtest/validation.py`, `models/calibration.py`, and relevant `config.py` constants. Walk-forward/embargo and calibration interfaces are present and callable as specified. However, report-correctness issues exist around fragility edge cases, NaN handling, and threshold/config wiring.

- **P0:** 0
- **P1:** 2
- **P2:** 3
- **P3:** 2

No production code changes were made.

---

## Scope & Ledger (T1)

### File Coverage

| # | File | Lines | Reviewed |
|---|---|---:|---|
| 1 | `evaluation/engine.py` | 826 | Yes |
| 2 | `evaluation/fragility.py` | 452 | Yes |
| 3 | `evaluation/slicing.py` | 400 | Yes |
| 4 | `evaluation/visualization.py` | 386 | Yes |
| 5 | `evaluation/metrics.py` | 324 | Yes |
| 6 | `evaluation/ml_diagnostics.py` | 253 | Yes |
| 7 | `evaluation/calibration_analysis.py` | 142 | Yes |
| 8 | `evaluation/__init__.py` | 33 | Yes |

**Total: 2,816 lines reviewed (100% of Subsystem 7).**

### Read-Only Contract Files Reviewed

| File | Why |
|---|---|
| `backtest/validation.py` | `walk_forward_with_embargo`, `rolling_ic`, `detect_ic_decay` contract + time-order semantics |
| `models/calibration.py` | `compute_ece`, `compute_reliability_curve` signature/shape contract |
| `config.py` | Evaluation threshold/config constant integrity (`evaluation_to_config_23`) |

---

## Metric/Field Registry Snapshot (T1)

### Primary outputs produced by evaluation subsystem

| Producer | Output keys/fields |
|---|---|
| `compute_slice_metrics()` | `mean_return`, `annualized_return`, `sharpe`, `sharpe_se`, `max_dd`, `max_dd_duration`, `recovery_time_mean`, `n_samples`, `start_date`, `end_date`, `win_rate`, `ic`, `confidence`, `std_return` |
| `decile_spread()` | `spread`, `spread_t_stat`, `spread_pvalue`, `decile_returns`, `decile_counts`, `monotonicity`, `n_total`, `significant`, optional `per_regime` |
| `analyze_calibration()` | `calibration_error`, `ece`, `overconfident`, `max_gap`, `reliability_curve`, `n_samples`, `overconfidence_threshold` |
| `pnl_concentration()` | `top_{n}_pct`, `total_pnl`, `n_trades`, `herfindahl_index`, `fragile` |
| `drawdown_distribution()` | `max_dd`, `max_dd_single_day`, `avg_dd_during_episodes`, `n_episodes`, `dd_concentration`, `pct_time_underwater`, `max_dd_duration` |
| `feature_importance_drift()` | `drift_detected`, `correlations_per_period`, `mean_correlation`, `min_correlation`, `top_k_stability`, `top_k_features_per_period`, `n_periods` |
| `ensemble_disagreement()` | `mean_correlation`, `min_correlation`, `max_correlation`, `disagreement_pairs`, `n_models`, `high_disagreement`, `pairwise_correlations` |
| `EvaluationResult` | Aggregate + all section outputs + `red_flags`, `overall_pass`, `summary` |

### Internal consumers

| Consumer path | Fields consumed |
|---|---|
| `EvaluationEngine._generate_json()` | Most `EvaluationResult` fields serialized under stable top-level keys |
| `EvaluationEngine._generate_html()` | `aggregate_metrics`, slice metric dicts, walk-forward stats, IC, decile spread, calibration, fragility, drawdown, critical slowing, red flags |
| `evaluation/visualization.py` | Receives precomputed metrics/curves/folds and renders charts without mutating metric values |

---

## Time-Ordering & Validation Contract Pass (T2)

### Walk-forward / embargo semantics

- `walk_forward_with_embargo()` is temporally ordered and applies embargo gap correctly:
  - train `[t, t+train_window)`
  - embargo `[train_end, train_end+embargo)`
  - test starts at `train_end + embargo`
  - Source: `backtest/validation.py:875-905`
- Engine integration matches expected call pattern and parameters: `evaluation/engine.py:272-280`.

### Rolling IC and decay

- `rolling_ic()` uses trailing windows only: `backtest/validation.py:979-985`.
- `detect_ic_decay()` computes recent weakness and slope criteria: `backtest/validation.py:1025-1044`.
- Engine integration present: `evaluation/engine.py:315-330`.

### Fragility triggers & edge cases

- Fragility, drawdown, recovery, and slowing diagnostics execute with broad exception guards in engine (`evaluation/engine.py:376-423`).
- Edge-case defects documented in findings F-01 and F-04.

---

## Calibration & Diagnostics Pass (T3)

### Calibration interface integrity

- `compute_ece` signature matches contract exactly: `models/calibration.py:228-232`.
- `compute_reliability_curve` signature + key schema match: `models/calibration.py:277-327`.
- Evaluation lazy import and usage are correct: `evaluation/calibration_analysis.py:92,95,98`.

### ML diagnostics threshold interpretation

- Feature drift detection compares min/mean Spearman against threshold: `evaluation/ml_diagnostics.py:138`.
- Ensemble disagreement flags any pair below threshold: `evaluation/ml_diagnostics.py:217-218,237`.

---

## Visualization & Reporting Integrity Pass (T4)

### Non-mutation check

Visualization functions construct `data` dicts from inputs and return chart payloads; no in-place mutation of source metric dicts/series was observed.

### Missing-data / low-sample handling

- Metrics low-sample guard exists (`confidence` tiering + empty results): `evaluation/metrics.py:52-60,113-118,293-324`.
- Visualization no-data guards are present across functions (return `html=None`) but NaN handling is inconsistent for underwater charts (F-05).

---

## Boundary Checks & Contract Disposition (T5)

### `evaluation_to_models_backtest_11` disposition

| Contract symbol/artifact | Status | Evidence |
|---|---|---|
| `compute_ece` | PASS | Defined at `models/calibration.py:228`; imported/used at `evaluation/calibration_analysis.py:92,95` |
| `compute_reliability_curve` | PASS | Defined at `models/calibration.py:277`; imported/used at `evaluation/calibration_analysis.py:92,98` |
| `walk_forward_with_embargo` | PASS | Defined at `backtest/validation.py:818`; consumed at `evaluation/engine.py:272-280` |
| `rolling_ic` | PASS | Defined at `backtest/validation.py:948`; consumed at `evaluation/engine.py:315-317` |
| `detect_ic_decay` | PASS (caveat) | Defined at `backtest/validation.py:990`; consumed at `evaluation/engine.py:327-330` (window default used; config lookback not passed — see F-03) |
| `results/backtest_*d_summary.json` artifact coupling | PARTIAL / STALE CONTRACT DOC | Contract metadata claims `evaluation/engine.py` reader; current engine has no such reader path (F-07) |

### `evaluation_to_config_23` disposition

All referenced evaluation constants exist and are typed consistently in `config.py` (`config.py:661-710`, `config.py:217-222` for `REGIME_NAMES`).

Caveat: Several imported constants are not wired into active logic (F-03, F-06).

---

## Verification Commands Executed

- `jq -r '.subsystems.evaluation_diagnostics.files[]' docs/audit/data/SUBSYSTEM_MAP.json | xargs wc -l`
  - Output: 8 files, `2816 total`
- `rg -n "walk_forward|embargo|rolling_ic|detect_ic_decay|fragility|drawdown" ...`
  - Confirmed expected call paths in evaluation + backtest validation modules
- `rg -n "compute_ece|compute_reliability_curve|drift|disagreement|threshold" ...`
  - Confirmed calibration + ML diagnostics interfaces and threshold usage
- `rg -n "plot|figure|chart|nan|min_samples|insufficient" ...`
  - Confirmed visualization/reporting surfaces and low-sample handling points
- `jq -r '.edges[] | select(.source_module=="evaluation" or .target_module=="evaluation") ...' docs/audit/data/DEPENDENCY_EDGES.json`
  - Confirmed evaluation cross-module edges and import types

### Test verification run

- Command:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/test_evaluation_engine.py tests/test_evaluation_metrics.py tests/test_evaluation_slicing.py tests/test_evaluation_fragility.py tests/test_evaluation_ml_diagnostics.py tests/test_spec_v01_uncertainty_slicing.py`
- Result: **99 passed in 3.73s**

---

## Acceptance Criteria Status

1. **100% line coverage across all 8 files:** **PASS**
2. **Time-order/embargo-dependent diagnostics validated:** **PASS (with caveat F-03 on config lookback wiring)**
3. **Calibration contracts with models validated:** **PASS**
4. **Evaluation report fields stable and accurately derived:** **PARTIAL FAIL** (F-01, F-02, F-05)

---

## Final Assessment

Subsystem 07 is structurally sound and contract-compatible with backtest/model calibration interfaces, but it does **not** fully satisfy report-correctness expectations under edge conditions. The highest-priority fixes are fragility net-zero handling (F-01) and NaN-safe metric alignment/date reporting (F-02).
