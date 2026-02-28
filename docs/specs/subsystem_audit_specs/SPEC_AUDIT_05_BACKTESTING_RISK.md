# Feature Spec: Subsystem Audit Spec — Backtesting + Risk (Execution Layer)

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 24-32 hours across 7 tasks

---

## Why

This subsystem is the core simulation and risk-control engine. Defects in execution realism, sizing, drawdown handling, or validation logic can invalidate historical performance claims and promotion decisions.

## What

Create a complete audit spec for Subsystem `backtesting_risk` with exhaustive line coverage, execution-contract verification, and downstream artifact/schema stability checks.

## Constraints

### Must-haves
- Audit all `28` files (`13,132` lines).
- Validate execution realism (costs/fills/participation) as a non-negotiable correctness contract.
- Validate `validation/preconditions.py` contract shared with model training.
- Validate result file schema stability consumed by API/evaluation/autopilot/kalshi.

### Must-nots
- No tuning changes to execution/risk parameters during audit.
- No acceptance of undocumented schema drift in `BacktestResult` outputs.
- No unresolved `P0`/`P1` findings in `backtest/engine.py`, `backtest/execution.py`, `risk/position_sizer.py`, `risk/portfolio_risk.py`, `risk/drawdown.py`, `validation/preconditions.py`.

### Out of scope
- Model architecture retraining strategy.
- Frontend rendering behavior.

## Current State

Subsystem details are in [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md), [DEPENDENCY_MATRIX.md](/Users/justinblaise/Documents/quant_engine/docs/audit/DEPENDENCY_MATRIX.md), and [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/INTERFACE_CONTRACTS.yaml).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `backtest/engine.py` |     2488 | 43 | 227 |
| `risk/position_sizer.py` |     1254 | 17 | 78 |
| `backtest/validation.py` |     1074 | 38 | 71 |
| `backtest/execution.py` |      936 | 15 | 50 |
| `risk/portfolio_risk.py` |      885 | 17 | 89 |
| `backtest/cost_calibrator.py` |      685 | 7 | 59 |
| `backtest/advanced_validation.py` |      665 | 15 | 40 |
| `risk/stress_test.py` |      547 | 11 | 29 |
| `risk/attribution.py` |      368 | 6 | 23 |
| `risk/covariance.py` |      355 | 12 | 13 |
| `risk/metrics.py` |      320 | 5 | 22 |
| `risk/factor_monitor.py` |      300 | 6 | 15 |
| `risk/universe_config.py` |      295 | 6 | 45 |
| `risk/stop_loss.py` |      293 | 6 | 20 |
| `risk/portfolio_optimizer.py` |      279 | 8 | 17 |
| `risk/factor_exposures.py` |      268 | 6 | 14 |
| `risk/drawdown.py` |      260 | 6 | 20 |
| `backtest/null_models.py` |      296 | 11 | 24 |
| `backtest/cost_stress.py` |      221 | 6 | 22 |
| `risk/factor_portfolio.py` |      220 | 4 | 10 |
| `risk/cost_budget.py` |      221 | 5 | 12 |
| `backtest/optimal_execution.py` |      200 | 5 | 14 |
| `risk/constraint_replay.py` |      197 | 4 | 10 |
| `backtest/adv_tracker.py` |      190 | 4 | 7 |
| `backtest/survivorship_comparison.py` |      163 | 5 | 6 |
| `validation/preconditions.py` |       71 | 2 | 6 |
| `backtest/__init__.py` |       15 | 0 | 0 |
| `risk/__init__.py` |       11 | 0 | 0 |

### Existing patterns to follow
- `backtest/engine.py` integrates execution, risk, and validation via mixed top-level/lazy imports.
- Risk managers are shared across backtest and autopilot paper trading paths.
- Result artifacts (`summary.json`, `trades.csv`) are consumed as interfaces by other layers.

### Configuration
- Core contracts/invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).
- High-risk boundaries include: `backtest_to_regime_shock_1`, `backtest_to_regime_uncertainty_2`, `backtest_to_risk_3`, `backtest_to_config_19`, `backtest_to_validation_20`, `autopilot_to_backtest_31`, `kalshi_to_backtest_33`, `api_to_backtest_41`.

## Tasks

### T1: Full ledger + artifact schema baseline

**What:** Build complete line ledger and snapshot output schemas.

**Files:**
- `backtest/engine.py`
- `backtest/execution.py`
- `validation/preconditions.py`

**Implementation notes:**
- Capture current `BacktestResult` shape and output file column/schema expectations.
- Map precondition enforcement points in both backtest and training consumers.

**Verify:**
```bash
jq -r '.subsystems.backtesting_risk.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Execution realism pass

**What:** Audit fills, slippage, spread, impact, and participation controls.

**Files:**
- `backtest/engine.py`
- `backtest/execution.py`
- `backtest/optimal_execution.py`

**Implementation notes:**
- Validate shock-mode policy and cost model interactions.
- Confirm assumptions are explicit, bounded, and testable.

**Verify:**
```bash
rg -n "slippage|impact|spread|participation|fill|execution|shock" backtest/engine.py backtest/execution.py backtest/optimal_execution.py
```

---

### T3: Risk manager behavior pass

**What:** Audit sizing, drawdown, stop-loss, and portfolio constraints.

**Files:**
- `risk/position_sizer.py`
- `risk/drawdown.py`
- `risk/portfolio_risk.py`

**Implementation notes:**
- Confirm uncertainty modifiers and regime multipliers are deterministic.
- Validate threshold transitions and hard-stop behaviors.

**Verify:**
```bash
rg -n "kelly|drawdown|stop|constraint|vol|uncertainty|regime" risk/position_sizer.py risk/drawdown.py risk/portfolio_risk.py
```

---

### T4: Statistical validation pass

**What:** Audit walk-forward, embargo, CPCV, SPA, PBO, and DSR logic.

**Files:**
- `backtest/validation.py`
- `backtest/advanced_validation.py`
- `backtest/null_models.py`

**Implementation notes:**
- Confirm time-ordering, purge/embargo, and null-model assumptions.
- Validate outputs consumed by evaluation and promotion gates.

**Verify:**
```bash
rg -n "walk_forward|embargo|purged|cpcv|spa|pbo|deflated_sharpe|monte" backtest/validation.py backtest/advanced_validation.py
```

---

### T5: Shared contract pass (`validation/preconditions.py`)

**What:** Validate execution contract consistency between backtest and training.

**Files:**
- `validation/preconditions.py`
- `backtest/engine.py`
- `models/trainer.py` (read-only consumer check)

**Implementation notes:**
- Confirm RET/LABEL/PX assumptions are identical in both paths.
- Validate strict-vs-soft enforcement behavior.

**Verify:**
```bash
rg -n "enforce_preconditions|RET_TYPE|LABEL_H|PX_TYPE|ENTRY_PRICE_TYPE" validation/preconditions.py backtest/engine.py models/trainer.py
```

---

### T6: Boundary compatibility pass

**What:** Validate contracts with autopilot/kalshi/api/evaluation consumers.

**Files:**
- `backtest/engine.py`
- `backtest/validation.py`
- `backtest/advanced_validation.py`

**Implementation notes:**
- Confirm consumer expectations of return fields and metric names.
- Validate no hidden behavior drift in lazy import paths.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="backtest" or .target_module=="backtest" or .source_module=="risk" or .target_module=="risk") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 140
```

---

### T7: Findings and closure

**What:** Publish severity-ranked findings, unresolved risks, and required tests.

**Files:**
- All subsystem files.

**Implementation notes:**
- Require explicit disposition of every critical hotspot file.

**Verify:**
```bash
# Manual gate: 13132/13132 lines reviewed; all HIGH boundaries dispositioned
```

## Validation

### Acceptance criteria
1. 100% line coverage across all 28 files.
2. Execution realism assumptions are validated and documented.
3. Shared execution contract (`validation/preconditions.py`) is consistent across backtest and training.
4. Output artifact schemas are verified against all known consumers.

### Verification steps
```bash
jq -r '.subsystems.backtesting_risk.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="backtest" or .source_module=="risk" or .target_module=="backtest" or .target_module=="risk") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Documentation-only rollback: revert this file if subsystem grouping changes.

---

## Notes

This is the largest and most behavior-critical subsystem in the 10-spec plan, and should remain a top-priority audit domain.
