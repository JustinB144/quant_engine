# Implementation Spec Index: SPEC 04, 05, 06

This directory contains three comprehensive implementation specifications for the quant trading engine, covering critical enhancements across the signal pipeline, risk management, and execution layers.

## Quick Navigation

### SPEC 04: Signal Enhancement + Meta-Labeling + Fold-Level Validation
- **File:** `SPEC_04_SIGNAL_META_LABELING.md`
- **Status:** Draft
- **Size:** 424 lines (22 KB)
- **Effort:** 120 hours across 8 tasks
- **Focus:** Improving signal selection, adding meta-labeling, tracking fold-level metrics

**Key Improvements:**
- Cross-sectional top-K selection replacing fixed threshold
- XGBoost meta-labeling for signal reliability prediction
- Fold-level metric granularity in promotion gate
- Fold consistency penalty for regime concentration risk

**Files Modified:** autopilot/engine.py, autopilot/meta_labeler.py (NEW), validation.py, promotion_gate.py, config.py

**Start Here:** Read "Why" section for problem statement, "Tasks" section for implementation roadmap

---

### SPEC 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing
- **File:** `SPEC_05_RISK_GOVERNOR.md`
- **Status:** Draft
- **Size:** 671 lines (32 KB)
- **Effort:** 110 hours across 9 tasks
- **Focus:** Consolidating position sizing, adding uncertainty awareness, implementing budget constraints

**Key Improvements:**
- Unified PositionSizer interface (paper trader + autopilot)
- Uncertainty-aware sizing (signal, regime, drift inputs)
- Shock budget (reserve 5% capital)
- Turnover budget enforcement
- Concentration limits (max 20% per position)
- Regime-conditional blend weights

**Files Modified:** position_sizer.py, paper_trader.py, autopilot/engine.py, config.py

**Start Here:** Read "Why" section for codebase analysis, "Tasks" for detailed implementation

---

### SPEC 06: Execution Layer + Structural State-Aware Costs
- **File:** `SPEC_06_EXECUTION_LAYER.md`
- **Status:** Draft
- **Size:** 738 lines (33 KB)
- **Effort:** 95 hours across 7 tasks
- **Focus:** Improving execution costs, implementing ADV tracking, adding cost calibration

**Key Improvements:**
- Structural state-aware cost multipliers (break_prob, uncertainty, drift, stress)
- Explicit ADV computation with EMA smoothing
- Entry/exit urgency differentiation
- No-trade gate during extreme stress
- Cost calibration per market cap segment
- Volume trend adjustments

**Files Modified:** execution.py, execution/adv_tracker.py (NEW), execution/cost_calibrator.py (NEW), paper_trader.py, autopilot/engine.py, config.py

**Start Here:** Read "Why" section for execution gaps, "Tasks" for phased implementation

---

## Template Structure (All Three Specs Follow)

Each specification includes:

```
# Feature Spec: [Title]
> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** [X hours across Y tasks]

## Why
  - Current state analysis
  - Gaps identified in codebase
  - Why each gap matters

## What
  - High-level solution overview
  - Multi-part breakdown of features
  - Clear problem-solution mapping

## Constraints
  ### Must-haves
  ### Must-nots
  ### Out of scope

## Current State
  ### Key files
  ### Existing patterns to follow
  ### Configuration

## Tasks (T1, T2, T3, ...)
  ### T[N]: [Title]
  **What:** ...
  **Files:** ...
  **Implementation notes:** ...
  **Verify:** ...

## Validation
  ### Acceptance criteria
  ### Verification steps
  ### Rollback plan

## Notes
```

---

## Key Integration Points

### Data Flow (How Specs Work Together)

```
SPEC 04: Signals
└─ Top-K Selection + Meta-Labeling
   └─ Fold-Level Validation
      └─ Promotion Gate (Fold Consistency)
         └─ SPEC 05: Position Sizing
            ├─ Kelly (regime-conditional)
            ├─ Uncertainty Scaling
            └─ Budget Constraints
               └─ SPEC 06: Execution
                  ├─ Structural Costs
                  ├─ ADV Tracking
                  ├─ Urgency-Based Limits
                  └─ Cost Calibration
```

### Dependencies
- **SPEC 04 → 05:** Meta-labeling confidence feeds as signal_uncertainty input
- **SPEC 05 → 06:** Position size target passed to execution layer
- **SPEC 06 → 04:** Realized execution costs inform future cost calibration

---

## Implementation Timeline

| Phase | Spec | Duration | Risk | Validation |
|-------|------|----------|------|------------|
| 1 | SPEC 04 | Weeks 1-3 | Low | >95% test coverage, isolated |
| 2 | SPEC 05 | Weeks 4-6 | Medium | >95% test coverage, P&L±0.1% |
| 3 | SPEC 06 | Weeks 7-8 | Medium | >95% test coverage, cost accuracy |

**Final Validation:** 4-week production A/B test

---

## Configuration Parameters Summary

### SPEC 04 Additions (~8 params)
```python
SIGNAL_TOPK_QUANTILE = 0.70
META_LABELING_ENABLED = True
META_LABELING_CONFIDENCE_THRESHOLD = 0.55
FOLD_CONSISTENCY_PENALTY_WEIGHT = 0.15
# ... plus 4 meta-labeling training params
```

### SPEC 05 Additions (~15 params)
```python
SHOCK_BUDGET_PCT = 0.05
CONCENTRATION_LIMIT_PCT = 0.20
BLEND_WEIGHTS_BY_REGIME = {...}
KELLY_BAYESIAN_PRIOR_ALPHA = 2
UNCERTAINTY_REDUCTION_FACTOR = 0.30
# ... plus other sizing/budget params
```

### SPEC 06 Additions (~18 params)
```python
EXEC_STRUCTURAL_STRESS_ENABLED = True
ADV_EMA_SPAN = 20
EXEC_EXIT_URGENCY_COST_LIMIT_MULT = 1.5
EXEC_NO_TRADE_STRESS_THRESHOLD = 0.95
EXEC_CALIBRATION_ENABLED = True
# ... plus other execution params
```

**Total New Parameters:** 40+

---

## Files Modified Across All Specs

### Core Engine Files
- `autopilot/engine.py` (signals, uncertainty inputs, structural state inputs)
- `position_sizer.py` (unified interface, uncertainty, budgets)
- `paper_trader.py` (unified sizing, urgency inputs)
- `execution.py` (structural costs, urgency, ADV)
- `validation.py` (fold-level metrics)
- `promotion_gate.py` (fold consistency)
- `config.py` (40+ new parameters)

### New Modules
- `autopilot/meta_labeler.py` (~300 lines)
- `execution/adv_tracker.py` (~200 lines)
- `execution/cost_calibrator.py` (~250 lines)

### Testing & Documentation
- `tests/test_signal_meta_labeling.py` (~400 lines)
- `tests/test_position_sizing_unification.py` (~400 lines)
- `tests/test_execution_layer.py` (~500 lines)
- `docs/DESIGN_META_LABELING.md`
- `docs/DESIGN_POSITION_SIZING.md`
- `docs/DESIGN_EXECUTION.md`

---

## Success Criteria Summary

### SPEC 04 Success
- Top-K quantile selection adapts to signal dispersion
- Meta-labeling confidence r ≥ 0.40 with realized accuracy
- Fold consistency penalizes regime concentration
- All backward compatible

### SPEC 05 Success
- No duplicate sizing logic (unified interface)
- Uncertainty inputs reduce size correctly
- Three budgets enforced and logged
- P&L within ±0.1% of baseline

### SPEC 06 Success
- Structural multipliers scale costs [1.0, 3.0] appropriately
- ADV EMA tracking accurate
- Entry vs exit urgency working (exits > entries)
- Cost calibration per market cap segment functional

---

## How to Use These Specs

### For Implementation Team
1. Read through all three "Why" sections to understand codebase gaps
2. Start with SPEC 04 (lowest risk, isolated)
3. Each task has **What/Files/Implementation notes/Verify** for clarity
4. Use acceptance criteria for completion validation
5. Follow rollback plan if issues arise

### For Code Review
1. Check that all tasks marked complete actually implement the spec
2. Verify acceptance criteria satisfied before merging
3. Run >95% test coverage requirement
4. Validate backtest P&L and risk metrics unchanged

### For Operations/Monitoring
1. Review comprehensive logging sections for metrics to monitor
2. Understand configuration parameters and tuning guidance
3. Know the rollback plan for each spec before production
4. Monitor fold consistency, meta-labeling accuracy, budget utilization

---

## Quick Reference: Task Counts

- **SPEC 04:** 8 tasks (120 hours)
  - 3 core features (top-K, meta-labeling, fold metrics)
  - 2 integration tasks (gate, retraining)
  - 2 monitoring/test tasks

- **SPEC 05:** 9 tasks (110 hours)
  - 3 core features (unified interface, uncertainty, budgets)
  - 4 constraint tasks (shock, turnover, concentration, blend weights)
  - 2 integration/test tasks

- **SPEC 06:** 7 tasks (95 hours)
  - 2 core features (structural costs, ADV)
  - 2 algorithmic tasks (urgency, calibration)
  - 1 integration task
  - 2 monitoring/test tasks

**Total:** 23 tasks, 325 hours

---

## Questions? Refer To:

- **Signal selection, validation, meta-labeling** → SPEC 04
- **Position sizing, Kelly, risk budgets, uncertainty** → SPEC 05
- **Execution costs, ADV, urgency, calibration** → SPEC 06
- **Integration between specs** → Data flow diagram above
- **Deployment and rollback** → Each spec's Validation section

---

**Last Updated:** 2026-02-26  
**Status:** Draft  
**Next Review:** Post-implementation validation
