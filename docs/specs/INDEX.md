# Quant Trading Engine - Implementation Specs Index

**Last Updated:** 2026-02-26
**Total Specs:** 10
**Status:** 6 Complete (SPEC 1–6) + 4 New Complete (SPEC 7–10)

---

## Complete Spec List

### Foundation & Infrastructure (SPEC 1–2)

| Spec | Title | Status | Effort | Focus |
|------|-------|--------|--------|-------|
| **1** | Foundational Hardening | ✓ Complete | 80h | Core robustness, logging, error handling, versioning |
| **2** | Structural Features | ✓ Complete | 90h | Spectral analysis, SSA decomposition, BOCPD changepoints |

### State & Signal Processing (SPEC 3–5)

| Spec | Title | Status | Effort | Focus |
|------|-------|--------|--------|-------|
| **3** | Structural State Layer | ✓ Complete | 110h | State manifold learning, anomaly detection |
| **4** | Signal Meta-Labeling | ✓ Complete | 70h | ML labels, confidence scoring, target engineering |
| **5** | Risk Governor | ✓ Complete | 100h | Capital allocation, position gating, risk limits |

### Execution & Operations (SPEC 6–10)

| Spec | Title | Status | Effort | Focus |
|------|-------|--------|--------|-------|
| **6** | Execution Layer | ✓ Complete | 95h | Order routing, slippage modeling, FIX protocol |
| **7** | Portfolio Layer (NEW) | ✓ Complete | 120h | Regime-conditioned constraints, factor exposure, sizing backoff |
| **8** | Evaluation Layer (NEW) | ✓ Complete | 140h | Regime slicing, walk-forward with embargo, fragility metrics |
| **9** | Health System (NEW) | ✓ Complete | 100h | Information Ratio, survivorship bias, health feedback |
| **10** | Regime Detection (NEW) | ✓ Complete | 130h | Ensemble voting, uncertainty integration, online updates |

---

## Quick Navigation

### By Component

**Portfolio & Risk:**
- [SPEC 7: Portfolio Layer](./SPEC_07_PORTFOLIO_LAYER.md) - Regime-conditioned constraints, factor exposure
- [SPEC 5: Risk Governor](./SPEC_05_RISK_GOVERNOR.md) - Capital allocation, position gating

**Market Regime:**
- [SPEC 10: Regime Detection](./SPEC_10_REGIME_DETECTION.md) - Ensemble voting, uncertainty, consensus
- [SPEC 2: Structural Features](./SPEC_02_STRUCTURAL_FEATURES.md) - Spectral, SSA, BOCPD inputs to regime

**Performance & Health:**
- [SPEC 8: Evaluation Layer](./SPEC_08_EVALUATION_LAYER.md) - Regime slicing, walk-forward, fragility
- [SPEC 9: Health System](./SPEC_09_HEALTH_SYSTEM.md) - Information Ratio, health feedback loop

**Execution & Trading:**
- [SPEC 6: Execution Layer](./SPEC_06_EXECUTION_LAYER.md) - Order routing, slippage, FIX
- [SPEC 4: Signal Meta-Labeling](./SPEC_04_SIGNAL_META_LABELING.md) - ML labels, confidence

**Core Infrastructure:**
- [SPEC 1: Foundational Hardening](./SPEC_01_FOUNDATIONAL_HARDENING.md) - Logging, versioning
- [SPEC 3: Structural State Layer](./SPEC_03_STRUCTURAL_STATE_LAYER.md) - Anomaly detection

### By Implementation Order

**Immediate Priority (NEW):**
1. [SPEC 7: Portfolio Layer](./SPEC_07_PORTFOLIO_LAYER.md) - 120h (foundational)
2. [SPEC 10: Regime Detection](./SPEC_10_REGIME_DETECTION.md) - 130h (blocks evaluation & health)
3. [SPEC 8: Evaluation Layer](./SPEC_08_EVALUATION_LAYER.md) - 140h (validates constraints)
4. [SPEC 9: Health System](./SPEC_09_HEALTH_SYSTEM.md) - 100h (completes monitoring)

**Existing (Reference):**
- [SPEC 1–6](./SPEC_01_FOUNDATIONAL_HARDENING.md) - Previous implementation phases

---

## Configuration Files Reference

### New Config Files (SPEC 7–10)

| File | Spec | Purpose |
|------|------|---------|
| `/mnt/quant_engine/config/universe.yaml` | 7 | Sector mapping, liquidity tiers, borrowability |
| `/mnt/quant_engine/config/health.yaml` | 9 | Domain weights, check thresholds, alerts |
| `/mnt/quant_engine/config/validation.yaml` | 8 | Walk-forward params, slicing config, ML diagnostics |
| `/mnt/quant_engine/config/regime.yaml` | 10 | HMM, ensemble, consensus, uncertainty thresholds |

---

## Test Coverage Goals

| Spec | Unit Tests | Integration | Coverage |
|------|-----------|-------------|----------|
| 7 | test_portfolio_risk_regime.py | test_portfolio_integration.py | > 85% |
| 8 | test_slicing.py, test_ic_decay.py, ... | test_evaluation_engine.py | > 85% |
| 9 | test_health_information_ratio.py, ... | (integrated into health_service) | > 85% |
| 10 | test_jump_model_validation.py, ... | test_regime_detection_integration.py | > 85% |

---

## Critical Paths & Dependencies

```
SPEC 10 (Regime Detection)
  ├─ regime_state, uncertainty → SPEC 7 (constraint multipliers)
  ├─ regime labels → SPEC 8 (performance slicing)
  └─ regime-specific thresholds → SPEC 9 (health checks)

SPEC 7 (Portfolio Layer)
  ├─ constraint multipliers ← SPEC 10 (regime-aware)
  └─ sizing backoff → SPEC 9 (health feedback)

SPEC 8 (Evaluation Layer)
  ├─ regime slicing ← SPEC 10
  └─ validation → SPEC 7 (constraint robustness)

SPEC 9 (Health System)
  ├─ regime thresholds ← SPEC 10
  ├─ constraint utilization ← SPEC 7
  └─ alert → SPEC 7 (sizing adjustment)
```

---

## Key Artifacts

### Documents
- **Summary:** [SPECS_SUMMARY.md](./SPECS_SUMMARY.md) - Overview of all 4 new specs
- **This Index:** [INDEX.md](./INDEX.md) - Navigation and reference

### Code Locations
- **Config:** `/mnt/quant_engine/config/*.yaml`
- **Engine modules:** `/mnt/quant_engine/engine/`
- **Tests:** `/mnt/quant_engine/tests/test_*.py`
- **Guides:** `/mnt/quant_engine/docs/<feature>_guide.md`

---

## Implementation Timeline (Estimated)

```
Weeks 1–4:   SPEC 7 (Portfolio Layer) - 120h
             └─ universe.yaml, constraint multipliers, factor exposure

Weeks 5–8:   SPEC 10 (Regime Detection) - 130h
             └─ confidence weighting, uncertainty, structural features, consensus

Weeks 9–12:  SPEC 8 (Evaluation Layer) - 140h
             └─ regime slicing, walk-forward, ML diagnostics, fragility

Weeks 13–14: SPEC 9 (Health System) - 100h
             └─ Information Ratio, health history, feedback loop

Total: 14 weeks @ 35h/week = 490 hours
```

---

## Version History

| Date | Event |
|------|-------|
| 2026-02-26 | SPEC 7–10 created (4 new implementation specs) |
| 2026-01-XX | SPEC 1–6 completed (6 previous specs) |

---

## Maintenance & Updates

Each spec includes:
- ✓ Status field (Draft → In Progress → Complete)
- ✓ Author attribution
- ✓ Estimated effort (hours and task count)
- ✓ Change log (in spec document Notes section)

For updates, modify the spec document directly and increment version in Status field.

---

## Questions & Support

Refer to the individual spec documents for:
- **Architecture details:** See "Current State" section
- **Implementation tasks:** See "Tasks" section (T1–T8 per spec)
- **Validation criteria:** See "Validation" section
- **Rollback procedures:** See "Rollback plan" subsection
- **Design rationale:** See "Notes" section

Each spec is self-contained and can be implemented independently, though they have dependencies as shown in the Critical Paths diagram.

---

*Last generated: 2026-02-26 by Claude Code*
