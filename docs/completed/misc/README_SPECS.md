# Quant Engine Implementation Specifications

This directory contains three comprehensive feature specifications for the quant trading engine. Each spec follows a rigorous template covering business rationale, technical design, implementation tasks, validation criteria, and rollback plans.

## Specifications

### SPEC 1: Foundational Hardening — Truth Layer
**File:** `SPEC_01_FOUNDATIONAL_HARDENING.md`  
**Status:** Draft  
**Estimated Effort:** 85 hours across 6 tasks  
**Focus:** Data integrity, leakage prevention, cost stress, cache fixes

**Objective:** Lock down foundational execution contracts before any modeling/backtesting. Prevent silent data corruption, feature leakage, missing cost stress, and cache staleness.

**Key Tasks:**
- T1: Global preconditions (RET_TYPE, PX_TYPE, LABEL_H, ENTRY_PRICE) with validation
- T2: Data integrity preflight (fail-fast on corrupt OHLCV)
- T3: Leakage tripwires (runtime causality enforcement, time-shift tests)
- T4: Null model baselines (random, zero, momentum)
- T5: Cost stress sweep (0.5x–5.0x transaction cost multipliers)
- T6: Cache staleness fix (trading calendar, not date.today())

**Critical Files Modified:**
- `config.py` (add execution preconditions)
- `data/quality.py` (fail-fast semantics)
- `data/loader.py` (data integrity, cache staleness)
- `features/pipeline.py` (causality enforcement)
- `backtest/engine.py` (validation, null baselines)

**Expected Impact:**
- Prevents models training on corrupted data
- Catches feature leakage before production
- Quantifies execution cost sensitivity
- Eliminates cache staleness on weekends/holidays

---

### SPEC 2: Structural Feature Expansion
**File:** `SPEC_02_STRUCTURAL_FEATURES.md`  
**Status:** Draft  
**Estimated Effort:** 120 hours across 6 tasks  
**Focus:** 26 new features (spectral, SSA, tail, eigenvalue, OT)

**Objective:** Add structural features capturing hidden market dynamics that complement the existing 100 classical indicators.

**Feature Families:**
- **Spectral Features** (T1): HF/LF energy, entropy, dominant frequency (4 features)
- **SSA Features** (T2): Trend strength, singular entropy, noise ratio (4 features)
- **Jump/Tail Features** (T3): Intensity, expected shortfall, vol-of-vol, semi-relative modulus (5 features)
- **Eigenvalue Features** (T4): Concentration, effective rank, correlation stress (4 features)
- **OT Features** (T5): Wasserstein, Sinkhorn divergence (2 features)
- **Integration & Validation** (T6): Pipeline integration, redundancy detection (7 features)

**New Indicator Classes:**
- `indicators/spectral.py`: SpectralAnalyzer (FFT with Hann windowing)
- `indicators/ssa.py`: SSADecomposer (SVD-based decomposition)
- `indicators/tail_risk.py`: TailRiskAnalyzer (jump detection, extreme values)
- `indicators/eigenvalue.py`: EigenvalueAnalyzer (systemic stress)
- `indicators/ot_divergence.py`: OptimalTransportAnalyzer (Wasserstein & Sinkhorn)

**Technical Highlights:**
- Proper FFT windowing, SVD with rank estimation
- Numerical stability (log-domain Sinkhorn, regularized covariance)
- FEATURE_METADATA causality tags enforced at runtime
- Winsorizing to [1st, 99th] percentile per existing pipeline

**Expected Impact:**
- +50–200 bps Sharpe on regime-sensitive strategies
- Better anticipation of volatility regime changes
- Improved tail-risk identification

---

### SPEC 3: Structural State Layer — BOCPD + HMM Enhancement + Unified Shock Vector
**File:** `SPEC_03_STRUCTURAL_STATE_LAYER.md`  
**Status:** Draft  
**Estimated Effort:** 95 hours across 4 tasks  
**Focus:** BOCPD, HMM audit, ShockVector, integration

**Objective:** Provide real-time regime intelligence with structured shock metadata. Replace retrospective HMM with online changepoint detection.

**Key Tasks:**
- T1: BOCPD implementation (Bayesian Online Change-Point Detection)
  - Gaussian likelihood, run-length posterior
  - Detects regime transitions within 1 bar
  - Performance: <10ms per bar for 1000 stocks
- T2: HMM observation matrix audit
  - Document all 11 features (currently underdocumented)
  - Validate causality
- T3: Unified shock vector contract
  - `ShockVector` dataclass with version 1.0
  - Combines HMM regime, BOCPD probability, jump flags, structural features
- T4: Integration
  - `RegimeDetector.detect_with_shock_context()` returns `Dict[str, ShockVector]`
  - BacktestEngine consumes ShockVector for risk alerts
  - ModelTrainer uses for regime-aware sample weighting

**New Classes:**
- `regime/bocpd.py`: BOCPDDetector (Adams & MacKay algorithm)
- `regime/shock_vector.py`: ShockVector, ShockVectorValidator

**Technical Highlights:**
- BOCPD: Bayesian online changepoint detection with run-length posterior
- Sufficient statistics model: mean, variance, alpha, beta (normal-inverse-gamma conjugate)
- Student-t predictive distribution for robust likelihood
- Numerical stability via logsumexp, log-domain arithmetic

**Current State Correction:**
- HMM observation matrix already has 11 features (not 4 as claimed)
- Features: returns, vol_20d, NATR, SMA slope, credit spread, breadth, VIX rank, volume regime, momentum, Hurst exponent, cross-asset correlation

**Expected Impact:**
- 2–5 day earlier regime transition detection
- Real-time alerts for earnings/macro shocks
- Regime-aware model weighting reduces overfitting
- Audit trail via versioned ShockVector

---

## Implementation Roadmap

### Phase 1: Foundation (SPEC 1)
**Duration:** 2–3 weeks  
**Resources:** 1 engineer  
**Dependencies:** None

Implement the Truth Layer first to establish foundational guardrails. Highest ROI: prevents downstream errors.

### Phase 2: Features + Regime (SPEC 2 + SPEC 3, parallel)
**Duration:** 4–6 weeks  
**Resources:** 2–3 engineers  
**Dependencies:** Phase 1 (SPEC 1 preconditions)

Expand feature set and enhance regime detection. Can proceed in parallel.

**SPEC 2 Priority:** T3 (jump/tail), T2 (SSA) highest impact
**SPEC 3 Priority:** T1 (BOCPD), T3 (ShockVector) highest impact

### Phase 3: Validation & Integration
**Duration:** 1–2 weeks  
**Resources:** 1–2 engineers  
**Dependencies:** Phases 1 & 2

End-to-end testing, documentation, rollback verification.

---

## Shared Principles

All three specs adhere to these principles:

1. **Backward Compatibility**
   - All changes are additive or feature-flagged
   - Existing configs continue to work with sensible defaults
   - Rollback plans documented for each spec

2. **Numerical Stability**
   - Log-domain computation for expensive operations
   - Regularized covariance, Cholesky verification
   - Winsorizing, clipping, fallbacks for edge cases

3. **Causality & Leakage Prevention**
   - FEATURE_METADATA causality tags enforced at runtime
   - Time-shift tripwire tests in validation
   - Temporal metadata in ShockVector for audit

4. **Comprehensive Validation**
   - Unit tests for each task/module (target: 80%+ coverage)
   - Integration tests for full pipelines
   - Synthetic test cases (known patterns, edge cases)
   - Verification steps for operators

5. **Production-Grade Documentation**
   - Detailed docstrings (parameters, returns, examples)
   - Configuration rationale (STATUS annotations)
   - Mathematical formulas and references
   - Human-readable interpretations

6. **Operability**
   - Logging at INFO/WARNING/ERROR levels
   - JSON serialization for downstream systems
   - Schema versioning for future compatibility
   - Performance metrics (timing, memory)

---

## File Structure

```
docs/completed/feature_specs/
├── SPEC_01_FOUNDATIONAL_HARDENING.md       (39 KB, 952 lines)
├── SPEC_02_STRUCTURAL_FEATURES.md          (52 KB, 1390 lines)
├── SPEC_03_STRUCTURAL_STATE_LAYER.md       (43 KB, 1085 lines)
├── ...
└── SPEC_12_AUTODOC_SYSTEM.md
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Specifications | 3 |
| Total Tasks | 16 |
| Total Lines | 3,427 |
| Total Size | 134 KB |
| Estimated Effort | ~300 hours |
| Expected Sharpe Improvement | +50–200 bps |
| Regime Detection Latency | 2–5 days earlier |

---

## How to Use These Specs

1. **Read the executive summary** (Why, What sections) to understand business rationale
2. **Review Current State** to understand what exists and what's missing
3. **Study Tasks** for implementation details, file paths, and code snippets
4. **Use Validation section** as acceptance criteria and testing roadmap
5. **Refer to Notes** for future extensions and known limitations

Each spec is self-contained but complements the others. All three should be read to understand the full system architecture.

---

## Questions & Clarifications

For questions about specific specs:
- SPEC 1 (data integrity, leakage): See Sections "T2: Data Integrity", "T3: Leakage Tripwires"
- SPEC 2 (features): See "Current State → Existing patterns to follow"
- SPEC 3 (regime): See "Current State Correction" for HMM clarification

---

**Last Updated:** 2026-02-26  
**Author:** Claude Opus 4  
**Status:** Draft (awaiting review & feedback)
