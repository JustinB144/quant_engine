# SPEC_AUDIT_FIX_34: Model Prediction Cross-Boundary Contract Fixes

**Priority:** CRITICAL — Predictor hard-suppression bypasses configurable regime policy; walk-forward evaluates stale registry model; three callers dereference None on trainer rejection.
**Scope:** `models/predictor.py`, `autopilot/engine.py`, `models/trainer.py`, `run_train.py`, `run_retrain.py`, `api/orchestrator.py`, `api/services/backtest_service.py`
**Estimated effort:** 5–6 hours
**Depends on:** SPEC_14 T1 (regime ID correction must land first so the suppression target is correct before reworking the suppression mechanism)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 6 (Model Training & Prediction) surfaced cross-boundary contract failures that existing SPEC_14 does not address. SPEC_14 T1 corrects *which* regime is suppressed (2→3) but does not fix the fundamental interaction where the predictor hard-zeros confidence before the backtest engine evaluates `REGIME_TRADE_POLICY`, making the policy's high-confidence override permanently unreachable. A walk-forward evaluation bug in autopilot can load a stale registry model instead of the freshly-trained fold artifact. Three entry points crash with `AttributeError` when `train_ensemble()` returns a rejected (None) global model. A wrong-module import silently hides a config drift.

### Cross-Audit Reconciliation

| Finding | Auditor 1 | Auditor 2 | Disposition |
|---------|-----------|-----------|-------------|
| Regime policy bypass | — | F-01 (CRITICAL) | **NEW** — SPEC_14 T1 fixes the ID, not the interaction |
| Walk-forward version mismatch | — | F-02 (CRITICAL) | **NEW** — not in any existing spec |
| Trainer None dereference | — | F-07 (MEDIUM) | **NEW** — upgraded to HIGH; crash in 3 callers |
| RETRAIN_MAX_DAYS wrong import | F-05 (LOW) | F-09 (LOW) | **NEW** — both auditors agree; silent fallback |

---

## Tasks

### T1: Decouple Regime Suppression From Predictor Hard-Zero

**Problem:** `predictor.py:411-418` unconditionally sets `confidence = 0.0` for the suppressed regime. The backtest engine at `engine.py:910-923` implements `REGIME_TRADE_POLICY` with a high-confidence override (`min_confidence` threshold) that allows select trades through even in disabled regimes. Because the predictor zeros confidence *before* the engine evaluates the policy, the override can never fire — the configurable policy is effectively dead code.

**Files:** `models/predictor.py`, `backtest/engine.py`

**Root cause:** Suppression is implemented in two places with incompatible semantics: the predictor suppresses absolutely (confidence=0), while the engine suppresses conditionally (via policy thresholds). The predictor suppression preempts the engine's nuanced policy.

**Implementation:**
1. Remove the hard confidence-zero from the predictor. Instead, flag suppressed rows without altering confidence:
   ```python
   # predictor.py — replace lines 411-418
   # ── Regime suppression flag ──
   # Mark rows in the suppressed regime but preserve original confidence
   # so downstream consumers (backtest engine, live trader) can apply
   # REGIME_TRADE_POLICY with the high-confidence override intact.
   suppress_regime = 3  # high_volatility (canonical; after SPEC_14 T1)
   regime_vals = regimes.reindex(features.index).fillna(-1).astype(int).values
   regime_suppress_mask = regime_vals == suppress_regime
   result["regime_suppressed"] = regime_suppress_mask
   # DO NOT zero confidence here — let the backtest engine / live trader
   # apply REGIME_TRADE_POLICY gating with its min_confidence override.
   ```
2. Verify that the backtest engine at `engine.py:916-923` already handles the `REGIME_TRADE_POLICY` gating correctly (it does — `enabled=False` with `min_confidence > 0` allows high-confidence trades through).
3. Verify that `autopilot/paper_trader.py` also applies `REGIME_TRADE_POLICY` if it consumes predictions directly. If not, add the same gating logic there.
4. If live trading code reads `regime_suppressed` and skips trades, ensure it also checks the policy override.

**Acceptance:** A regime-3 prediction with `confidence=0.85` and `REGIME_TRADE_POLICY[3]["min_confidence"]=0.80` is NOT suppressed — the high-confidence override fires. A regime-3 prediction with `confidence=0.50` IS suppressed.

---

### T2: Fix Walk-Forward Fold Version Resolution in Autopilot

**Problem:** `autopilot/engine.py:504-511` trains walk-forward folds with `versioned=False`, which writes flat model files to `MODEL_DIR`. At line 519, it loads the fold model with `EnsemblePredictor(version="latest")`. The predictor's `_resolve_model_dir()` at `predictor.py:94-100` checks the version registry first — if a registry exists from any previous training run, it resolves to the *registry* latest (a different model), not the just-trained flat files. The OOS evaluation for that fold uses a stale or wrong model.

**Files:** `autopilot/engine.py`, `models/predictor.py`

**Implementation:**
1. When training walk-forward folds, use a temporary isolated model directory to prevent registry contamination:
   ```python
   # autopilot/engine.py — in the walk-forward fold loop
   import tempfile
   from pathlib import Path

   with tempfile.TemporaryDirectory(prefix="wf_fold_") as fold_dir:
       fold_model_dir = Path(fold_dir)
       trainer = ModelTrainer(model_dir=fold_model_dir)
       trainer.train_ensemble(
           features=train_features,
           targets=train_targets,
           regimes=train_regimes,
           regime_probabilities=train_regime_probs,
           horizon=self.horizon,
           verbose=False,
           versioned=False,
       )
       # Load from the isolated fold directory — no registry can interfere
       fold_predictor = EnsemblePredictor(
           horizon=self.horizon,
           version="latest",
           model_dir=fold_model_dir,
       )
   ```
2. If `EnsemblePredictor` does not accept a `model_dir` parameter, add one as an optional override:
   ```python
   # predictor.py __init__
   def __init__(self, horizon: int = 10, version: str = "latest",
                model_dir: Optional[Path] = None):
       self.model_dir = model_dir or Path(MODEL_DIR)
   ```
3. Add a regression test: train two models (A with high R², B with negative R²), populate the registry with B, then run a walk-forward fold training A. Verify the fold predictor loads A, not B.

**Acceptance:** Walk-forward fold evaluation always uses the model trained in that fold, regardless of what version is in the registry.

---

### T3: Guard Trainer Rejection Path in All Callers

**Problem:** `trainer.train_ensemble()` returns `EnsembleResult(global_model=None, ...)` when quality gates reject the model (trainer.py:763-774). Three callers dereference `result.global_model.cv_scores`, `.holdout_correlation`, and `.cv_gap` without checking for None:
- `run_train.py:202-204`
- `run_retrain.py:279-281`
- `api/orchestrator.py:198-209`

All three will crash with `AttributeError: 'NoneType' object has no attribute 'cv_scores'`.

**Files:** `run_train.py`, `run_retrain.py`, `api/orchestrator.py`

**Implementation:**
1. In each caller, add a None guard immediately after `train_ensemble()` returns:
   ```python
   result = trainer.train_ensemble(...)

   if result.global_model is None:
       logger.warning(
           "Training rejected by quality gates for horizon=%d — "
           "no model was saved.",
           horizon,
       )
       # Skip governance, versioning, and metric extraction.
       # Return a failure result to the caller.
       results[str(horizon)] = {
           "status": "rejected",
           "reason": "quality_gates",
       }
       continue  # or return appropriate error
   ```
2. In `run_train.py` (line 202): Wrap the metric extraction block in the guard.
3. In `run_retrain.py` (line 279): Same guard pattern.
4. In `api/orchestrator.py` (line 198): Same guard, and return a 200 response with `"status": "rejected"` (not a 500 error).
5. Add a test: inject quality gate rejection (e.g., negative holdout R²) and verify each caller handles it gracefully without crash.

**Acceptance:** `run_train.py`, `run_retrain.py`, and `api/orchestrator.py` handle `global_model=None` without crash. Warning is logged. Downstream logic (governance, versioning) is skipped.

---

### T4: Fix RETRAIN_MAX_DAYS Wrong-Module Import

**Problem:** `api/services/backtest_service.py:106` imports `RETRAIN_MAX_DAYS` from `models/retrain_trigger.py`, but this constant does not exist there — it lives in `config.py:119`. A `try/except ImportError` silently catches the failure and falls back to a hardcoded `30`. This hides the config drift: if an operator changes `RETRAIN_MAX_DAYS` in `config.py`, the backtest service still uses 30.

**Files:** `api/services/backtest_service.py`

**Implementation:**
1. Fix the import source:
   ```python
   # OLD (backtest_service.py:106):
   from quant_engine.models.retrain_trigger import RETRAIN_MAX_DAYS
   # NEW:
   from quant_engine.config import RETRAIN_MAX_DAYS
   ```
2. Remove the `try/except ImportError` fallback — the constant must be available from config.
3. Verify no other files import this constant from the wrong module:
   ```bash
   grep -r "from.*retrain_trigger.*import.*RETRAIN_MAX_DAYS" .
   ```

**Acceptance:** `backtest_service.py` imports `RETRAIN_MAX_DAYS` from `config.py`. No silent fallback. Changing the config value is immediately reflected in the backtest service.

---

## Verification

- [ ] Run `pytest tests/ -k "model or predictor or trainer or autopilot"` — all pass
- [ ] Verify regime-3 high-confidence trades pass through when policy override allows
- [ ] Verify walk-forward fold evaluations use fold-trained model, not registry
- [ ] Verify `run_train.py` with quality-gate rejection does not crash
- [ ] Verify `run_retrain.py` with quality-gate rejection does not crash
- [ ] Verify `api/orchestrator.py` with quality-gate rejection returns graceful error
- [ ] Verify `backtest_service.py` reads `RETRAIN_MAX_DAYS` from `config.py`
