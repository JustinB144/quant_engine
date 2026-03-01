# SPEC_AUDIT_FIX_06: Autopilot Reliability, Integration & State Management Fixes

**Priority:** HIGH
**Scope:** `autopilot/` — `paper_trader.py`, `registry.py`, `engine.py`, `promotion_gate.py`, `meta_labeler.py`, `strategy_allocator.py`
**Estimated effort:** 4–5 hours
**Depends on:** SPEC_05 (critical fixes land first)
**Blocks:** Nothing

---

## Context

The autopilot subsystem has numerous P1-P2 reliability issues: non-atomic state persistence that can corrupt on crash, a configured-but-never-evaluated transition-drawdown gate, stale entry prices distorting turnover calculations, cost feedback clipping that biases the calibrator, promotion registry churn that destabilizes active strategies, fail-open exception handling that hides degradation, and several dead code / config drift issues.

---

## Tasks

### T1: Make State Persistence Atomic

**Problem:** `registry.py:48-49` and `paper_trader.py:249-250` write JSON directly to the state file. A crash during serialization corrupts the file, and there is no file locking for concurrent cycles.

**Files:** `autopilot/registry.py`, `autopilot/paper_trader.py`

**Implementation:**
1. Create a shared utility `autopilot/_atomic_write.py`:
   ```python
   import json, os, tempfile
   def atomic_json_write(path: Path, payload: dict, **kwargs):
       tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
       try:
           with os.fdopen(tmp_fd, "w") as f:
               json.dump(payload, f, **kwargs)
           os.replace(tmp_path, str(path))
       except:
           os.unlink(tmp_path)
           raise
   ```
2. Replace `json.dump` + `open` in `registry._save()` and `paper_trader._save_state()` with `atomic_json_write()`.
3. In `_load_state()` / `_load()`, add JSON decode error handling with backup recovery:
   ```python
   try:
       return json.load(f)
   except json.JSONDecodeError:
       logger.error("Corrupt state file %s, attempting backup", path)
       backup = path.with_suffix(".bak")
       if backup.exists():
           return json.load(open(backup))
       return default_state
   ```
4. Before each atomic write, copy current file to `.bak`.

**Acceptance:** Simulated crash during write (kill process mid-write) does not corrupt state. On next startup, either current or backup state is loaded.

---

### T2: Activate Transition-Drawdown Gate in Promotion

**Problem:** `promotion_gate.py:35,90,120` imports and stores `max_transition_drawdown` but never uses it in any decision logic. Strategies with catastrophic regime-transition losses pass promotion unchecked.

**File:** `autopilot/promotion_gate.py`

**Implementation:**
1. In `evaluate()`, after existing checks, add:
   ```python
   # Check transition drawdown if backtest results contain regime transition metrics
   if hasattr(candidate, 'transition_max_drawdown') or 'transition_max_drawdown' in metrics:
       transition_dd = metrics.get('transition_max_drawdown', 0.0)
       if abs(transition_dd) > self.max_transition_drawdown:
           violations.append(f"Transition drawdown {transition_dd:.1%} exceeds limit {self.max_transition_drawdown:.1%}")
   ```
2. If the backtest result doesn't include transition-specific drawdown, compute it from the equity curve + regime labels: filter equity to regime-transition windows (±3 bars around regime changes), compute max drawdown in those windows.
3. Add to the gate summary output.

**Acceptance:** A strategy with 25% drawdown during regime transitions is rejected when `max_transition_drawdown=0.20`.

---

### T3: Fix Turnover Anchoring With Stale Entry Prices

**Problem:** `engine.py:1284` uses `entry_price` as fallback for `last_price` when computing portfolio weights for the optimizer's turnover penalty. Entry prices can be days/weeks old, systematically underestimating current position values.

**Files:** `autopilot/engine.py`, `autopilot/paper_trader.py`

**Implementation:**
1. In `paper_trader._save_state()`, add `last_price` to each position record, updated each cycle with current market data.
2. In `paper_trader._execute_trade()` (or wherever positions are updated), set `pos["last_price"] = current_market_price`.
3. In `engine.py:_get_current_portfolio_weights()`, require `last_price`:
   ```python
   price = float(pos.get("last_price", 0))
   if price <= 0:
       logger.warning("Missing last_price for %s, skipping from optimizer", pos.get("permno"))
       continue
   ```
4. Remove the `entry_price` fallback.

**Acceptance:** Portfolio weights for the optimizer use prices from the current or most recent cycle, not from entry.

---

### T4: Remove Cost Feedback Clipping Bias

**Problem:** `paper_trader.py:498,509` clips `actual_cost_bps` to `max(0.0, ...)`, preventing negative costs (price improvement) from reaching the cost calibrator. This creates systematic upward bias in measured cost surprise.

**File:** `autopilot/paper_trader.py`

**Implementation:**
1. Remove the `max(0.0, ...)` wrapper at both lines 498 and 509:
   ```python
   # OLD: actual_cost_bps=float(max(0.0, actual_cost))
   # NEW:
   actual_cost_bps=float(actual_cost)
   ```
2. Verify `cost_calibrator.py` handles negative actual costs correctly. Check `record_trade()` and `compute_cost_surprise()` for any assumptions about non-negative costs. Fix if needed.
3. Add a log message when actual cost is negative (price improvement) for observability.

**Acceptance:** Negative actual costs (price improvement) flow through to the calibrator. Cost surprise calculations reflect true execution quality.

---

### T5: Fix Promotion Registry Churn

**Problem:** `registry.py:92,65-68` replaces the entire active strategy list each cycle. Incumbents disappear immediately if omitted from the current candidate set.

**File:** `autopilot/registry.py`

**Implementation:**
1. Track incumbent strategies separately. Add a `consecutive_failures` counter per strategy.
2. On each cycle:
   - Strategies that pass re-evaluation: keep active, reset failure counter.
   - Strategies that fail re-evaluation: increment failure counter.
   - Strategies NOT in current candidate set: increment failure counter (they weren't evaluated, not necessarily bad).
   - Only remove a strategy when `consecutive_failures >= PROMOTION_GRACE_CYCLES` (default 3).
3. Add config constant `PROMOTION_GRACE_CYCLES = 3`.
4. Log when a strategy enters grace period and when it's finally removed.

**Acceptance:** An active strategy missing from one candidate set survives 3 cycles before removal. A strategy that consistently fails is removed after the grace period.

---

### T6: Fix Meta-Label Threshold Mismatch

**Problem:** `engine.py:1001` uses `META_LABELING_CONFIDENCE_THRESHOLD` from global config, but the loaded model at `meta_labeler.py:439-441` stores its own `confidence_threshold` that may differ (e.g., tuned during training).

**File:** `autopilot/engine.py`

**Implementation:**
1. After loading the meta-labeler model, read its stored threshold:
   ```python
   threshold = self.meta_labeler.confidence_threshold
   ```
2. Replace all references to `META_LABELING_CONFIDENCE_THRESHOLD` in engine.py's meta-label filtering logic with the model's own threshold.
3. Log the threshold being used for auditability.

**Acceptance:** A model trained with threshold 0.65 uses 0.65 for filtering, regardless of the global config value.

---

### T7: Fix MetaLabelingModel.save() Double Write

**Problem:** `meta_labeler.py:404-406` unconditionally writes to `meta_labeler_current.joblib` in addition to the requested filepath, creating unexpected side effects.

**File:** `autopilot/meta_labeler.py`

**Implementation:**
1. Remove the unconditional write to `meta_labeler_current.joblib` from `save()`.
2. Add a separate public method:
   ```python
   def update_current_pointer(self):
       """Write current model state to the default 'current' path."""
       current_path = self._model_dir / "meta_labeler_current.joblib"
       self.save(current_path)
   ```
3. Call `update_current_pointer()` explicitly in the engine after a successful training cycle (not buried inside `save()`).

**Acceptance:** Calling `save("/tmp/experimental_model.joblib")` does NOT overwrite `meta_labeler_current.joblib`.

---

### T8: Wire StrategyAllocator Into Runtime

**Problem:** `strategy_allocator.py:12-15` implements regime-aware parameter allocation but is never imported or used in the autopilot runtime. All strategies use fixed parameters regardless of market regime.

**File:** `autopilot/engine.py`

**Implementation:**
1. In `AutopilotEngine.__init__()`, instantiate `StrategyAllocator`.
2. Before each strategy evaluation, query the allocator for regime-specific parameters:
   ```python
   regime_params = self.allocator.get_regime_profile(regime=current_regime, regime_confidence=confidence)
   ```
3. Apply regime-specific overrides to the strategy candidate's sizing and risk parameters.
4. If this is too invasive for this spec, at minimum: wire it in and log what the allocator WOULD recommend, without applying it yet. Mark as "shadow mode" for validation.

**Acceptance:** (Shadow mode) StrategyAllocator output appears in logs for each cycle. (Full mode) Strategy parameters change based on regime.

---

## Verification

- [ ] Run `pytest tests/ -k "autopilot or paper_trader or promotion or meta_label or registry"` — all pass
- [ ] Simulate crash during state save — verify recovery from backup
- [ ] Verify transition-drawdown gate rejects appropriate candidates
- [ ] Verify promotion registry respects grace period
- [ ] Verify negative costs flow through to calibrator
