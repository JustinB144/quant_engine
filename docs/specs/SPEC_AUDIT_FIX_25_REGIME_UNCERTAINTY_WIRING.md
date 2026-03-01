# SPEC_AUDIT_FIX_25: Regime Uncertainty Semantic Wiring Fixes

**Priority:** HIGH — Backtest ShockVector uncertainty uses model confidence instead of regime confidence; autopilot uncertainty gate is mathematically neutralized by immediate renormalization.
**Scope:** `backtest/engine.py`, `regime/uncertainty_gate.py`, `autopilot/engine.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The regime uncertainty subsystem has two P1 wiring defects. First (F-02), `backtest/engine.py:866-878` passes `predictions["confidence"]` (model prediction confidence, derived from holdout/CV correlations and regime model reliability metrics in predictor.py:343-390) to `compute_shock_vectors()` as `confidence_series`. But `compute_shock_vectors()` at shock_vector.py:408-415 treats this parameter as *regime* confidence — its comments state "When the **regime detector** is confident (high probability on one state), uncertainty is low" and computes `uncertainty = 1 - confidence`. Model prediction confidence and regime detection confidence are semantically different: model confidence reflects the ML model's expected accuracy; regime confidence reflects posterior probability mass on the dominant regime state. The backtest entry point `run_backtest.py:170-172` correctly extracts regime confidence from the detector, but this is not threaded through to the ShockVector computation.

Second (F-03), `uncertainty_gate.py:123` (`apply_uncertainty_gate`) multiplies all weight vector elements by a single scalar multiplier. `autopilot/engine.py:1668-1675` immediately renormalizes the result to sum to 1.0. For any normalized weight vector **w** where Σwᵢ = 1: scaling by c produces c·**w** with sum = c, and dividing by sum recovers **w** exactly. The gate is mathematically neutralized — it has zero net effect on portfolio allocations.

---

## Tasks

### T1: Pass Regime Confidence (Not Model Confidence) to compute_shock_vectors in Backtest

**Problem:** `backtest/engine.py:866-868` reads `ticker_preds["confidence"]` — the model prediction confidence output from the predictor. This is passed to `compute_shock_vectors()` at line 878 as `confidence_series`. But the function expects regime detection confidence (max posterior probability), not model prediction confidence.

**File:** `backtest/engine.py`

**Implementation:**
1. Extract regime confidence from the predictions DataFrame (the predictor includes `regime_confidence` in its output):
   ```python
   # BEFORE (line 866-868):
   conf_s = (
       ticker_preds["confidence"]
       if "confidence" in ticker_preds.columns else None
   )

   # AFTER — use regime confidence, not model confidence:
   conf_s = (
       ticker_preds["regime_confidence"]
       if "regime_confidence" in ticker_preds.columns
       else ticker_preds["confidence"]  # Fallback to model confidence
       if "confidence" in ticker_preds.columns
       else None
   )
   ```
2. If `regime_confidence` is not available (older predictor output), fall back to `confidence` with a warning:
   ```python
   if "regime_confidence" not in ticker_preds.columns and "confidence" in ticker_preds.columns:
       logger.warning(
           "Using model confidence as ShockVector uncertainty proxy for %s — "
           "regime_confidence column not available. This is semantically imprecise.",
           ticker,
       )
   ```
3. Verify that the predictor's output includes `regime_confidence`. Check `predictor.py` for the column — the predictor receives `regime_confidence` as input (from the detector at `run_backtest.py:172`) and should pass it through or store it in the output DataFrame.
4. If the predictor does not currently include `regime_confidence` in its output, add it:
   ```python
   # In predictor.py, when building result DataFrame:
   if regime_confidence is not None:
       result["regime_confidence"] = regime_confidence
   ```

**Acceptance:** `compute_shock_vectors()` receives regime detection confidence (max posterior probability), not model prediction confidence. ShockVector's `hmm_uncertainty` field reflects regime detection uncertainty. A test verifies that when model confidence is 0.9 but regime confidence is 0.4, the ShockVector shows high uncertainty (≈0.6), not low uncertainty (≈0.1).

---

### T2: Fix Uncertainty Gate to Reduce Total Allocation, Not Just Relative Weights

**Problem:** `uncertainty_gate.py:123` returns `weights * multiplier`. `autopilot/engine.py:1672-1675` renormalizes to sum=1. For any normalized **w**: `(c·w) / sum(c·w) = (c·w) / (c·1) = w`. The gate has zero effect.

**File:** `autopilot/engine.py`

**Implementation:**

The fix must be in the *consumer* (autopilot), not in the gate itself — the gate correctly reduces position sizes, and other consumers (risk/position_sizer) use the multiplier directly without renormalization. The autopilot's renormalization is what breaks the gate.

**Option A (Recommended): Convert multiplier to cash allocation:**

Instead of scaling weights and renormalizing, use the multiplier to determine what fraction of capital should be invested vs held as cash:

```python
# BEFORE (lines 1666-1679):
if current_uncertainty > 0.0:
    gate = UncertaintyGate()
    weights = pd.Series(
        gate.apply_uncertainty_gate(weights.values, current_uncertainty),
        index=weights.index,
    )
    w_sum = weights.sum()
    if abs(w_sum) > 1e-10:
        weights = weights / w_sum

# AFTER:
if current_uncertainty > 0.0:
    gate = UncertaintyGate()
    multiplier = gate.compute_size_multiplier(current_uncertainty)

    # Scale weights to reduce total invested capital.
    # multiplier=0.7 means invest 70% of capital, hold 30% in cash.
    # Weights still sum to multiplier (not 1.0), so the remaining
    # (1 - multiplier) fraction is implicitly cash.
    weights = weights * multiplier

    # Do NOT renormalize to sum=1 — the sub-1.0 sum represents
    # intentional cash allocation from uncertainty gating.

    # Clean up near-zero weights
    weights[weights.abs() < 1e-6] = 0.0

    self._log(
        f"  Uncertainty gate: multiplier={multiplier:.3f} "
        f"(entropy={current_uncertainty:.3f}), "
        f"invested_fraction={weights.sum():.1%}"
    )
```

**Option B (Alternative): Move renormalization before the gate:**

If the optimizer requires weights to sum to 1.0 for other reasons, renormalize *before* the gate and then apply the multiplier as a post-processing step that reduces total allocation:

```python
# Normalize optimizer output first
w_sum = weights.sum()
if abs(w_sum) > 1e-10:
    weights = weights / w_sum

# THEN apply uncertainty gate (no subsequent renormalization)
if current_uncertainty > 0.0:
    gate = UncertaintyGate()
    weights = pd.Series(
        gate.apply_uncertainty_gate(weights.values, current_uncertainty),
        index=weights.index,
    )
```

**Downstream compatibility check:** Verify that the paper_trader and execution logic correctly handle weights that sum to less than 1.0. If they assume sum=1.0, add a `cash_weight = 1.0 - weights.sum()` field that is passed to position sizing.

**Acceptance:** With `current_uncertainty=0.8` and a sizing map that yields `multiplier=0.5`, the final portfolio is 50% invested and 50% cash. The gate materially reduces risk exposure.

---

### T3: Add Semantic Type Annotation to compute_shock_vectors confidence_series

**Problem:** The `confidence_series` parameter in `compute_shock_vectors()` is ambiguous — the docstring says "regime confidence" but the name could mean either regime or model confidence. This caused the semantic mismatch in F-02.

**File:** `regime/shock_vector.py`

**Implementation:**
1. Rename the parameter for clarity:
   ```python
   def compute_shock_vectors(
       ohlcv: pd.DataFrame,
       regime_series: Optional[pd.Series] = None,
       regime_confidence_series: Optional[pd.Series] = None,  # RENAMED from confidence_series
       ticker: str = "",
       vol_lookback: int = 20,
   ) -> Dict[pd.Timestamp, ShockVector]:
       """Compute ShockVector for each bar in the OHLCV DataFrame.

       Parameters
       ----------
       regime_confidence_series : pd.Series, optional
           Per-bar **regime detection** confidence in [0, 1], aligned to
           ``ohlcv.index``. This is the maximum posterior probability across
           regime states (from RegimeDetector), NOT model prediction confidence.
           If ``None``, defaults to 0.5 (maximum uncertainty).
       """
   ```
2. Update all callers to use the new parameter name.
3. Add a runtime type hint or assertion:
   ```python
   if regime_confidence_series is not None:
       # Warn if values look like model confidence (typically > 0.8 for most bars)
       # vs regime confidence (typically 0.4-0.9 with more variance)
       mean_conf = float(regime_confidence_series.mean())
       if mean_conf > 0.95:
           logger.warning(
               "regime_confidence_series mean=%.3f is suspiciously high — "
               "verify this is regime confidence, not model prediction confidence",
               mean_conf,
           )
   ```

**Acceptance:** The parameter is unambiguously named `regime_confidence_series`. All callers pass the correct semantic type.

---

## Verification

- [ ] Run `pytest tests/ -k "backtest or shock_vector or uncertainty"` — all pass
- [ ] Verify ShockVector uncertainty reflects regime confidence, not model confidence
- [ ] Verify autopilot with uncertainty=0.8 produces weights summing to ~0.5 (not 1.0)
- [ ] Verify risk/position_sizer usage of UncertaintyGate is unaffected
- [ ] Verify paper_trader handles sub-1.0 weight sums correctly
