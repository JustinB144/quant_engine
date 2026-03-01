# SPEC_AUDIT_FIX_23: Pipeline Observability & Feature Integration Fixes

**Priority:** MEDIUM — Feature blocks silently degrade to imputed medians without operator visibility; 2 phantom metadata entries; 4 orphaned features.
**Scope:** `features/pipeline.py`, `features/options_factors.py`, `features/intraday.py`
**Estimated effort:** 2.5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The feature pipeline has three integration/observability issues. First (F-06/P2), 9 of 14 lazy import paths catch bare `Exception` (rather than specific exception types) and only report failures via `if verbose: print(...)` — when `verbose=False` (the default in production), features silently disappear and the predictor substitutes imputed medians. The 4 structural analyzer blocks (Spectral, SSA, TailRisk, OT at lines 769-854) all catch bare `Exception`, while `EigenvalueAnalyzer` at line 1380 and `CorrelationRegimeDetector` at line 1320 correctly catch `(ImportError, ValueError, RuntimeError)`. Second (F-07/P2), `FEATURE_METADATA` includes `rolling_vwap_20` and `rolling_vwap_deviation_20` (pipeline.py:400-401) but `compute_rolling_vwap()` in intraday.py:203 is never called from the pipeline — these are phantom metadata entries. Third (F-09/P2), `compute_iv_shock_features()` in options_factors.py:92 produces 4 features but is never imported or called by the pipeline.

---

## Tasks

### T1: Promote Fallback Failures to logging.warning() and Narrow Exception Catches

**Problem:** 4 structural analyzer blocks (pipeline.py:781, 801, 828, 852) and 5 other lazy import blocks (1294, 1402, 1451, 1487, 1499) catch exceptions and only report via `if verbose: print(...)`. In production (`verbose=False`), these failures are completely invisible. Additionally, the 4 structural analyzer blocks catch bare `Exception`, swallowing genuine bugs like `TypeError`, `KeyError`, `AttributeError`.

**File:** `features/pipeline.py`

**Implementation:**
1. Replace all `if verbose: print(...)` failure reporting with `logging.warning()`:
   ```python
   import logging
   logger = logging.getLogger(__name__)

   # BEFORE (line 781-783):
   except Exception as e:
       if verbose:
           print(f"    WARNING: Spectral features failed: {e}")

   # AFTER:
   except (ImportError, ValueError, RuntimeError) as e:
       logger.warning("Spectral features failed: %s", e)
   ```
2. Narrow all 4 structural analyzer exception catches from bare `Exception` to `(ImportError, ValueError, RuntimeError)`, matching the pattern already used by `EigenvalueAnalyzer` (line 1380) and `CorrelationRegimeDetector` (line 1320).
3. Apply to all 4 structural blocks:
   - Spectral (line 781): `except Exception` → `except (ImportError, ValueError, RuntimeError)`
   - SSA (line 801): same
   - TailRisk (line 828): same
   - OT (line 852): same
4. For blocks that already catch specific exceptions (HARX:1294, Correlation:1320, Eigenvalue:1380, Macro:1402, WRDS:1451), change their reporting from `if verbose: print(...)` to `logger.warning(...)`.
5. The `load_ohlcv` import at line 1528 has no try/except (hard fail) — leave unchanged, this is correct.
6. Add a counter tracking how many optional feature blocks succeeded vs failed:
   ```python
   # After all optional blocks:
   if n_failed > 0:
       logger.warning(
           "%d/%d optional feature blocks failed — model may use imputed values",
           n_failed, n_total,
       )
   ```

**Acceptance:** With `verbose=False` and production-level logging (WARNING), failed feature blocks appear in logs. Genuine bugs (TypeError, KeyError) are NOT silently caught by structural analyzer blocks.

---

### T2: Wire compute_rolling_vwap Into Pipeline or Remove Phantom Metadata

**Problem:** `FEATURE_METADATA` at pipeline.py:400-401 registers `rolling_vwap_20` and `rolling_vwap_deviation_20`, but `compute_rolling_vwap()` in intraday.py:203 is never called from the pipeline. Downstream code expecting these features gets NaN/missing values.

**Files:** `features/pipeline.py`, `features/intraday.py`

**Implementation — Option A (Recommended): Wire into pipeline:**
1. Add a call to `compute_rolling_vwap()` in the intraday feature computation block:
   ```python
   # In the intraday features section (near line 1440):
   try:
       from .intraday import compute_rolling_vwap
       if intraday_df is not None and len(intraday_df) > 0:
           rolling_vwap_feats = compute_rolling_vwap(intraday_df, window=20)
           if rolling_vwap_feats is not None:
               for col in rolling_vwap_feats.columns:
                   intraday_features[col] = rolling_vwap_feats[col]
   except (ImportError, ValueError, RuntimeError) as e:
       logger.warning("Rolling VWAP features failed: %s", e)
   ```
2. Verify `compute_rolling_vwap()` returns columns named exactly `rolling_vwap_20` and `rolling_vwap_deviation_20` to match metadata.

**Implementation — Option B (Remove phantom metadata):**
1. Remove lines 400-401 from `FEATURE_METADATA`:
   ```python
   # DELETE:
   "rolling_vwap_20": {"type": "CAUSAL", "category": "vwap"},
   "rolling_vwap_deviation_20": {"type": "CAUSAL", "category": "vwap"},
   ```

**Acceptance (Option A):** `rolling_vwap_20` appears in computed feature frames when intraday data is available. **Acceptance (Option B):** No phantom metadata entries exist — every `FEATURE_METADATA` entry has a corresponding emitter.

---

### T3: Integrate compute_iv_shock_features or Move to Experimental

**Problem:** `options_factors.py:92` defines `compute_iv_shock_features()` producing 4 features (`iv_shock_{window}`, `iv_shock_magnitude_{window}`, `vol_regime_shift_{window}`, `iv_compression_{window}`), but pipeline.py:70 only imports `compute_option_surface_factors`. The 4 IV shock features are never computed in production.

**Files:** `features/pipeline.py`, `features/options_factors.py`

**Implementation — Option A (Integrate):**
1. Add import and call to `compute_iv_shock_features` in the options section of the pipeline:
   ```python
   from .options_factors import compute_option_surface_factors, compute_iv_shock_features
   ```
2. Call after option surface factors:
   ```python
   try:
       iv_shock_feats = compute_iv_shock_features(df, windows=[20])
       if iv_shock_feats is not None:
           features = pd.concat([features, iv_shock_feats], axis=1)
   except (ValueError, RuntimeError) as e:
       logger.warning("IV shock features failed: %s", e)
   ```
3. Register the 4 new features in `FEATURE_METADATA`:
   ```python
   "iv_shock_20": {"type": "CAUSAL", "category": "options"},
   "iv_shock_magnitude_20": {"type": "CAUSAL", "category": "options"},
   "vol_regime_shift_20": {"type": "CAUSAL", "category": "options"},
   "iv_compression_20": {"type": "CAUSAL", "category": "options"},
   ```

**Implementation — Option B (Deprecate):**
1. Add a deprecation comment to the function.
2. Move to `features/experimental/` or add `_experimental` suffix.

**Acceptance (Option A):** IV shock features appear in production feature frames. **Acceptance (Option B):** Orphaned function is clearly marked as non-production.

---

## Verification

- [ ] Run `pytest tests/ -k "pipeline or features"` — all pass
- [ ] Verify failed feature blocks appear in WARNING-level logs with verbose=False
- [ ] Verify TypeErrors in structural analyzers propagate (not silently caught)
- [ ] Verify every FEATURE_METADATA entry has a corresponding emitter (or phantom entries removed)
