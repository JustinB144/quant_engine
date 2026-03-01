# SPEC_AUDIT_FIX_17: Data Quality Gate Enforcement

**Priority:** CRITICAL — Corrupt OHLCV data can enter feature computation, backtests, and autopilot without quality validation; survivorship-safe loading is not a hard path.
**Scope:** `data/loader.py`, `data/quality.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The data ingestion pipeline has three quality-gate enforcement gaps. First (F-01/P0), `load_with_delistings()` loads OHLCV data for all tickers including delisted companies but never calls `DataIntegrityValidator.validate_universe()` or `assess_ohlcv_quality()` — corrupt data from delisted companies enters the pipeline unchecked. Second (F-03/P1), when `load_ohlcv()` fetches fresh data from WRDS, it saves the result to cache at line 452 without running `assess_ohlcv_quality()` first — a corrupt WRDS fetch becomes "trusted" cached data. Third (Report 2 Finding 1/P0), when WRDS is unavailable, `load_survivorship_universe()` silently falls back to a static `UNIVERSE_FULL` list at four separate fallback points (lines 608, 621, 673, 696), which is survivorship-biased — violating the invariant that survivorship-safe loading is a hard correctness path.

---

## Tasks

### T1: Add DataIntegrityValidator to load_with_delistings

**Problem:** `loader.py:707-849` `load_with_delistings()` returns data without ever calling `DataIntegrityValidator`. The validator only runs inside `load_universe()` at line 566, but `load_with_delistings()` returns before reaching that code path when WRDS data is available.

**File:** `data/loader.py`

**Implementation:**
1. Add a validation step at the end of `load_with_delistings()`, before the return statement:
   ```python
   # Truth Layer: validate delisting-augmented data before returning
   if TRUTH_LAYER_FAIL_ON_CORRUPT and data:
       from ..validation.data_integrity import DataIntegrityValidator
       validator = DataIntegrityValidator(fail_fast=False)  # Don't fail on first bad ticker
       integrity_result = validator.validate_universe(data)
       if not integrity_result.passed:
           # Remove corrupt tickers rather than failing entirely —
           # delisted companies may legitimately have unusual price patterns
           for bad_ticker in integrity_result.failed_tickers:
               logger.warning("Removing corrupt delisted ticker %s from universe", bad_ticker)
               data.pop(bad_ticker, None)
           if not data:
               raise RuntimeError(
                   f"All {integrity_result.n_stocks_failed} tickers failed integrity check"
               )
   ```
2. Use `fail_fast=False` (unlike the `load_universe` path which uses `fail_fast=True`) because delisted tickers may have unusual but legitimate price patterns — we want to remove only the truly corrupt ones.
3. Log a summary of removed tickers at WARNING level.

**Acceptance:** `load_with_delistings()` output has been validated by `DataIntegrityValidator`. Corrupt tickers are removed with warning logs. A test passes corrupt OHLCV data through `load_with_delistings()` and verifies it is caught.

---

### T2: Add Quality Check on WRDS-Fetched Data Before Caching

**Problem:** `loader.py:420-462` normalizes WRDS data and saves it to cache at line 452 without calling `assess_ohlcv_quality()`. A corrupt WRDS fetch (NaN prices, impossible OHLC violations) becomes trusted cached data.

**File:** `data/loader.py`

**Implementation:**
1. After normalization (line 421) and before caching (line 451), add a quality check:
   ```python
   result = _normalize_ohlcv(wrds_df)
   result = _harmonize_return_columns(result) if result is not None else None

   # Quality-gate WRDS data before caching
   if result is not None and DATA_QUALITY_ENABLED:
       from .quality import assess_ohlcv_quality
       quality = assess_ohlcv_quality(result)
       if not quality.passed:
           logger.warning(
               "WRDS data for %s failed quality check (score=%.2f): %s — not caching",
               requested_symbol, quality.score,
               "; ".join(quality.issues[:3]),
           )
           # Don't cache bad data; fall through to fallback paths
           result = None
   ```
2. This ensures corrupt WRDS data is never saved to cache as trusted data.
3. If the quality check fails, `result` is set to `None`, allowing the existing fallback logic to try cached data or other sources.

**Acceptance:** A WRDS fetch with NaN prices is not saved to cache. The quality check runs before `cache_save()`. The fallback path activates when WRDS data fails quality.

---

### T3: Make Survivorship-Safe Loading a Hard Path

**Problem:** `load_survivorship_universe()` has four fallback points (lines 608, 621, 673, 696) that all fall back to `UNIVERSE_FULL`, a static survivorship-biased list. This violates the invariant that survivorship-safe loading is a hard correctness path.

**File:** `data/loader.py`

**Implementation:**
1. Add a `survivorship_safe` flag to the returned data dict (via metadata) indicating whether the universe is actually survivorship-safe:
   ```python
   def load_survivorship_universe(
       as_of_date: Optional[str] = None,
       years: int = 15,
       verbose: bool = True,
       strict: bool = False,  # NEW: if True, raise instead of falling back
   ) -> Dict[str, pd.DataFrame]:
   ```
2. At each fallback point, log a WARNING and tag the result:
   ```python
   if not WRDS_ENABLED:
       logger.warning(
           "WRDS disabled — survivorship-safe universe unavailable. "
           "Falling back to static universe (SURVIVORSHIP BIAS RISK)."
       )
       if strict:
           raise RuntimeError("Survivorship-safe loading requires WRDS but WRDS is disabled")
       result = load_universe(fallback, years=years, verbose=verbose)
       # Tag the result so downstream consumers know this is not survivorship-safe
       for df in result.values():
           df.attrs["survivorship_safe"] = False
       return result
   ```
3. For the WRDS-available path that succeeds, tag as safe:
   ```python
   for df in data.values():
       df.attrs["survivorship_safe"] = True
   ```
4. Update backtest and training entry points to check `survivorship_safe` and emit warnings when running on biased data:
   ```python
   # In backtest engine / trainer entry points:
   sample_df = next(iter(data.values()), None)
   if sample_df is not None and not sample_df.attrs.get("survivorship_safe", True):
       logger.warning("⚠ Running on survivorship-BIASED universe — results may overstate performance")
   ```

**Acceptance:** When WRDS is unavailable, the returned data is tagged `survivorship_safe=False`. With `strict=True`, fallback raises `RuntimeError` instead. Backtest/training consumers emit a warning when using biased data.

---

## Verification

- [ ] Run `pytest tests/ -k "loader or data_integrity or survivorship"` — all pass
- [ ] Verify `load_with_delistings()` calls DataIntegrityValidator on output
- [ ] Verify corrupt WRDS data is not cached (inject bad data, confirm cache_save not called)
- [ ] Verify survivorship fallback tags data with `survivorship_safe=False`
- [ ] Verify `strict=True` raises RuntimeError when WRDS is unavailable
