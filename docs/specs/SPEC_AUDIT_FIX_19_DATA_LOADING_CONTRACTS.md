# SPEC_AUDIT_FIX_19: Data Loading Contract & Validation Fixes

**Priority:** HIGH — Hardcoded bar minimum diverges from config; fallback tracker keys mismatch data dict; intraday quality can double-count; no ticker validation.
**Scope:** `data/loader.py`, `data/intraday_quality.py`
**Estimated effort:** 2.5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The data loading contracts have four validated issues. First (F-04/P1, also Report 2 Finding 2), `load_universe()` hardcodes `500` at line 523 instead of using the imported `MIN_BARS` config constant — config changes have no effect. Second (F-10/P2, also Report 2 Finding 5), `_fallback_tracker` is keyed by ticker while the data dict is keyed by PERMNO, and the tracker is process-global without per-cycle reset, accumulating stale entries. Third (Report 2 Finding 4/P1, partially valid), intraday quality scoring's soft-flag path can double-count bars that trigger multiple soft checks — hard rejections are correctly deduplicated via a set, but soft flags are not. Fourth (F-11/P2), `load_ohlcv()` accepts any string as a ticker with no validation, creating a theoretical path-traversal risk since ticker strings are used to construct cache file paths.

---

## Tasks

### T1: Replace Hardcoded 500-Bar Minimum With MIN_BARS Config

**Problem:** `loader.py:523` uses `if df is not None and len(df) >= 500` and line 550 has `"need 500"` in the skip reason. The config constant `MIN_BARS` is imported at line 34 but never used in `load_universe()`. Meanwhile, `_cache_is_usable()` at line 191 correctly uses `MIN_BARS`.

**File:** `data/loader.py`

**Implementation:**
1. Replace both hardcoded references:
   ```python
   # Line 523: Replace hardcoded 500
   if df is not None and len(df) >= MIN_BARS:
       ...

   # Line 550: Replace hardcoded message
   reason = f"insufficient data ({bars} bars, need {MIN_BARS})"
   ```
2. Verify that `MIN_BARS` is imported and accessible in the function scope (it is — line 34).
3. Add a comment noting the config source for future maintainers.

**Acceptance:** Changing `MIN_BARS` in config from 500 to 252 causes `load_universe()` to accept tickers with 252+ bars. Both the threshold and the skip-reason message reflect the config value.

---

### T2: Fix Fallback Tracker Key Mismatch and Process-Global Accumulation

**Problem:** `loader.py:27-28` defines `_fallback_tracker` as a module-level dict keyed by ticker string (line 478). The returned data dict from `load_universe()` is keyed by PERMNO. There is no mapping between the two key spaces. Additionally, `load_universe()` resets `_skip_tracker` at line 516 but does NOT reset `_fallback_tracker`, so it accumulates across multiple invocations.

**File:** `data/loader.py`

**Implementation:**
1. Reset `_fallback_tracker` at the start of `load_universe()`:
   ```python
   def load_universe(tickers, years=15, verbose=True):
       global _skip_tracker, _fallback_tracker
       _skip_tracker = {}
       _fallback_tracker = {}  # Reset per load cycle
       ...
   ```
2. Include PERMNO in fallback tracker entries for cross-referencing:
   ```python
   _fallback_tracker[requested_symbol] = {
       "source": source,
       "reason": "wrds_unavailable" if use_wrds else "wrds_disabled",
       "bars": len(cached),
       "trusted": False,
       "permno": permno_key,  # NEW: cross-reference to data dict key
   }
   ```
3. Update `get_data_provenance()` / `get_skip_reasons()` to include the PERMNO mapping in the output.

**Acceptance:** Calling `load_universe()` twice does not accumulate stale entries from the first call. Each fallback entry includes the PERMNO key for cross-referencing with the data dict.

---

### T3: Deduplicate Intraday Quality Soft-Flag Counts

**Problem:** `intraday_quality.py:82-94` accumulates `total_flagged` additively via `report.add_check(result)`. A bar that triggers both the stale-price and extreme-return soft checks is counted twice in the total. Hard rejections are correctly deduplicated via a set (`all_hard_rejected`) at line 1028, but soft flags have no equivalent deduplication.

**File:** `data/intraday_quality.py`

**Implementation:**
1. Add a soft-flag deduplication set, mirroring the hard-rejection pattern:
   ```python
   all_soft_flagged: Set[int] = set()  # Bar indices flagged by any soft check

   # After each soft check:
   for check_name, flagged_indices in soft_results:
       all_soft_flagged.update(flagged_indices)
       report.add_check(check_result)

   # Use len(all_soft_flagged) for unique flag count
   report.unique_flagged_bars = len(all_soft_flagged)
   ```
2. Add `unique_flagged_bars` and `unique_rejected_bars` fields to the quality report alongside the per-check breakdown.
3. Use the deduplicated counts for quarantine threshold evaluation:
   ```python
   # Quarantine decision should use unique bars, not summed overlapping counts
   rejection_rate = len(all_hard_rejected) / total_bars
   if rejection_rate > QUARANTINE_THRESHOLD:
       quarantine_ticker(...)
   ```
4. Keep the per-check breakdown for diagnostic detail, but use unique counts for decisions.

**Acceptance:** A bar that fails both stale-price and extreme-return soft checks is counted once in `unique_flagged_bars`. Quarantine decisions use deduplicated counts.

---

### T4: Add Ticker String Validation to load_ohlcv

**Problem:** `loader.py:340` `load_ohlcv(ticker: str)` accepts any string without validation. Ticker strings are used to construct cache file paths (e.g., `DATA_CACHE_DIR / f"{ticker}_daily.parquet"`). While practical risk is low (tickers come from WRDS/config), the API layer at `api/services/data_service.py:45` passes user-provided ticker strings.

**File:** `data/loader.py`

**Implementation:**
1. Add ticker validation using a pattern consistent with `wrds_provider.py`'s existing `_TICKER_RE`:
   ```python
   import re
   _TICKER_RE = re.compile(r'^[A-Z0-9.\-/^]{1,12}$')

   def load_ohlcv(ticker: str, years: int = 15, ...) -> Optional[pd.DataFrame]:
       requested_symbol = str(ticker).upper().strip()
       if not _TICKER_RE.match(requested_symbol):
           logger.warning("Invalid ticker format rejected: %r", ticker)
           return None
       ...
   ```
2. This prevents path traversal (`../../etc/passwd`), empty strings, and overly long inputs.
3. Apply the same validation at the top of `load_universe()` for each ticker in the input list:
   ```python
   valid_tickers = [t for t in tickers if _TICKER_RE.match(str(t).upper().strip())]
   if len(valid_tickers) < len(tickers):
       logger.warning("Rejected %d invalid ticker strings", len(tickers) - len(valid_tickers))
   ```

**Acceptance:** `load_ohlcv("../../etc/passwd")` returns `None` with a warning log. `load_ohlcv("AAPL")` works normally. `load_ohlcv("BRK.B")` works (dot is allowed).

---

## Verification

- [ ] Run `pytest tests/ -k "loader or intraday_quality"` — all pass
- [ ] Verify MIN_BARS config change affects load_universe threshold
- [ ] Verify fallback tracker is reset between load_universe calls
- [ ] Verify intraday quality deduplicates overlapping soft flags
- [ ] Verify invalid ticker strings are rejected
