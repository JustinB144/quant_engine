# SPEC_AUDIT_FIX_20: WRDS SQL Safety & Code Hygiene

**Priority:** MEDIUM — SQL injection risk via date f-strings in WRDS provider; dead code and minor cleanup.
**Scope:** `data/wrds_provider.py`, `data/loader.py`, `data/local_cache.py`, `data/alternative.py`
**Estimated effort:** 2 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The WRDS provider has a defense-in-depth gap (F-05/P1): 50+ SQL queries interpolate date parameters via f-strings (`f"WHERE datadate >= '{start_date}'"`) instead of using parameterized queries. Ticker and PERMNO inputs ARE sanitized via regex (`_TICKER_RE`, `_sanitize_permno_list`), but date parameters are not validated. While WRDS is a trusted internal source and dates come from config/datetime objects, SQL injection is technically possible if an unsanitized date string reaches these methods. Additionally, five P3 hygiene issues exist: dead code in `loader.py` (F-12, empty if-block), unused `_daily_cache_files` function in `local_cache.py` (F-13), unused `Tuple` imports (F-14), misleading docstring in `alternative.py` (F-15), and a non-thread-safe singleton in `wrds_provider.py` (F-16).

---

## Tasks

### T1: Validate Date Parameters in WRDS Provider

**Problem:** `wrds_provider.py` uses f-string interpolation for date parameters in 50+ SQL queries (e.g., `f"WHERE datadate >= '{start_date}'"` at lines 292, 312, 327, 411, 473, 562, 638, etc.). Ticker/PERMNO inputs are validated by `_TICKER_RE` and `_sanitize_permno_list`, but date parameters have no format validation.

**File:** `data/wrds_provider.py`

**Implementation:**
1. Add a date validation helper at the top of the module:
   ```python
   import re
   _DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

   def _validate_date(date_str: str, param_name: str) -> str:
       """Validate date string format to prevent SQL injection.

       Accepts YYYY-MM-DD format only. Raises ValueError for invalid input.
       """
       if not isinstance(date_str, str):
           # Convert datetime objects to string
           date_str = str(date_str)[:10]
       date_str = date_str.strip()
       if not _DATE_RE.match(date_str):
           raise ValueError(
               f"Invalid date format for {param_name}: {date_str!r} "
               f"(expected YYYY-MM-DD)"
           )
       return date_str
   ```
2. Apply validation at method entry points (not at every SQL interpolation site — too invasive):
   ```python
   def get_daily_prices(self, tickers, start_date, end_date, ...):
       start_date = _validate_date(start_date, "start_date")
       end_date = _validate_date(end_date, "end_date")
       ...
   ```
3. Apply to all public methods that accept date parameters: `get_daily_prices`, `get_fundamentals`, `get_earnings_surprises`, `get_sp500_members`, `get_options_data`, `get_delisting_returns`, `get_institutional_ownership`, `get_short_interest`.
4. Consider also using WRDS's `wrds.Connection.raw_sql(params=...)` parameterized query support where available, but date validation is the minimum required fix.

**Acceptance:** Passing `start_date="2024-01-01'; DROP TABLE--"` raises `ValueError`. Passing `start_date="2024-01-01"` works normally. Passing `start_date=datetime(2024, 1, 1)` works (auto-converted).

---

### T2: Remove Dead Code and Unused Imports

**Problem:** Multiple dead code artifacts exist across the data module:
- F-12: `loader.py:241-243` has an if-block with only `pass` (dead code).
- F-13: `local_cache.py:474` defines `_daily_cache_files()` which is never called.
- F-14: `loader.py` and `wrds_provider.py` both import `Tuple` from `typing` but never use it.

**Files:** `data/loader.py`, `data/local_cache.py`, `data/wrds_provider.py`

**Implementation:**
1. Remove the dead if-block in `loader.py:241-243`:
   ```python
   # BEFORE:
   if t not in named_cached:
       # Alias lookup via metadata may still resolve this symbol.
       pass
   # AFTER: Remove the entire if-block (lines 241-243)
   ```
2. Remove `_daily_cache_files()` function definition from `local_cache.py:474-488`.
3. Remove unused `Tuple` from import statements:
   ```python
   # loader.py line 13: Remove Tuple from typing import
   from typing import Dict, List, Optional  # removed Tuple
   # wrds_provider.py line 38: Same
   from typing import Dict, List, Optional  # removed Tuple
   ```

**Acceptance:** `ruff --select F` shows no unused import warnings for these files. The removed function has no callers (verified by grep).

---

### T3: Fix Misleading get_fundamentals Docstring

**Problem:** `alternative.py:835` defines `get_fundamentals()` whose docstring says "institutional ownership data" — misleading because "fundamentals" typically means balance sheet / income statement data.

**File:** `data/alternative.py`

**Implementation:**
1. Rename the function to match its actual behavior:
   ```python
   def get_institutional_ownership_data(
       ticker: str,
       as_of_date: Optional[datetime] = None,
   ) -> Optional[pd.DataFrame]:
       """Module-level convenience wrapper for institutional ownership data.

       Delegates to :meth:`AlternativeDataProvider.get_institutional_ownership`.
       Returns ``None`` if WRDS is unavailable.
       """
   ```
2. Add a deprecation alias for backward compatibility:
   ```python
   def get_fundamentals(ticker, as_of_date=None):
       """Deprecated: Use get_institutional_ownership_data() instead."""
       import warnings
       warnings.warn(
           "get_fundamentals() is deprecated and misleadingly named. "
           "Use get_institutional_ownership_data() instead.",
           DeprecationWarning, stacklevel=2,
       )
       return get_institutional_ownership_data(ticker, as_of_date)
   ```
3. Update all callers to use the new name.

**Acceptance:** `get_fundamentals()` emits a DeprecationWarning. `get_institutional_ownership_data()` works without warning. No callers use the old name without the warning.

---

### T4: Add Thread Lock to WRDS Provider Singleton

**Problem:** `wrds_provider.py:1604-1612` `get_wrds_provider()` uses a module-level `_default_provider` variable without a thread lock. Concurrent calls can create multiple instances. While the DB connection init is thread-safe, the singleton pattern itself is not.

**File:** `data/wrds_provider.py`

**Implementation:**
1. Add a threading lock:
   ```python
   import threading
   _provider_lock = threading.Lock()
   _default_provider: Optional['WRDSProvider'] = None

   def get_wrds_provider() -> 'WRDSProvider':
       """Get or create the default WRDSProvider singleton (thread-safe)."""
       global _default_provider
       if _default_provider is not None:
           return _default_provider
       with _provider_lock:
           # Double-checked locking
           if _default_provider is None:
               _default_provider = WRDSProvider()
       return _default_provider
   ```
2. The double-checked locking pattern avoids taking the lock on the fast path (singleton already exists).

**Acceptance:** Concurrent calls to `get_wrds_provider()` from multiple threads return the same instance. No duplicate `WRDSProvider` instances are created.

---

## Verification

- [ ] Run `pytest tests/ -k "wrds or provider or alternative"` — all pass
- [ ] Verify SQL injection via date parameter raises ValueError
- [ ] Verify no unused imports flagged by ruff
- [ ] Verify get_fundamentals() emits DeprecationWarning
- [ ] Verify concurrent get_wrds_provider() calls return same instance
