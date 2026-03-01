# SPEC_AUDIT_FIX_18: Cache Integrity & Reliability Fixes

**Priority:** HIGH — Greedy column matching can corrupt OHLCV data; cache errors invisible in production; metadata provenance can be overwritten.
**Scope:** `data/local_cache.py`
**Estimated effort:** 2.5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The cache layer has five reliability issues. First (F-02/P1), `_normalize_ohlcv_columns()` uses substring matching (`"low" in col_l`) which matches column names like "following", "shallow", "below", "flow" — corrupting OHLCV data if upstream sources introduce metadata columns with these substrings. Second (F-06/P2), `payload.update(meta)` at line 218 allows caller-supplied metadata to overwrite core provenance fields (`source`, `fetched_at`, `ticker`, `rows`), undermining cache trust hierarchy decisions. Third (F-07/P2), data parquet and metadata sidecar writes are separate non-atomic operations — a crash between them leaves orphaned data without metadata. Fourth (F-08/P2), all cache error conditions (missing files, parse failures, corrupt parquet) are logged at DEBUG level, making them invisible in production. Fifth (F-09/P2), `load_intraday_ohlcv()` does not search `FALLBACK_SOURCE_DIRS`, unlike daily loading functions.

---

## Tasks

### T1: Replace Greedy Substring Column Matching With Exact/Anchored Matching

**Problem:** `local_cache.py:74-82` uses `"low" in col_l`, `"high" in col_l`, `"open" in col_l`, `"close" in col_l` for OHLCV column detection. This matches any column name containing the substring (e.g., "following" → "Low", "highlight" → "High", "reopen" → "Open").

**File:** `data/local_cache.py`

**Implementation:**
1. Replace substring checks with a priority-ordered exact/anchored matching strategy:
   ```python
   import re

   # Exact matches first, then anchored patterns
   _OHLCV_PATTERNS = {
       "Open": [
           re.compile(r"^(1\.\s*)?open$", re.I),  # "open", "1. open"
           re.compile(r"^adj[\._\s]?open$", re.I),  # "adj_open"
       ],
       "High": [
           re.compile(r"^(2\.\s*)?high$", re.I),
           re.compile(r"^adj[\._\s]?high$", re.I),
       ],
       "Low": [
           re.compile(r"^(3\.\s*)?low$", re.I),
           re.compile(r"^adj[\._\s]?low$", re.I),
       ],
       "Close": [
           re.compile(r"^(4\.\s*)?close$", re.I),
           re.compile(r"^adj[\._\s]?close$", re.I),
       ],
       "Volume": [
           re.compile(r"^(5\.\s*|6\.\s*)?volume$", re.I),
       ],
   }

   def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
       rename_map = {}
       for col in df.columns:
           col_stripped = str(col).strip()
           for target, patterns in _OHLCV_PATTERNS.items():
               if any(p.match(col_stripped) for p in patterns):
                   rename_map[col] = target
                   break
       return df.rename(columns=rename_map)
   ```
2. This ensures "following", "shallow", "below", "highlight", "reopen" etc. are NOT matched.
3. Add known column name variants from each data source as explicit patterns (Alpha Vantage uses "1. open", WRDS uses "PRC" mapped elsewhere).
4. Add a warning log when an unrecognized column is encountered that looks OHLCV-like but doesn't match patterns.

**Acceptance:** Column "following" is NOT remapped to "Low". Column "1. open" IS remapped to "Open". Column "Open" IS remapped to "Open". A test with ambiguous column names verifies no false matches.

---

### T2: Protect Core Metadata Fields From Caller Overwrite

**Problem:** `local_cache.py:218` `payload.update(meta)` allows caller-supplied metadata to overwrite core provenance fields (`source`, `fetched_at`, `ticker`, `rows`, `date_range`). The `source` field is critical for cache trust hierarchy decisions.

**File:** `data/local_cache.py`

**Implementation:**
1. Build the core payload after merging caller metadata, or use a protected-keys pattern:
   ```python
   _PROTECTED_META_KEYS = {"source", "saved_at_utc", "ticker", "n_bars", "start_date", "end_date", "format"}

   def _write_cache_meta(path, *, ticker, df, source, meta=None):
       payload = {}
       # Merge caller metadata first (lower priority)
       if meta:
           # Warn and strip protected keys from caller meta
           protected_found = set(meta.keys()) & _PROTECTED_META_KEYS
           if protected_found:
               logger.warning(
                   "Caller meta for %s tried to set protected keys %s — ignoring",
                   ticker, protected_found,
               )
           payload.update({k: v for k, v in meta.items() if k not in _PROTECTED_META_KEYS})
       # Core fields always written last (higher priority)
       payload["source"] = source
       payload["saved_at_utc"] = datetime.utcnow().isoformat()
       payload["ticker"] = ticker
       payload["n_bars"] = len(df)
       # ... other core fields
   ```
2. This ensures core provenance fields are always authoritative regardless of what callers pass.

**Acceptance:** Calling `cache_save("AAPL", df, source="wrds", meta={"source": "fake"})` produces metadata with `source="wrds"`, not `"fake"`. A warning is logged.

---

### T3: Atomic Data+Metadata Pair Writes

**Problem:** `local_cache.py:271-272` writes parquet data first, then metadata sidecar in a separate operation. A crash between writes leaves orphaned data with no metadata.

**File:** `data/local_cache.py`

**Implementation:**
1. Write both files to a temporary directory first, then atomically move both:
   ```python
   import tempfile

   def _atomic_pair_write(parquet_path: Path, meta_path: Path, df, meta_payload):
       """Write data + metadata as an atomic pair."""
       parent = parquet_path.parent
       with tempfile.TemporaryDirectory(dir=parent) as tmp_dir:
           tmp_parquet = Path(tmp_dir) / parquet_path.name
           tmp_meta = Path(tmp_dir) / meta_path.name
           # Write both to temp dir
           df.to_parquet(tmp_parquet)
           tmp_meta.write_text(json.dumps(meta_payload, default=str, indent=2))
           # Atomic move both (on same filesystem, os.replace is atomic)
           os.replace(tmp_parquet, parquet_path)
           os.replace(tmp_meta, meta_path)
   ```
2. If either write fails, neither file is updated (the temp dir is cleaned up).
3. Update the main cache save path to use `_atomic_pair_write`.
4. For backward compatibility, add a guard in `load_ohlcv_with_meta` that handles orphaned parquet files without metadata (log WARNING, treat as untrusted).

**Acceptance:** A simulated crash after the first write leaves neither file updated. Orphaned data files (from before this fix) log a WARNING and are treated as untrusted.

---

### T4: Elevate Cache Error Logging From DEBUG to WARNING

**Problem:** `local_cache.py` logs all error conditions (missing files, parse failures, corrupt parquet, write failures) at `logger.debug()` level, making them invisible in production.

**File:** `data/local_cache.py`

**Implementation:**
1. Elevate error conditions to appropriate levels:
   ```python
   # File not found / missing — INFO (expected during cold start)
   logger.info("Cache miss for %s: file not found", path.name)

   # Parse failure / corrupt data — WARNING (unexpected, needs attention)
   logger.warning("Could not read parquet %s: %s", path.name, e)
   logger.warning("Parquet write failed for %s, falling back to CSV: %s", ticker, e)
   logger.warning("Could not write cache meta for %s: %s", ticker, e)

   # Missing columns after read — WARNING
   logger.warning("Cache file %s missing required columns", path.name)
   ```
2. Keep expected cache misses (file not found on first load) at INFO level to avoid log noise.
3. All error conditions involving data corruption or write failures should be at WARNING.

**Acceptance:** With production logging at INFO level, cache corruption events appear in logs. Cache misses are at INFO (not WARNING) to avoid noise.

---

### T5: Add FALLBACK_SOURCE_DIRS Search to load_intraday_ohlcv

**Problem:** `local_cache.py:391-398` `load_intraday_ohlcv()` only searches a single directory (`cache_dir` or `DATA_CACHE_DIR/"intraday"`). Daily loading functions search `FALLBACK_SOURCE_DIRS` for additional cache directories, but intraday does not.

**File:** `data/local_cache.py`

**Implementation:**
1. Mirror the daily loading fallback pattern:
   ```python
   def load_intraday_ohlcv(
       ticker: str,
       timeframe: str = "5m",
       cache_dir: Optional[str] = None,
   ) -> Optional[pd.DataFrame]:
       d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR / "intraday"
       t = ticker.upper()

       # Search primary dir + fallback dirs (matching daily load behavior)
       search_roots = [d]
       for fallback in FALLBACK_SOURCE_DIRS:
           intraday_fallback = Path(fallback) / "intraday"
           if intraday_fallback != d and intraday_fallback.exists():
               search_roots.append(intraday_fallback)

       for root in search_roots:
           # existing pattern-matching logic, searching in 'root' instead of 'd'
           ...
           if result is not None:
               return result

       return None
   ```
2. Maintain the same search order: primary dir first, then fallbacks.

**Acceptance:** Intraday data stored in a fallback directory is found by `load_intraday_ohlcv()`. A test with data in a fallback dir verifies discovery.

---

## Verification

- [ ] Run `pytest tests/ -k "cache or local_cache"` — all pass
- [ ] Verify "following" column is NOT remapped to "Low"
- [ ] Verify caller meta cannot overwrite `source` field
- [ ] Verify atomic pair write survives simulated crash
- [ ] Verify cache write failures appear in production-level logs
- [ ] Verify intraday data in fallback dirs is discoverable
