# SPEC_AUDIT_FIX_38: Kalshi API Contracts, Data Integrity & Security Fixes

**Priority:** HIGH — API service layer completely broken; outcome table separation defeated; security exposure in fallback signing path.
**Scope:** `api/services/kalshi_service.py`, `kalshi/storage.py`, `kalshi/events.py`, `kalshi/walkforward.py`, `kalshi/promotion.py`, `kalshi/client.py`, `kalshi/distribution.py`, `config.py`
**Estimated effort:** 5–6 hours
**Depends on:** SPEC_13 (data integrity fixes land first)
**Blocks:** Nothing

---

## Context

Three independent audits of Subsystem 9 (Kalshi / Event Markets) identified findings not covered by the existing SPEC_13. SPEC_13 addresses threshold-direction substring matching, event-to-market merge mis-assignment, INSERT OR REPLACE nullification, walk-forward capacity overstatement, regime zero-value drop, unresolved direction moments, lexicographic version ordering, and bin validation gating. This spec covers the **remaining high-priority gaps**: a completely broken API service layer, bidirectional outcome upsert defeating first-print/revised separation, unactivated purge/embargo config, event strategies bypassing stress-regime gates, OpenSSL credential exposure, and incomplete early-return schemas.

### Cross-Audit Reconciliation

| Finding | Auditor 1 (Claude) | Auditor 2 (Claude) | Auditor 3 (Codex) | Existing Spec | Disposition |
|---------|--------------------|--------------------|-------------------|---------------|-------------|
| Threshold direction substring | K-I/SPEC_13 | — | — | **SPEC_13 T1** | Already covered |
| Event-to-market merge fallback | — | — | — | **SPEC_13 T2** | Already covered |
| INSERT OR REPLACE nullification | — | — | — | **SPEC_13 T3** | Already covered |
| Walk-forward capacity overstatement | — | — | — | **SPEC_13 T4** | Already covered |
| Regime zero-value drop | — | — | — | **SPEC_13 T5** | Already covered |
| Unresolved direction moments | — | — | — | **SPEC_13 T6** | Already covered |
| Lexicographic version ordering | — | — | — | **SPEC_13 T7** | Already covered |
| Bin validation quality gating | — | — | — | **SPEC_13 T8** | Already covered |
| kalshi_service.py broken API | K-01/K-02/K-05 (P1/P2) | — | F-01 (P1) | Not in any spec | **NEW → T1** |
| Bidirectional outcome upsert | K-06 (P2) | F-01 (P1) | — | Not in any spec | **NEW → T2** |
| Purge/embargo config not wired | K-03 (P1) | — | F-04 (P2) | Not in any spec | **NEW → T3** |
| Event strategies bypass stress gates | K-09 (P2) | — | — | Not in any spec | **NEW → T4** |
| OpenSSL passphrase in args | K-18 (P3) | F-02 (P1) | — | Not in any spec | **NEW → T5** |
| Private key in temp file | — | F-03 (P2) | — | Not in any spec | **NEW → T5** |
| Early-return dicts missing fields | K-07 (P2) | — | — | Not in any spec | **NEW → T6** |

---

## Tasks

### T1: Rewrite kalshi_service.py to Use Correct query_df API

**Problem:** `api/services/kalshi_service.py:27` calls `store.query_df("kalshi_markets", limit=200)` and line 45 calls `store.query_df("kalshi_contracts", where=f"event_id = '{market_id}'")`. The actual `query_df()` signature at `storage.py:637` is `query_df(sql: str, params=None)` — it expects a SQL string, not a table name with keyword arguments. Both calls raise `TypeError` at runtime. Additionally:
- Line 45 uses f-string interpolation of `market_id` into SQL, creating a SQL injection vector.
- Line 45 queries `kalshi_contracts` with an `event_id` filter, but that table has no `event_id` column. The correct table for distribution data is `kalshi_distributions`, filtered by `market_id`.

**Files:** `api/services/kalshi_service.py`

**Implementation:**
1. Rewrite `get_events()` to use proper SQL with `query_df()`:
   ```python
   def get_events(self, event_type: Optional[str] = None) -> Dict[str, Any]:
       if not self.enabled:
           return {"enabled": False, "events": []}
       try:
           from quant_engine.config import KALSHI_DB_PATH
           from quant_engine.kalshi.storage import EventTimeStore

           store = EventTimeStore(KALSHI_DB_PATH)
           if event_type:
               df = store.query_df(
                   "SELECT * FROM kalshi_markets WHERE event_type LIKE ? LIMIT 200",
                   params=[f"%{event_type}%"],
               )
           else:
               df = store.query_df("SELECT * FROM kalshi_markets LIMIT 200")
           records = df.to_dict(orient="records")
           return {"enabled": True, "events": records, "total": len(records)}
       except Exception as exc:
           logger.warning("Kalshi query failed: %s", exc)
           return {"enabled": True, "events": [], "error": str(exc)}
   ```
2. Rewrite `get_distributions()` to query the correct table with parameterized SQL:
   ```python
   def get_distributions(self, market_id: str) -> Dict[str, Any]:
       if not self.enabled:
           return {"enabled": False}
       try:
           from quant_engine.config import KALSHI_DB_PATH
           from quant_engine.kalshi.storage import EventTimeStore

           store = EventTimeStore(KALSHI_DB_PATH)
           df = store.query_df(
               "SELECT * FROM kalshi_distributions WHERE market_id = ?",
               params=[market_id],
           )
           records = df.to_dict(orient="records") if len(df) else []
           return {"enabled": True, "market_id": market_id, "distributions": records}
       except Exception as exc:
           logger.warning("Kalshi distribution query failed: %s", exc)
           return {"enabled": True, "market_id": market_id, "distributions": [], "error": str(exc)}
   ```
3. Ensure `store` is properly closed after use (coordinate with SPEC_39 T5 once context manager is added).
4. Add a basic smoke test that verifies both methods work without TypeError when Kalshi is enabled.

**Acceptance:** Both `get_events()` and `get_distributions()` execute without TypeError. SQL uses parameterized queries — no f-string interpolation of user input. `get_distributions()` queries `kalshi_distributions`, not `kalshi_contracts`.

---

### T2: Fix Bidirectional Outcome Upsert

**Problem:** `storage.py:550-557` — `upsert_event_outcomes()` writes to both `event_outcomes` AND `event_outcomes_first_print`. `storage.py:559-564` — `upsert_event_outcomes_first_print()` writes to both `event_outcomes_first_print` AND `event_outcomes`. This makes the two tables identical mirrors, defeating the entire first-print vs. revised outcome design. `build_event_labels()` at `events.py:399-403` supports `label_mode="first_print"` vs `"latest"` and reads from different tables, but both tables always contain the same data, rendering the mode distinction meaningless.

**File:** `kalshi/storage.py`

**Implementation:**
1. Remove the cross-table write from `upsert_event_outcomes()`:
   ```python
   def upsert_event_outcomes(self, rows: Iterable[Mapping[str, object]]):
       """Upsert event outcomes (latest/revised) into storage."""
       payload = list(rows)
       if not payload:
           return
       self._insert_or_replace("event_outcomes", payload)
       # REMOVED: self._insert_or_replace("event_outcomes_first_print", payload)
   ```
2. Remove the cross-table write from `upsert_event_outcomes_first_print()`:
   ```python
   def upsert_event_outcomes_first_print(self, rows: Iterable[Mapping[str, object]]):
       """Upsert first-print event outcomes into storage."""
       payload = list(rows)
       if not payload:
           return
       self._insert_or_replace("event_outcomes_first_print", payload)
       # REMOVED: self._insert_or_replace("event_outcomes", payload)
   ```
3. `upsert_event_outcomes_revised()` (line 566+) should continue writing only to `event_outcomes_revised` — verify it does not have the same bidirectional issue.
4. Consider adding a one-time migration note: existing databases have identical data in both tables. A cleanup script may be needed to restore the first-print distinction if historical first-print data is available from a separate source.

**Acceptance:** Calling `upsert_event_outcomes()` does NOT write to `event_outcomes_first_print`. Calling `upsert_event_outcomes_first_print()` does NOT write to `event_outcomes`. The two tables can contain different data.

---

### T3: Wire Purge/Embargo Config Into Event Feature Pipeline

**Problem:** `config.py:306-307` defines `KALSHI_PURGE_WINDOW_BY_EVENT` and `KALSHI_DEFAULT_PURGE_WINDOW` as placeholder config values, but these are never imported or used in `events.py`. The walk-forward layer (`walkforward.py:25`) has `purge_window_by_event` in its config dataclass but it defaults to an empty dict `{}`, and the pipeline runner at `run_kalshi_event_pipeline.py:182` uses the default config without injecting the event-type purge map. The outer walk-forward purge window is correctly applied, but the event-type-specific purge windows from config are not.

**Files:** `kalshi/events.py`, `kalshi/walkforward.py`, `config.py`

**Implementation:**
1. In `events.py`, import the purge config constants:
   ```python
   from ..config import (
       KALSHI_PURGE_WINDOW_BY_EVENT,
       KALSHI_DEFAULT_PURGE_WINDOW,
   )
   ```
2. In `build_event_snapshot_grid()` (events.py), apply additional purge gap between feature snapshot time and release time:
   ```python
   # After computing asof_ts = release_ts - horizon_delta:
   event_type = row.get("event_type", "")
   purge_days = KALSHI_PURGE_WINDOW_BY_EVENT.get(event_type, KALSHI_DEFAULT_PURGE_WINDOW)
   purge_delta = pd.Timedelta(days=purge_days)
   # Ensure feature snapshot is at least purge_delta before release
   asof_ts = min(asof_ts, release_ts - purge_delta)
   ```
3. Update `KALSHI_PURGE_WINDOW_BY_EVENT` and `KALSHI_DEFAULT_PURGE_WINDOW` from placeholder status to active values in config.py. Remove STATUS: PLACEHOLDER comments. Set sensible defaults:
   ```python
   KALSHI_DEFAULT_PURGE_WINDOW = 2  # days
   KALSHI_PURGE_WINDOW_BY_EVENT = {
       "CPI": 3,
       "FOMC": 1,
       "UNEMPLOYMENT": 2,
       "GDP": 3,
   }
   ```
4. In `walkforward.py`, update `EventWalkForwardConfig` to load defaults from config if not provided:
   ```python
   def __post_init__(self):
       if not self.purge_window_by_event:
           from ..config import KALSHI_PURGE_WINDOW_BY_EVENT
           self.purge_window_by_event = dict(KALSHI_PURGE_WINDOW_BY_EVENT)
       if self.default_purge_days == 10:  # sentinel for "not explicitly set"
           from ..config import KALSHI_DEFAULT_PURGE_WINDOW
           self.default_purge_days = KALSHI_DEFAULT_PURGE_WINDOW
   ```
5. Verify that the pipeline runner in `run_kalshi_event_pipeline.py` now gets the correct purge windows via the defaults.

**Acceptance:** Event-type-specific purge windows from config are applied in both the feature pipeline and walk-forward evaluation. CPI events get a 3-day purge window by default. The config constants are no longer placeholders.

---

### T4: Fix Event Strategy Stress-Regime Gate Bypass

**Problem:** `promotion.py:33-124` — `_to_backtest_result()` builds a `BacktestResult` from event returns but does not populate `regime_performance`. The `PromotionGate.evaluate_event_strategy()` in `autopilot/promotion_gate.py` checks `if regime_perf:` before applying stress-regime validation, so an empty/missing `regime_performance` dict silently skips all stress-regime checks. This means event strategies are never validated against adverse regime behavior.

**File:** `kalshi/promotion.py`

**Implementation:**
1. Compute basic regime-conditioned performance from event returns. After the `_to_backtest_result()` call in `evaluate_event_promotion()`, estimate regime performance:
   ```python
   def _compute_event_regime_performance(
       walkforward_result: EventWalkForwardResult,
   ) -> Dict[str, Dict[str, float]]:
       """Compute regime-conditioned returns from walk-forward OOS traces."""
       regime_returns: Dict[str, list] = {}
       for fold in walkforward_result.oos_traces:
           event_types = fold.get("event_types", [])
           returns = fold.get("event_returns", [])
           for etype, ret in zip(event_types, returns):
               if np.isfinite(ret):
                   regime_returns.setdefault(etype, []).append(ret)

       regime_perf = {}
       for regime, rets in regime_returns.items():
           arr = np.array(rets)
           if len(arr) >= 3:
               regime_perf[regime] = {
                   "mean_return": float(np.mean(arr)),
                   "sharpe_ratio": float(np.mean(arr) / max(np.std(arr), 1e-8)),
                   "max_drawdown": float(np.min(np.minimum.accumulate(np.cumprod(1 + arr)) / np.maximum.accumulate(np.cumprod(1 + arr)) - 1)),
                   "trade_count": len(arr),
               }
       return regime_perf
   ```
2. Attach the regime performance to the BacktestResult or pass it separately to the promotion gate:
   ```python
   regime_perf = _compute_event_regime_performance(walkforward_result)
   # Pass to promotion gate
   decision = gate.evaluate_event_strategy(
       candidate=candidate,
       backtest_result=bt_result,
       regime_performance=regime_perf,
       event_metrics=event_metrics,
   )
   ```
3. If `evaluate_event_strategy()` does not accept `regime_performance` as a separate parameter, set it on the BacktestResult object directly:
   ```python
   bt_result.regime_performance = regime_perf
   ```
4. Log when regime performance is computed and when it triggers gate violations.

**Acceptance:** Event strategies with poor performance in specific event-type regimes (e.g., negative Sharpe during FOMC events) are flagged by the stress-regime gate. The `if regime_perf:` guard in PromotionGate no longer trivially skips.

---

### T5: Fix OpenSSL Credential Exposure in Fallback Signing Path

**Problem:** Two related security issues in the OpenSSL subprocess fallback path:

1. **Passphrase in process arguments** (`client.py:223,242`): `cmd.extend(["-passin", f"pass:{self.passphrase}"])` passes the private key passphrase as a plaintext command-line argument. On Unix systems, all command-line arguments are visible via `ps aux`, `/proc/PID/cmdline`, or process monitoring tools.

2. **Private key in temp file** (`client.py:277-287`): When `private_key_path` is empty but `private_key_pem` is set, the PEM key is written to a `NamedTemporaryFile` with `delete=False`. During the window between write and `os.unlink()`, the key is readable on the filesystem. If the process crashes before cleanup, the file persists.

**Mitigating factor:** The primary signing path uses the `cryptography` library for in-process signing (`client.py:265-270`), which has neither issue. The fallback triggers only when `cryptography` is unavailable or raises an exception.

**File:** `kalshi/client.py`

**Implementation:**
1. Replace `-passin pass:` with `-passin stdin` and pipe the passphrase via stdin:
   ```python
   def _sign_with_openssl(self, message: bytes, key_path: str) -> bytes:
       """Sign message using OpenSSL subprocess with secure passphrase handling."""
       stdin_data = message
       env = dict(os.environ)

       cmd = [
           "openssl", "dgst", "-sha256", "-sign", key_path,
           "-sigopt", "rsa_padding_mode:pss",
           "-sigopt", "rsa_pss_saltlen:-1",
       ]
       if self.passphrase:
           # Use environment variable to pass passphrase securely
           env["_KALSHI_OPENSSL_PASS"] = self.passphrase
           cmd.extend(["-passin", "env:_KALSHI_OPENSSL_PASS"])

       proc = subprocess.run(
           cmd, input=stdin_data, capture_output=True, env=env,
       )
       if proc.returncode == 0:
           return proc.stdout

       # Fallback to pkeyutl
       cmd2 = [
           "openssl", "pkeyutl", "-sign", "-inkey", key_path,
           "-pkeyopt", "rsa_padding_mode:pss",
           "-pkeyopt", "rsa_pss_saltlen:-1",
           "-pkeyopt", "digest:sha256",
       ]
       if self.passphrase:
           cmd2.extend(["-passin", "env:_KALSHI_OPENSSL_PASS"])

       proc2 = subprocess.run(
           cmd2, input=stdin_data, capture_output=True, env=env,
       )
       if proc2.returncode == 0:
           return proc2.stdout

       stderr = (proc.stderr or b"") + b"\n" + (proc2.stderr or b"")
       raise RuntimeError(f"OpenSSL signing failed: {stderr.decode(errors='ignore').strip()}")
   ```
   Using `env:VAR_NAME` is preferred over `pass:` because environment variables for a child process are not visible in the process table (unlike command-line arguments).

2. Fix the temp file handling to use restricted permissions and stdin-based key passing:
   ```python
   # Option A (preferred): Pass PEM key via stdin to OpenSSL
   # This requires restructuring _sign_with_openssl to accept key via stdin
   # using -inkey /dev/stdin

   # Option B (simpler): Use mkstemp with explicit permissions
   import stat
   fd, tmp_path = tempfile.mkstemp(suffix=".pem")
   try:
       os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
       with os.fdopen(fd, "w") as f:
           f.write(str(self.private_key_pem))
       raw_sig = self._sign_with_openssl(payload, tmp_path)
   finally:
       try:
           os.unlink(tmp_path)
       except OSError:
           logger.warning("Failed to clean up temporary key file: %s", tmp_path)
   ```
3. Add a `logger.warning` when falling back to OpenSSL subprocess path, so operators are aware the less-secure path is active.

**Acceptance:** Passphrase is never visible in `ps aux` output during signing. Temp key files use `0o600` permissions. A warning is logged when the OpenSSL fallback is used.

---

### T6: Fix Early-Return Dicts Missing Schema Fields

**Problem:** `distribution.py:510-622` has three early-return paths (empty contracts, no matching quotes, insufficient data) that return dicts with only ~15 of the expected ~24 fields. Missing fields include `direction_source`, `direction_confidence`, `bin_overlap_count`, `bin_gap_mass_estimate`, `bin_support_is_ordered`, `isotonic_adjustment_magnitude`, `renormalization_delta`, `violated_constraints_pre`, `violated_constraints_post`. This causes `KeyError` or silent NaN injection when downstream code (panel builders) expects the full schema.

**File:** `kalshi/distribution.py`

**Implementation:**
1. Define a canonical empty result template with all expected fields:
   ```python
   _EMPTY_SNAPSHOT_TEMPLATE = {
       "market_id": None,
       "ts": None,
       "n_contracts": 0,
       "n_quotes_used": 0,
       "mean": np.nan,
       "std": np.nan,
       "skew": np.nan,
       "kurtosis": np.nan,
       "median": np.nan,
       "quality_score": 0.0,
       "quality_low": 1,
       "direction": None,
       "direction_source": "none",
       "direction_confidence": "none",
       "bin_overlap_count": 0,
       "bin_gap_mass_estimate": 0.0,
       "bin_support_is_ordered": 0,
       "isotonic_adjustment_magnitude": 0.0,
       "renormalization_delta": 0.0,
       "violated_constraints_pre": 0,
       "violated_constraints_post": 0,
       "monotonic_violations_pre": 0,
       "monotonic_violation_magnitude": 0.0,
       "renorm_delta": 0.0,
   }
   ```
2. Replace each early-return dict with a copy of the template, overriding only the fields that differ:
   ```python
   result = dict(_EMPTY_SNAPSHOT_TEMPLATE)
   result.update({
       "market_id": market_id,
       "ts": asof_ts,
       # ... any path-specific overrides
   })
   return result
   ```
3. This ensures all early-return paths produce schema-consistent output.
4. Add a runtime assertion or test that verifies all return paths produce dicts with the same set of keys.

**Acceptance:** All return paths from `build_distribution_snapshot()` produce dicts with the same complete set of keys. No `KeyError` when early-return results flow into panel builders.

---

## Verification

- [ ] Run `pytest kalshi/tests/ -v` — all pass
- [ ] Verify `kalshi_service.py` `get_events()` and `get_distributions()` execute without TypeError
- [ ] Verify SQL uses parameterized queries (no f-string interpolation)
- [ ] Verify `upsert_event_outcomes()` writes only to `event_outcomes`, not `event_outcomes_first_print`
- [ ] Verify purge config constants are imported and applied in event feature pipeline
- [ ] Verify event strategies with poor regime-conditioned performance are flagged by promotion gate
- [ ] Verify OpenSSL passphrase is not visible in process arguments during signing
- [ ] Verify all early-return dicts from `build_distribution_snapshot()` have the same keys as the full-path result
