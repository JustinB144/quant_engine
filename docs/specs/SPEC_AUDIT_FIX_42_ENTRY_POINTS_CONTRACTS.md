# SPEC_AUDIT_FIX_42: Entry Points — Backtest Artifacts, SPA Parity & Cache Path Fixes

**Priority:** HIGH — Zero-trade runs leave misleading stale artifacts; SPA validation threshold mismatch produces unreliable statistical inference; Alpaca intraday data invisible to default loader.
**Scope:** `run_backtest.py`, `api/orchestrator.py`, `scripts/alpaca_intraday_download.py`, `scripts/ibkr_intraday_download.py`, `scripts/ibkr_daily_gapfill.py`, `scripts/alpaca_intraday_download.py`, `data/local_cache.py`
**Estimated effort:** 4–5 hours
**Depends on:** SPEC_30 T4 (backtest summary 17-field contract)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 11 (Entry Points & Scripts) identified findings not covered by existing specs. SPEC_30 T4 already covers the backtest summary schema drift (emitting all 17 declared fields). This spec covers the remaining high-priority gaps: zero-trade runs leaving stale artifacts, SPA validation using a mismatched confidence threshold, Alpaca intraday data written to a non-canonical cache root invisible to the default loader, and OHLCV column merge logic that silently drops required columns.

### Cross-Audit Reconciliation

| Finding | Auditor 1 (Codex) | Auditor 2 (Claude Opus) | Existing Spec | Disposition |
|---------|-------------------|------------------------|---------------|-------------|
| Backtest summary 11 vs 17 fields | F-01 (P1) | F1 (P2) + F2 (P2) | **SPEC_30 T4** | Already covered |
| Zero-trade stale artifacts | F-02 (P1) | — | Not in any spec | **NEW → T1** |
| SPA confidence threshold mismatch | F-03 (P2) | — | Not in any spec | **NEW → T2** |
| Alpaca writes to non-canonical cache | F-06 (P2) | — | Not in any spec | **NEW → T3** |
| Merge logic drops OHLCV columns | F-07 (P2) | — | Not in any spec | **NEW → T4** |

### Severity Reconciliation

Auditor 1 rated F-01 (backtest summary drift) as P1, while Auditor 2 rated it P2 (spec documentation error). Verification confirms the consumers do passthrough `json.load()` without field-level access, so runtime impact is low — but the INTERFACE_CONTRACTS.yaml being wrong is a contract integrity issue. SPEC_30 T4 already addresses this by adding the 6 missing fields to both writers and updating the contract.

Auditor 2 found additional issues Auditor 1 missed: run_server.py import pattern inconsistency (F3), ibkr_daily_gapfill.py private function imports (F6), and private method access in run_retrain.py (F11).

---

## Tasks

### T1: Clear Stale Artifacts on Zero-Trade Backtest Runs

**Problem:** `run_backtest.py:396-442` writes summary and trade CSV files ONLY when `result.total_trades > 0`. There is no `else` branch to clear or rewrite these files when zero trades occur. If a previous run produced artifacts at `results/backtest_10d_summary.json` and `results/backtest_10d_trades.csv`, a subsequent zero-trade run leaves those files untouched. Operators and dashboards see stale results with no indication they're from a different run. Same pattern in `api/orchestrator.py:332-363`.

**Files:** `run_backtest.py`, `api/orchestrator.py`

**Implementation:**
1. Add an `else` branch that writes a zero-trade summary and clears the trade file:
   ```python
   if result.total_trades > 0:
       # ... existing trade/summary write logic ...
   else:
       # Write zero-trade summary to prevent stale artifacts
       summary = {
           "horizon": args.horizon,
           "total_trades": 0,
           "win_rate": 0.0,
           "avg_return": 0.0,
           "sharpe": 0.0,
           "sortino": 0.0,
           "max_drawdown": 0.0,
           "profit_factor": 0.0,
           "annualized_return": 0.0,
           "trades_per_year": 0.0,
           "regime_breakdown": {},
           # Include the 6 additional fields from SPEC_30 T4:
           "winning_trades": 0,
           "losing_trades": 0,
           "avg_win": 0.0,
           "avg_loss": 0.0,
           "total_return": 0.0,
           "avg_holding_days": 0.0,
       }
       summary_path = RESULTS_DIR / f"backtest_{args.horizon}d_summary.json"
       with open(summary_path, "w") as f:
           json.dump(summary, f, indent=2, default=str)

       # Write empty trade CSV to prevent stale trade data
       trades_path = RESULTS_DIR / f"backtest_{args.horizon}d_trades.csv"
       pd.DataFrame().to_csv(trades_path, index=False)

       if verbose:
           print("  Zero trades — wrote empty summary and cleared trade file.")
   ```
2. Apply the same pattern to `api/orchestrator.py` backtest path.
3. Coordinate with SPEC_30 T4 to ensure the zero-trade summary includes all 17 declared fields.

**Acceptance:** After a zero-trade backtest run, the summary file exists with `total_trades: 0` and the trade CSV is empty. No stale data from previous runs persists.

---

### T2: Align SPA Validation Confidence Threshold With Backtester

**Problem:** `run_backtest.py:338` sets `effective_conf = float(args.min_confidence if args.min_confidence is not None else 0.0)` — defaulting to `0.0`. This is passed to `strategy_signal_returns()` at line 366 as `min_confidence=effective_conf`. However, the backtester itself uses `self.confidence_threshold` which defaults to `CONFIDENCE_THRESHOLD` from config (typically `0.6`). This means SPA validation evaluates a signal-returns series that includes ALL predictions (confidence >= 0.0), while the actual backtest only trades on predictions with confidence >= 0.6. The SPA bootstrap result does not reflect the actual traded strategy.

**File:** `run_backtest.py`

**Implementation:**
1. Change the default `effective_conf` to use the same config constant as the backtester:
   ```python
   from quant_engine.config import CONFIDENCE_THRESHOLD

   effective_conf = float(
       args.min_confidence if args.min_confidence is not None
       else CONFIDENCE_THRESHOLD
   )
   ```
2. If `args.min_confidence` is explicitly provided via CLI, use it — this allows deliberate override for research purposes.
3. Add a log message when the default is used:
   ```python
   if args.min_confidence is None:
       logger.info(
           "SPA: Using backtester confidence threshold %.2f (from config CONFIDENCE_THRESHOLD)",
           effective_conf,
       )
   ```
4. Consider doing the same for `effective_entry` to ensure consistency with the backtester's entry threshold.

**Acceptance:** With no CLI override, SPA validation uses the same confidence threshold as the backtester. The SPA result reflects the actual traded strategy's signal filter.

---

### T3: Fix Alpaca Intraday Cache Root Mismatch

**Problem:** `scripts/alpaca_intraday_download.py:59` writes data to `DATA_CACHE_ALPACA_DIR` (e.g., `data/cache_alpaca/`), but the default intraday loader at `data/local_cache.py:391` reads from `DATA_CACHE_DIR` (e.g., `data/cache/`). Data downloaded by the Alpaca script is invisible to the standard feature pipeline and backtester unless callers explicitly pass a custom `cache_dir` parameter.

`config.py:40-41` confirms these are separate directories:
```python
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"
DATA_CACHE_ALPACA_DIR = ROOT_DIR / "data" / "cache_alpaca"
```

**Files:** `scripts/alpaca_intraday_download.py`, `data/local_cache.py`

**Implementation:**

Option A (preferred — make default loader aware of Alpaca cache):
1. In `local_cache.py`, add `DATA_CACHE_ALPACA_DIR` to the fallback search path for intraday data:
   ```python
   from ..config import DATA_CACHE_DIR, DATA_CACHE_ALPACA_DIR, FALLBACK_SOURCE_DIRS

   def load_intraday_ohlcv(ticker, timeframe="5m", cache_dir=None):
       d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR / "intraday"
       t = ticker.upper()

       # Search primary + Alpaca + fallback dirs
       search_roots = [d]
       alpaca_intraday = DATA_CACHE_ALPACA_DIR / "intraday"
       if alpaca_intraday.exists() and alpaca_intraday != d:
           search_roots.append(alpaca_intraday)
       for fallback in FALLBACK_SOURCE_DIRS:
           fb = Path(fallback) / "intraday"
           if fb.exists() and fb not in search_roots:
               search_roots.append(fb)

       for root in search_roots:
           # existing pattern-matching logic
           ...
   ```
   This also addresses SPEC_18 T5 (fallback dirs for intraday).

Option B (simpler — write Alpaca data to canonical dir):
1. Change `alpaca_intraday_download.py` to write directly to `DATA_CACHE_DIR / "intraday"` instead of `DATA_CACHE_ALPACA_DIR`.
2. This loses the source-specific directory separation, which may be undesirable for provenance.

**Recommendation:** Use Option A. Source-specific directories are valuable for provenance tracking, and adding them to the search path is cleaner than losing that separation.

**Acceptance:** Intraday data downloaded by the Alpaca script is discoverable by `load_intraday_ohlcv()` without requiring a custom `cache_dir` argument.

---

### T4: Add OHLCV Column Assertion to Merge Logic in Data Utility Scripts

**Problem:** `alpaca_intraday_download.py:699-703` and `ibkr_daily_gapfill.py:220-225` merge new data with existing cached data using column intersection: `[c for c in REQUIRED_OHLCV if c in old.columns and c in df.columns]`. If either DataFrame is missing an OHLCV column, that column is silently dropped from the merged output. The downstream feature pipeline and backtester expect all OHLCV columns to be present.

Also affects `ibkr_intraday_download.py:482-487` (same pattern).

**Files:** `scripts/alpaca_intraday_download.py`, `scripts/ibkr_intraday_download.py`, `scripts/ibkr_daily_gapfill.py`

**Implementation:**
1. After computing the column intersection, assert all required OHLCV columns are present:
   ```python
   REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

   cols = [c for c in REQUIRED_OHLCV if c in old.columns and c in df.columns]
   missing = set(REQUIRED_OHLCV) - set(cols)
   if missing:
       logger.warning(
           "%s: Merge would drop required OHLCV columns %s — skipping merge, using new data only",
           ticker, missing,
       )
       # Fallback: use new data only (which should have all columns)
       merged = df[REQUIRED_OHLCV] if all(c in df.columns for c in REQUIRED_OHLCV) else df
   else:
       merged = pd.concat([old[cols], df[cols]]).sort_index()
       merged = merged[~merged.index.duplicated(keep="last")]
   ```
2. Apply to all three scripts: `alpaca_intraday_download.py`, `ibkr_intraday_download.py`, `ibkr_daily_gapfill.py`.
3. Add a post-merge assertion:
   ```python
   assert all(c in merged.columns for c in REQUIRED_OHLCV), \
       f"Post-merge OHLCV assertion failed: missing {set(REQUIRED_OHLCV) - set(merged.columns)}"
   ```

**Acceptance:** If an existing cached file is missing the "Volume" column, the merge does NOT silently drop Volume from the output. A warning is logged and the merge falls back to using only the new data.

---

## Verification

- [ ] Run a zero-trade backtest → verify summary file exists with `total_trades: 0` and trade CSV is empty
- [ ] Run SPA validation without CLI confidence override → verify it uses `CONFIDENCE_THRESHOLD` (0.6), not 0.0
- [ ] Download Alpaca intraday data → verify `load_intraday_ohlcv()` finds it without explicit cache_dir
- [ ] Merge data with a degraded cached file (missing Volume) → verify Volume is NOT silently dropped
- [ ] Run `pytest tests/ -k "backtest"` — all pass
