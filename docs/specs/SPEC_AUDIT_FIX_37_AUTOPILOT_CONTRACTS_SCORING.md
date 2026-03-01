# SPEC_AUDIT_FIX_37: Autopilot Contract Fixes, Scoring & Hygiene

**Priority:** HIGH — API job runner crashes at runtime; promotion ranking dominated by trade count; health check reads stale schema.
**Scope:** `api/jobs/autopilot_job.py`, `autopilot/promotion_gate.py`, `autopilot/engine.py`, `autopilot/paper_trader.py`, `docs/audit/data/INTERFACE_CONTRACTS.yaml`
**Estimated effort:** 4–5 hours
**Depends on:** SPEC_05 (critical logic fixes), SPEC_06 (reliability fixes)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 8 (Autopilot/Strategy Discovery) surfaced findings that the existing SPEC_05 and SPEC_06 do not cover. SPEC_05 addresses P0 logic defects (drawdown mutation, optimizer disconnect, A/B Kelly override, meta-label leakage). SPEC_06 addresses reliability (atomic state, transition-drawdown gate, stale prices, cost clipping, registry churn, meta-label threshold, save double-write, allocator integration). This spec covers the *remaining* gaps: a broken API job entry point, promotion score scaling issues, paper_state schema mismatch, IC tracking code quality, health gate staleness, and documentation/hygiene items.

### Cross-Audit Reconciliation

| Finding | Auditor 1 (Claude) | Auditor 2 (Codex) | Existing Spec | Disposition |
|---------|--------------------|--------------------|---------------|-------------|
| run_statistical_tests contract drift | — | P0 | **SPEC_30 T1** | Already covered |
| API job runner incompatible | — | P1 | Not in any spec | **NEW → T1** |
| Drawdown mutation during sizing | — | P1 | **SPEC_05 T1** | Already covered |
| State persistence non-atomic | — | P2 | **SPEC_06 T1** | Already covered |
| paper_state.json schema mismatch | — | P2 | Not in any spec | **NEW → T2** |
| Composite score trade-count dominance | F-04 (P2) | — | Not in any spec | **NEW → T3** |
| ENSEMBLE_DISAGREEMENT try/except | F-03 (P2) | — | Not in any spec | **NEW → T4** |
| IC tracking O(n²) + float equality | F-05 (P2) | — | Not in any spec | **NEW → T5** |
| Health gate no staleness check | F-06 (P3) | — | Not in any spec | **NEW → T6** |
| INTERFACE_CONTRACTS walk_forward stale | F-01 (P1) | — | Not in any spec | **NEW → T7** |
| INTERFACE_CONTRACTS risk method wrong | F-07 (P3) | — | Not in any spec | **NEW → T7** |
| strategy_allocator.py unused | F-02 (P2) | — | **SPEC_06 T8** | Already covered |
| __init__.py missing StrategyAllocator | F-08 (P3) | — | Resolved by SPEC_06 T8 | Already covered |

---

## Tasks

### T1: Fix API Job Runner AutopilotEngine Constructor Mismatch

**Problem:** `api/jobs/autopilot_job.py:17-20` instantiates `AutopilotEngine(years=..., full_universe=...)`. The actual constructor at `autopilot/engine.py:150-163` requires `tickers: List[str]` as the first positional parameter and has no `full_universe` parameter. This causes a `TypeError` at runtime, meaning the API-triggered autopilot job path is completely broken.

**Files:** `api/jobs/autopilot_job.py`, `autopilot/engine.py`

**Implementation:**
1. Determine the intended behavior for API-triggered autopilot runs. Two options:
   - **Option A (recommended):** Add a `full_universe` convenience constructor or class method:
     ```python
     # autopilot/engine.py
     @classmethod
     def from_universe(cls, years: int = 15, full_universe: bool = False, **kwargs):
         """Create engine from configured universe (for API job entry point)."""
         if full_universe:
             from ..data.loader import load_universe
             tickers = load_universe()
         else:
             from ..data.loader import load_survivorship_universe
             tickers = load_survivorship_universe()
         return cls(tickers=tickers, years=years, **kwargs)
     ```
   - **Option B:** Fix the job to supply tickers explicitly:
     ```python
     # autopilot_job.py
     from quant_engine.data.loader import load_survivorship_universe
     tickers = load_survivorship_universe()
     engine = AutopilotEngine(tickers=tickers, years=params.get("years", 5))
     ```
2. If Option A, update the job call:
   ```python
   engine = AutopilotEngine.from_universe(
       years=params.get("years", 5),
       full_universe=params.get("full_universe", False),
   )
   ```
3. Add a basic smoke test that verifies the API job can instantiate the engine without TypeError.

**Acceptance:** `autopilot_job.py` can instantiate `AutopilotEngine` without TypeError. Both `full_universe=True` and `full_universe=False` paths work.

---

### T2: Fix paper_state.json Schema for Health Service Consumption

**Problem:** `api/services/health_service.py:1463,1476` expects top-level `equity` and per-position `value` fields from paper_state.json. The paper trader at `paper_trader.py:232-243` does not persist `equity` (only `cash`), and position records at `paper_trader.py:1171-1188` lack a `value` field. The health service uses `.get()` defaults, so it doesn't crash — but capital utilization is materially under-reported because position values are all treated as 0.

**Files:** `autopilot/paper_trader.py`, `api/services/health_service.py`

**Implementation:**
1. Add `equity` to the persisted state in `_save_state()`:
   ```python
   # paper_trader.py — _save_state()
   state = {
       "cash": self.cash,
       "equity": self._compute_equity(),  # cash + sum(position values)
       "realized_pnl": self.realized_pnl,
       "positions": self._serialize_positions(),
       "trades": self.trades[-MAX_TRADE_HISTORY:],
       "last_update": datetime.now().isoformat(),
   }
   ```
2. Add `value` to each position record in `_serialize_positions()` and/or when creating position entries:
   ```python
   position["value"] = float(position["shares"] * current_price)
   ```
   The `current_price` should come from the latest market data available during the cycle. This also feeds into SPEC_06 T3 (which adds `last_price` to positions).
3. Ensure `_compute_equity()` sums cash + all position values.
4. Coordinate with SPEC_06 T3 (stale entry prices) — once `last_price` is tracked per position, computing `value = shares * last_price` is straightforward.

**Acceptance:** `paper_state.json` contains top-level `equity` field and each position has a `value` field. Health service `_check_capital_utilization()` computes correct utilization.

---

### T3: Fix Composite Promotion Score Trade-Count Dominance

**Problem:** `promotion_gate.py:304-325` computes a composite ranking score where the trade-count term `0.01 * min(total_trades, 5000)` contributes up to +50 points, while all other terms combined typically contribute ~3-7 points. This means the ranking is almost entirely determined by trade frequency, not strategy quality.

**File:** `autopilot/promotion_gate.py`

**Implementation:**
1. Normalize the trade-count contribution to the same scale as other terms. Replace:
   ```python
   # OLD: + 0.01 * min(result.total_trades, 5000)
   # NEW: Normalize trade count to [0, 1] range, then weight comparably
   trade_score = min(result.total_trades, 5000) / 5000.0  # 0.0 to 1.0
   ```
2. Update the composite score formula with balanced weights:
   ```python
   score = (
       1.30 * result.sharpe_ratio
       + 0.80 * result.annualized_return
       + 0.35 * result.win_rate
       + 0.20 * min(result.profit_factor, 5.0)
       + 0.25 * trade_score                           # was 0.01 * raw trades
       + 0.80 * max(result.max_drawdown, -0.50)
   )
   ```
3. This gives trade count a max contribution of 0.25 (comparable to win rate's 0.35), not 50.
4. Add a comment documenting the intended weight rationale and scale of each term.
5. Consider adding a log statement that shows the decomposition of each term for debugging:
   ```python
   logger.debug("Promotion score breakdown: sharpe=%.2f, return=%.2f, wr=%.2f, pf=%.2f, trades=%.2f, dd=%.2f",
                1.30*sharpe, 0.80*ann_return, 0.35*win_rate, 0.20*pf, 0.25*trade_score, 0.80*dd)
   ```

**Acceptance:** A strategy with 5000 trades and Sharpe=0.5 does NOT outscore a strategy with 100 trades and Sharpe=2.0.

---

### T4: Move ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD to Top-Level Import

**Problem:** `engine.py:1905-1908` uses `try/except ImportError` to import `ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD` from config, with a hardcoded fallback of 0.015. The constant exists at `config.py:683` and does not require guarding. This is inconsistent with all other config imports in the same file (lines 34-53), which are normal top-level imports.

**File:** `autopilot/engine.py`

**Implementation:**
1. Move the import to the top-level import block (around line 34-53):
   ```python
   from ..config import (
       ...existing imports...,
       ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD,
   )
   ```
2. Remove the `try/except ImportError` block at lines 1905-1908.
3. Remove the hardcoded fallback value.

**Acceptance:** `ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD` is a normal top-level import. Renaming it in config.py causes an immediate ImportError (not a silent fallback).

---

### T5: Fix IC Tracking O(n²) Scan and Float Equality

**Problem:** `engine.py:1846-1858` tracks best-IC strategy by calling `max(ic_values)` inside the loop (O(n²)) and using `float(ic) == max(ic_values)` for comparison (fragile float equality). With typical candidate counts (<100) the performance impact is negligible, but the float equality can misattribute the "best" strategy.

**File:** `autopilot/engine.py`

**Implementation:**
1. Replace with running-max tracking:
   ```python
   best_ic = -float("inf")
   best_strategy_id = None
   ic_values = []

   for d in decisions:
       metrics = d.metrics if hasattr(d, "metrics") else {}
       ic = metrics.get("ic_mean")
       if ic is not None:
           ic_val = float(ic)
           ic_values.append(ic_val)
           if ic_val > best_ic:
               best_ic = ic_val
               cand = d.candidate if hasattr(d, "candidate") else None
               if cand and hasattr(cand, "strategy_id"):
                   best_strategy_id = cand.strategy_id
   ```
2. This is O(n), eliminates the float equality issue, and is more readable.

**Acceptance:** IC tracking produces correct results. No `max()` call inside the loop.

---

### T6: Add Health Gate Staleness Check in Paper Trader

**Problem:** `paper_trader.py:189-198` reads the latest health score from the health service without checking the timestamp. If the health service hasn't updated in days (outage, data staleness), the paper trader uses an arbitrarily stale score for position sizing. The degradation mode (pass-through on failure) means stale-but-available data is worse than missing data — missing data gets full position sizes, while stale bad data could permanently reduce sizes.

**File:** `autopilot/paper_trader.py`

**Implementation:**
1. Add a staleness check after retrieving the health record:
   ```python
   from datetime import datetime, timedelta

   history = svc.get_health_history(limit=1)
   if history:
       record = history[-1]
       latest_score = record.get("overall_score")
       record_ts = record.get("timestamp") or record.get("created_at")

       # Check staleness
       if record_ts is not None:
           try:
               ts = datetime.fromisoformat(str(record_ts))
               age = datetime.now() - ts.replace(tzinfo=None)
               max_age = timedelta(hours=HEALTH_GATE_MAX_STALENESS_HOURS)  # config, default 24
               if age > max_age:
                   logger.warning(
                       "Health score is %.1f hours old (max %.1f), "
                       "using pass-through sizing",
                       age.total_seconds() / 3600,
                       max_age.total_seconds() / 3600,
                   )
                   return position_size_pct  # Pass-through
           except (ValueError, TypeError):
               pass  # If timestamp parsing fails, proceed with score

       if latest_score is not None:
           self._health_risk_gate.update_health(latest_score)
           return self._health_risk_gate.apply_health_gate(
               position_size_pct, latest_score,
           )
   ```
2. Add config constant `HEALTH_GATE_MAX_STALENESS_HOURS = 24` to config.py.

**Acceptance:** A health score older than 24 hours triggers a warning and uses pass-through sizing.

---

### T7: Update INTERFACE_CONTRACTS.yaml Boundary Documentation

**Problem:** Two boundary contract entries in INTERFACE_CONTRACTS.yaml are stale:

1. **`autopilot_to_backtest_31`** (lines 360-367): Documents `walk_forward_validate(features, targets, regimes, n_folds, horizon, verbose) -> WalkForwardResult`. Actual signature at `backtest/validation.py:197-206` is `walk_forward_validate(predictions, actuals, n_folds, entry_threshold, max_overfit_ratio, purge_gap, embargo, max_train_samples) -> WalkForwardResult`.

2. **`autopilot_to_risk_30`**: Documents `PositionSizer.size_position()` (21 params). Actual call in `paper_trader.py` uses `PositionSizer.size_position_paper_trader()` (18 params, at `risk/position_sizer.py:338`).

**File:** `docs/audit/data/INTERFACE_CONTRACTS.yaml`

**Implementation:**
1. Update `autopilot_to_backtest_31` signature to match actual:
   ```yaml
   signature: |
     def walk_forward_validate(
         predictions: pd.Series, actuals: pd.Series,
         n_folds: int = 5, entry_threshold: float = 0.005,
         max_overfit_ratio: float = 2.5, purge_gap: int = 10,
         embargo: int = 5, max_train_samples: Optional[int] = None,
     ) -> WalkForwardResult
   ```
2. Add `size_position_paper_trader` as a separate contract entry under `autopilot_to_risk_30`, or update the existing entry to reflect the actual method used.

**Acceptance:** INTERFACE_CONTRACTS.yaml entries match actual function signatures.

---

## Verification

- [ ] Run `pytest tests/ -k "autopilot or paper_trader or promotion"` — all pass
- [ ] Verify `autopilot_job.py` can instantiate AutopilotEngine without TypeError
- [ ] Verify paper_state.json contains `equity` and position `value` fields
- [ ] Verify promotion score for high-trade/low-Sharpe strategy < score for low-trade/high-Sharpe strategy
- [ ] Verify ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD import is top-level
- [ ] Verify IC tracking has no `max()` inside loop
- [ ] Verify stale health score (>24h) triggers pass-through
