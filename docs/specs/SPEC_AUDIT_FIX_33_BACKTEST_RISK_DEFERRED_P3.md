# SPEC_AUDIT_FIX_33: Backtest/Risk Deferred P3 Cleanup & Hardening

**Priority:** LOW — P3 items deferred from Audit Report 05 (Backtesting + Risk). None are correctness-critical, but they reduce code quality, observability, and maintainability over time.
**Scope:** `backtest/`, `risk/`, `config.py`, `validation/`
**Estimated effort:** 4–5 hours total (individually small)
**Depends on:** SPEC_30, SPEC_31, SPEC_32 (address critical/high issues first)
**Blocks:** Nothing
**Audit source:** Audit Report 05 (Auditor 1), findings F-27, F-30–F-36, F-38–F-39, F-41–F-50, F-53

---

## Context

After triaging all 55 findings from Audit 1 and 5 findings from Audit 2, 15 P3 items and 1 placeholder-gated P2 item remain unassigned to any actionable spec. These are dead code, misleading labels, missing logging, methodology notes, and minor enhancements. This spec captures them explicitly so they are tracked and can be addressed in a future cleanup pass without re-reading the full audit report.

---

## Tasks

### Dead Code & Unused Imports

#### T1: Remove Unused Import ALMGREN_CHRISS_RISK_AVERSION [F-30]
- **File:** `backtest/engine.py:36`
- **Action:** Remove the unused import. Verify no runtime reference exists.

#### T2: Remove Dead Branch in Regime Lookup [F-31]
- **File:** `backtest/engine.py:843-846`
- **Action:** Both branches of `if isinstance(idx, tuple)` do the identical thing. Collapse to a single branch.

#### T3: Remove Unused `field` Import [F-35]
- **File:** `risk/position_sizer.py:20`
- **Action:** Remove unused import.

---

### Typing & Validation Gaps

#### T4: Tighten risk_report and null_baselines Types [F-32]
- **File:** `backtest/engine.py:132,141`
- **Action:** Replace `Optional[object]` with proper typed dataclass references: `Optional[RiskReport]` and `Optional[NullBaselineResult]` (or define appropriate Protocol/TypedDict if the types don't exist yet).

#### T5: Add Constructor Validation in PositionSizer [F-36]
- **File:** `risk/position_sizer.py:69-81`
- **Action:** Add `__post_init__` or constructor assertions:
  ```python
  assert target_portfolio_vol > 0, "target_portfolio_vol must be positive"
  assert max_position_pct > min_position_pct, "max_position_pct must exceed min_position_pct"
  assert 0 < kelly_fraction <= 1.0, "kelly_fraction must be in (0, 1]"
  ```

---

### Performance

#### T6: Replace iterrows() in Regime Lookup Building [F-34]
- **File:** `backtest/engine.py:842`
- **Action:** Replace `iterrows()` with vectorized approach:
  ```python
  # Instead of iterrows(), use to_dict('records') or direct column access
  regime_lookup = dict(zip(regime_df.index, regime_df["regime"].values))
  ```
- **Impact:** Significant speedup for large DataFrames (iterrows is ~100× slower than vectorized).

#### T7: Fix Relative Path for CostCalibrator model_dir [F-33]
- **File:** `backtest/engine.py:259`
- **Action:** Change `model_dir=Path("trained_models")` to `model_dir=MODEL_DIR` (from config), removing working-directory dependency.

---

### Documentation & Labeling

#### T8: Document Double Uncertainty Reduction [F-38]
- **File:** `risk/position_sizer.py:264-270,525-592`
- **Action:** Add a docstring/comment explaining the intentional dual reduction:
  ```python
  # NOTE: Two uncertainty reductions are applied:
  # 1. UncertaintyGate reduces Kelly component by up to 20% (regime_uncertainty)
  # 2. _compute_uncertainty_scale() reduces composite by up to 30% (regime_entropy)
  # Combined worst-case: ~44% reduction. This is intentional — both dimensions
  # (regime model confidence and entropy) contribute independently to position sizing risk.
  ```

#### T9: Fix Attribution Module Label [F-42]
- **File:** `risk/attribution.py:4`
- **Action:** Change docstring from "Brinson-style" to "factor regression decomposition" to match the actual implementation.

#### T10: Fix Beta Fallback Comment [F-43]
- **File:** `risk/attribution.py:36`
- **Action:** Change comment from "market-neutral" to "market-tracking" since the fallback beta value is 1.0, not 0.0.

#### T11: Document Value Factor Proxy Semantics [F-44]
- **File:** `risk/factor_exposures.py:112-117`
- **Action:** The "value" factor uses negative trailing return (contrarian signal), not traditional value metrics (P/B, P/E). Add a comment:
  ```python
  # "Value" proxy: negative trailing return (mean-reversion / contrarian signal).
  # This is NOT a fundamental value factor (P/B, P/E). Consider renaming to
  # "reversal" or adding a proper fundamental value proxy when data is available.
  ```

---

### Methodology & Fairness

#### T12: Fix Factor Monitor Z-Score Scope [F-45]
- **File:** `risk/factor_monitor.py:177-211`
- **Action:** Z-scores are computed within the portfolio, not against the broader universe. Add a parameter to optionally provide universe-level statistics for proper z-scoring:
  ```python
  def compute_factor_zscores(self, portfolio_exposures, universe_stats=None):
      if universe_stats is not None:
          z = (portfolio_exposures - universe_stats["mean"]) / universe_stats["std"]
      else:
          # Fallback: within-portfolio z-scores (less meaningful)
          z = (portfolio_exposures - portfolio_exposures.mean()) / portfolio_exposures.std()
  ```

#### T13: Fix Factor Monitor Timestamp Field [F-46]
- **File:** `risk/factor_monitor.py:44,265-282`
- **Action:** The `timestamp` field in factor alerts is always empty. Populate it:
  ```python
  alert["timestamp"] = datetime.now(timezone.utc).isoformat()
  ```

#### T14: Add Transaction Costs to Null Models [F-47]
- **File:** `backtest/null_models.py:123-127`
- **Action:** Null models trade every bar with zero costs, making them easier to beat. Apply the same `TRANSACTION_COST_BPS` as the main backtest:
  ```python
  from config import TRANSACTION_COST_BPS
  cost_per_trade = TRANSACTION_COST_BPS / 10_000
  null_return = gross_return - cost_per_trade
  ```

#### T15: Wire Null Models into Validation Pipeline [F-48]
- **File:** `backtest/null_models.py`, `backtest/validation.py`
- **Action:** `compute_null_baselines()` exists but is not called from the validation pipeline. Add an optional step in `run_statistical_tests()` or as a post-validation check:
  ```python
  if null_baseline_enabled:
      null_results = compute_null_baselines(trades, returns)
      stat_result.null_comparison = null_results
  ```
- **Note:** This is also partially addressed by SPEC_08 T5 (populating null_baselines on BacktestResult).

#### T16: Increase SPA Test Default Bootstraps [F-49]
- **File:** `backtest/validation.py:719`
- **Action:** Change default `n_bootstraps=400` to `n_bootstraps=1000` for finer p-value resolution. Add a config constant:
  ```python
  SPA_BOOTSTRAPS = 1000  # STATUS: ACTIVE — backtest/validation.py; SPA bootstrap trials
  ```
- **Note:** Config already has `SPA_BOOTSTRAPS = 400`. Update the value to 1000.

---

### Code Duplication

#### T17: Deduplicate scipy Fallback Code [F-50]
- **File:** `backtest/validation.py:27-107`, `backtest/advanced_validation.py:21-34`
- **Action:** Extract the shared scipy import fallback into a common utility:
  ```python
  # In backtest/_scipy_compat.py:
  try:
      from scipy import stats as sp_stats
      from scipy.optimize import minimize as sp_minimize
  except ImportError:
      sp_stats = None
      sp_minimize = None

  def require_scipy():
      if sp_stats is None:
          raise ImportError("scipy is required for statistical validation")
  ```
  Import from both files.

---

### Data Quality

#### T18: Log Zero-Volume Days in ADV Tracker [F-53]
- **File:** `backtest/adv_tracker.py:65-67`
- **Action:** Zero-volume days are silently dropped. Add a warning when this happens frequently:
  ```python
  if volume <= 0:
      self._zero_vol_count += 1
      if self._zero_vol_count % 10 == 0:
          logger.warning("ADV tracker: %d zero-volume days dropped for %s", self._zero_vol_count, ticker)
      return  # Skip update
  ```

---

### Observability

#### T19: Add Logging to Factor Modules [F-41]
- **Files:** `risk/factor_monitor.py`, `risk/factor_portfolio.py`, `risk/cost_budget.py`
- **Action:** Add `logger = logging.getLogger(__name__)` and log key computation events (constraint violations, calibration updates, budget warnings).

---

### Placeholder-Gated (Deferred Until Feature Enabled)

#### T20: Document Almgren-Chriss Unit Incompatibility [F-27]
- **Files:** `backtest/optimal_execution.py:26`, `backtest/execution.py:181`
- **Action:** Almgren-Chriss uses `temporary_impact` in per-share units (default 0.01). ExecutionModel uses `impact_coefficient_bps` in bps per √participation (default 25.0). No conversion bridge exists.
- **Current state:** `ALMGREN_CHRISS_ENABLED = False` (placeholder), so this has no runtime impact.
- **Action for now:** Add a TODO comment and conversion stub:
  ```python
  # TODO: When ALMGREN_CHRISS_ENABLED is activated, add a conversion bridge
  # between AlmgrenChriss per-share impact units and ExecutionModel bps units.
  # See Audit Report 05, F-27 for details.
  ```
- **Action when enabled:** Create a proper unit conversion function before flipping the flag to True.

---

### size_portfolio() Enhancement

#### T21: Pass Regime and Uncertainty to size_portfolio() [F-39]
- **File:** `risk/position_sizer.py:1166-1177`
- **Action:** `size_portfolio()` calls `size_position()` without passing `regime`, `uncertainty`, or `equity` parameters. Add optional parameters:
  ```python
  def size_portfolio(
      self,
      signals: dict,
      current_positions: dict = None,
      regime: Optional[int] = None,
      regime_entropy: float = 0.0,
      portfolio_equity: float = 1_000_000.0,
  ) -> dict:
      for symbol, signal in signals.items():
          size = self.size_position(
              ...,
              regime=regime,
              regime_entropy=regime_entropy,
              portfolio_equity=portfolio_equity,
          )
  ```

---

## Verification

- [ ] `ruff check backtest/ risk/` — no unused imports flagged for items addressed here
- [ ] `grep -rn "iterrows" backtest/engine.py` — no hits after T6
- [ ] Null models include transaction costs (T14)
- [ ] Factor monitor alerts have timestamps (T13)
- [ ] All TODO comments for F-27 are present (T20)

---

## Finding Traceability

| Task | Audit Finding | Priority | Category |
|------|--------------|----------|----------|
| T1 | F-30 | P3 | Dead code |
| T2 | F-31 | P3 | Dead code |
| T3 | F-35 | P3 | Dead code |
| T4 | F-32 | P3 | Typing |
| T5 | F-36 | P3 | Validation |
| T6 | F-34 | P3 | Performance |
| T7 | F-33 | P3 | Path robustness |
| T8 | F-38 | P3 | Documentation |
| T9 | F-42 | P3 | Documentation |
| T10 | F-43 | P3 | Documentation |
| T11 | F-44 | P3 | Documentation |
| T12 | F-45 | P3 | Methodology |
| T13 | F-46 | P3 | Bug |
| T14 | F-47 | P3 | Fairness |
| T15 | F-48 | P3 | Enhancement |
| T16 | F-49 | P3 | Config |
| T17 | F-50 | P3 | DRY |
| T18 | F-53 | P3 | Data quality |
| T19 | F-41 | P3 | Observability |
| T20 | F-27 | P2 (deferred) | Placeholder-gated |
| T21 | F-39 | P3 | Enhancement |

---

*Generated from cross-audit reconciliation — 2026-02-28*
*This spec captures all 16 deferred P3 findings from Audit 1, plus 1 placeholder-gated P2.*
*No finding from either audit report is untracked.*
