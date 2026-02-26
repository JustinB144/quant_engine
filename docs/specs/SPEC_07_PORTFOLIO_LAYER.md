# Feature Spec: Portfolio Layer + Regime-Conditioned Constraints

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 120 hours across 8 tasks

---

## Why

Portfolio risk constraints in `/mnt/quant_engine/engine/portfolio_risk.py` are currently **static** — they do not adapt to market regime. Sector caps (40%), correlation limits (0.85), and gross exposure (100%) remain constant regardless of market conditions, volatility regimes, or stress scenarios. Under adverse regimes (flash crash, COVID, stagflation), static constraints are inadequate:

- Sector caps should tighten to prevent concentration risk contagion.
- Correlation limits should reflect actual correlation structure (which changes regime-to-regime per `/mnt/quant_engine/engine/covariance.py`).
- Turnover penalties should scale with execution costs, which vary with market liquidity.
- Factor exposure constraints (beta, size, value, momentum) are entirely absent, allowing unintended tilts.
- Universe metadata (sector mapping, liquidity) is hardcoded in `_resolve_sector()` instead of centralized.

Furthermore, the portfolio optimizer uses SLSQP (sequential least squares programming) which is prone to local optima for non-convex problems; the sizing layer (`PositionSizer.size_portfolio_aware()`) exists but is not integrated with `PortfolioRiskManager`'s binary gating, meaning we're not using regime-aware sizing backoff.

This spec unifies regime-conditioned constraints, factor limits, sector metadata, and sizing backoff under a single coherent framework.

---

## What

Implement a **Regime-Conditioned Portfolio Risk Manager** that:

1. **Adapts constraint tightness** based on detected market regime (normal vs. stress).
   - Base constraints (reference): 40% sector, 0.85 correlation, 100% gross, 500% annualized turnover.
   - Stress multipliers: 0.6x sector cap (24%), 0.7x correlation (0.60), 0.8x gross (80%), 0.5x turnover (250%).

2. **Uses regime-conditional covariance** from `compute_regime_covariance()` in `/mnt/quant_engine/engine/covariance.py` rather than global covariance.

3. **Enforces factor exposure bounds** for market beta (0.8–1.2), size (unconstrained but monitored), value (monitored), momentum (monitored), volatility (0.8–1.2 in normal regimes, 0.5–1.0 in stress).

4. **Centralizes universe metadata** (sectors, liquidity tiers, borrowability) in a configuration file `/mnt/quant_engine/config/universe.yaml` instead of hardcoded `GICS_SECTORS`.

5. **Integrates sizing backoff** so that when constraints are near binding, position sizes are automatically reduced (via `PositionSizer.size_portfolio_aware()`) rather than gated binary.

6. **Implements constraint tightening replay** for stress testing: given historical stress events, replay the portfolio and re-optimize under stress constraints to measure robustness.

---

## Constraints

### Must-haves

- Regime-conditioned constraint multipliers derive from regime state (`regime_state.regime` from `/mnt/quant_engine/engine/regime/detector.py`).
- Covariance estimation respects regime via `compute_regime_covariance(regime_state)`.
- All factor exposures (beta, size, value, momentum, volatility) are computed and checked pre-check.
- Sector constraints reference universe metadata, not hardcoded constants.
- Backoff sizing is continuous (not binary) when constraints approach binding.
- Constraint tightness transitions are smooth across regime changes (no step discontinuities).

### Must-nots

- **Do not** use global covariance when regime-conditional matrices exist.
- **Do not** gate positions binary; always prefer continuous sizing backoff.
- **Do not** hardcode sector names, liquidity tiers, or borrowability lists.
- **Do not** modify the portfolio optimizer objective function in ways that break existing SLSQP behavior for backward compatibility; use constraint relaxation instead.
- **Do not** implement new covariance estimation; reuse `compute_regime_covariance()` from `/mnt/quant_engine/engine/covariance.py`.

### Out of scope

- Alternative optimization solvers (e.g., interior-point, genetic algorithm). SLSQP will remain the default.
- Intraday rebalancing logic; constraint monitoring is daily close.
- Transaction cost modeling beyond the existing turnover penalty in optimizer.
- Liquidity-aware execution (order splitting, VWAP, TWAP). This is execution layer.
- Portfolio-level counterparty/credit risk; single-name credit checks are handled by data integrity layer.

---

## Current State

### Key files

- **`/mnt/quant_engine/engine/portfolio_risk.py`** (330 lines): `PortfolioRiskManager` class with checks for single-name (10%), gross (100%), sector (40%), correlation (0.85). Has `_resolve_sector()` with hardcoded GICS_SECTORS dict.
- **`/mnt/quant_engine/engine/portfolio_optimizer.py`** (277 lines): Mean-variance optimizer with L1 turnover penalty, sector neutrality objective, SLSQP solver. `MAX_ANNUALIZED_TURNOVER=500` is NOT enforced as hard cap.
- **`/mnt/quant_engine/engine/covariance.py`** (356 lines): Ledoit-Wolf, EWMA, sample estimation. Key function: `compute_regime_covariance(prices, regime_labels, regime_id)` returns per-regime covariance matrix.
- **`/mnt/quant_engine/engine/regime/detector.py`**: Detects market regime, exposes `regime_state` with `.regime` (0–3 for 4 regimes).
- **`/mnt/quant_engine/engine/position_sizer.py`**: Class `PositionSizer` with method `size_portfolio_aware(weights, constraints, backoff_factor)` for continuous sizing.
- **`/mnt/quant_engine/engine/stress_test.py`** (548 lines): 5 macro scenarios (2008, COVID, 2022, flash crash, stagflation). Useful for constraint tightening replay validation.

### Existing patterns to follow

- `check_new_position()` pattern in `PortfolioRiskManager`: return (pass: bool, reason: str, utilization: float).
- Regime-aware functions signature: `func(regime_state, ...)` or `func(..., regime=regime_state.regime)`.
- Configuration via YAML and environment overrides (see `/mnt/quant_engine/config/`).
- Use `_resolve_sector()` pattern but externalize data to config, not hardcode.

### Configuration

**New config file: `/mnt/quant_engine/config/universe.yaml`**

```yaml
sectors:
  "Technology": [5, 10, 15]  # GICS codes
  "Financials": [20, 25]
  "Healthcare": [30, 35]
  "Industrials": [40, 45]
  "Energy": [50]
  "Materials": [55]
  "Consumer Discretionary": [60, 65]
  "Consumer Staples": [70]
  "Utilities": [80]
  "Real Estate": [85]
  "Communication Services": [90]

liquidity_tiers:
  "Mega":   {market_cap_min: 200e9, dollar_volume_min: 1e8}
  "Large":  {market_cap_min: 10e9, dollar_volume_min: 1e6}
  "Mid":    {market_cap_min: 2e9, dollar_volume_min: 500e3}
  "Small":  {market_cap_min: 300e6, dollar_volume_min: 100e3}

borrowability:
  hard_to_borrow: ["TSLA", "GME", "AMC"]  # overrides from hard-to-borrow list
  restricted: ["ORCL"]  # no short selling

constraint_base:
  sector_cap: 0.40
  correlation_limit: 0.85
  gross_exposure: 1.00
  single_name_cap: 0.10
  annualized_turnover_max: 5.00  # 500%

stress_multipliers:
  normal: {sector_cap: 1.0, correlation_limit: 1.0, gross_exposure: 1.0, turnover: 1.0}
  stress: {sector_cap: 0.6, correlation_limit: 0.7, gross_exposure: 0.8, turnover: 0.5}

factor_limits:
  beta: {normal: [0.8, 1.2], stress: [0.9, 1.1]}
  volatility: {normal: [0.8, 1.2], stress: [0.5, 1.0]}
  size: {normal: null, stress: null}  # monitored but not constrained
  value: {normal: null, stress: null}
  momentum: {normal: null, stress: null}

backoff_policy:
  mode: "continuous"  # vs "binary"
  backoff_factors: [0.9, 0.7, 0.5, 0.25]  # applied when constraint utilization crosses [70%, 80%, 90%, 95%]
```

**Environment override in `/mnt/quant_engine/config/.env`**:

```
CONSTRAINT_STRESS_SECTOR_CAP=0.6
CONSTRAINT_STRESS_CORRELATION_LIMIT=0.7
FACTOR_BETA_MIN=0.8
FACTOR_BETA_MAX=1.2
```

---

## Tasks

### T1: Centralize Universe Metadata

**What:** Extract hardcoded sector names, liquidity tiers, and borrowability lists from `/mnt/quant_engine/engine/portfolio_risk.py` into `/mnt/quant_engine/config/universe.yaml`. Create a `UniverseConfig` class to load, validate, and query this data.

**Files:**
- Create `/mnt/quant_engine/config/universe.yaml` (template above).
- Create `/mnt/quant_engine/engine/universe_config.py` (new):
  - Class `UniverseConfig`:
    - `__init__(path: str)`: Load YAML and validate schema.
    - `get_sector(ticker: str) -> str`: Return sector name or raise KeyError.
    - `get_liquidity_tier(ticker: str, market_cap: float, dollar_volume: float) -> str`: Return tier.
    - `is_hard_to_borrow(ticker: str) -> bool`.
    - `is_restricted(ticker: str) -> bool`.
    - `get_sector_constituents(sector: str) -> List[str]`: Return all tickers in sector.
  - Tests in `/mnt/quant_engine/tests/test_universe_config.py`.
- Modify `/mnt/quant_engine/engine/portfolio_risk.py`:
  - Remove `GICS_SECTORS` hardcoded dict and `_resolve_sector()` method.
  - Replace with `self.universe_config.get_sector(ticker)`.

**Implementation notes:**
- YAML schema validation: sector codes must be integers, liquidity tiers must have min_cap and min_volume, borrowability lists must be lists of strings.
- Use `pydantic` or `jsonschema` for validation.
- Load config once in `__init__()`, cache in memory (reload on config file mtime change if needed).
- Fallback: if universe.yaml missing, raise informative error (do not silently use old hardcoded values).

**Verify:**
- `test_universe_config.py` passes with sample universe.yaml.
- `PortfolioRiskManager` initializes with `UniverseConfig` and `get_sector(ticker)` returns correct sector.
- All sector references in `portfolio_risk.py` now use `universe_config` (no more `_resolve_sector`).

---

### T2: Regime-Conditioned Constraint Multipliers

**What:** Extend `PortfolioRiskManager` to read constraint multipliers from regime state and apply them to all checks. Create a `ConstraintMultiplier` class that computes multipliers based on regime.

**Files:**
- Modify `/mnt/quant_engine/engine/portfolio_risk.py`:
  - Create nested class `ConstraintMultiplier`:
    - `__init__(config: Dict)`: Load base constraints and stress multipliers from universe.yaml.
    - `get_multipliers(regime_state: RegimeState) -> Dict`: Return {sector_cap, correlation_limit, gross_exposure, turnover} multipliers for current regime.
    - `is_stress_regime(regime_state: RegimeState) -> bool`: Return True if regime is stress (regimes 2 or 3 per detector.py).
  - Modify `PortfolioRiskManager.__init__()` to instantiate `ConstraintMultiplier`.
  - Modify `check_*()` methods to call `self.multiplier.get_multipliers(regime_state)` and apply to constraint thresholds.
  - Signature change: All `check_*()` methods now accept `regime_state: RegimeState` parameter.
- Tests in `/mnt/quant_engine/tests/test_portfolio_risk_regime.py`:
  - Test that multipliers = 1.0 in normal regime, < 1.0 in stress regime.
  - Test that all checks respect multiplied constraints.

**Implementation notes:**
- Regime mapping: assume regime 0–1 = normal, regime 2–3 = stress (align with detector.py enum).
- Multipliers should be applied multiplicatively: `effective_sector_cap = base_sector_cap * multiplier`.
- Smooth transitions: when regime changes, apply multiplier over 1–2 trading days (exponential smoothing) to avoid sharp position liquidations.

**Verify:**
- `test_portfolio_risk_regime.py` passes: sector cap check uses 0.40 in normal, 0.24 in stress.
- Correlation check uses 0.85 in normal, 0.595 (0.85 * 0.7) in stress.
- Turnover penalty scales correctly in optimizer.

---

### T3: Regime-Conditional Covariance in Risk Checks

**What:** Replace global covariance matrix in `PortfolioRiskManager.check_correlation()` with regime-conditional covariance from `compute_regime_covariance()`.

**Files:**
- Modify `/mnt/quant_engine/engine/portfolio_risk.py`:
  - Update `check_correlation()` to call `compute_regime_covariance(prices, regime_labels, regime_state.regime)` instead of using global cov matrix.
  - Signature change: `check_correlation(..., prices: pd.DataFrame, regime_labels: np.ndarray, regime_state: RegimeState)`.
  - Cache regime-conditional cov matrices (one per regime) to avoid recomputation.
- Modify `/mnt/quant_engine/engine/covariance.py`:
  - Ensure `compute_regime_covariance()` returns Positive-Definite matrix (already does via Ledoit-Wolf).
  - Export utility `nearest_psd_matrix()` for external use if needed.
- Update call sites in `/mnt/quant_engine/engine/trainer.py`, `/mnt/quant_engine/engine/paper_trader.py` to pass prices and regime_labels to `check_correlation()`.

**Implementation notes:**
- Covariance matrices are expensive to compute; cache them keyed by (regime, window_size, method).
- If regime-conditional matrix has insufficient samples (< 50 observations), fall back to global cov with warning.
- Correlation limit should be applied element-wise: if any pair exceeds limit, fail check.

**Verify:**
- Unit tests: call `check_correlation()` with regime-conditional prices and verify against regime-conditional cov matrix.
- Integration test: run trainer on historical data with regime changes, verify that correlation limits adapt.

---

### T4: Factor Exposure Constraints

**What:** Compute portfolio factor exposures (beta, size, value, momentum, volatility) and enforce bounds. Create `FactorExposureManager` to compute and check exposures.

**Files:**
- Create `/mnt/quant_engine/engine/factor_exposures.py` (new):
  - Class `FactorExposureManager`:
    - `__init__(config: Dict)`: Load factor limits from universe.yaml.
    - `compute_exposures(weights: np.ndarray, factor_matrix: pd.DataFrame) -> Dict[str, float]`: Return {beta, size, value, momentum, volatility}.
    - `check_factor_bounds(exposures: Dict, regime_state: RegimeState) -> Tuple[bool, Dict]`: Return (pass, violations).
    - Helper: `_compute_beta()`, `_compute_size()`, `_compute_value()`, `_compute_momentum()`, `_compute_volatility()`.
  - Factors sourced from fama-french or internal calculation (see `/mnt/quant_engine/engine/feature_engineering.py`).
- Modify `/mnt/quant_engine/engine/portfolio_risk.py`:
  - Add `FactorExposureManager` instance.
  - Add method `check_factor_exposures(weights, factor_matrix, regime_state)` that calls `FactorExposureManager.check_factor_bounds()`.
  - Integrate into `check_portfolio()` workflow.
- Tests in `/mnt/quant_engine/tests/test_factor_exposures.py`.

**Implementation notes:**
- Factor matrix can be pre-computed or fetched from Kenneth French data library.
- Beta computed as portfolio_return correlation / market_return variance.
- Size, value, momentum, volatility computed as weighted average of single-name factor loadings.
- Bounds can be tight (e.g., beta 0.8–1.2) or loose (null = monitored only).
- In stress regime, volatility bounds tighten.

**Verify:**
- Compute factor exposures on S&P 500 portfolio: beta ~1.0, volatility ~1.0.
- Enforce beta bounds; verify that portfolio with intentional short volatility-factor tilts is rejected in stress regime.

---

### T5: Integrating Sizing Backoff with Risk Manager

**What:** Connect `PositionSizer.size_portfolio_aware()` to `PortfolioRiskManager` so that when constraints approach binding, position sizes are automatically reduced rather than gated binary.

**Files:**
- Modify `/mnt/quant_engine/engine/position_sizer.py`:
  - Add method `size_with_backoff(weights: np.ndarray, constraint_utilization: Dict[str, float], backoff_policy: Dict) -> np.ndarray`:
    - Takes utilization dict (e.g., {sector_cap_util: 0.92, correlation_util: 0.85}).
    - Returns scaled weights based on backoff_factors in config.
    - Example: if sector_cap_util = 0.92 (>90%), apply 0.25x backoff.
- Modify `/mnt/quant_engine/engine/portfolio_risk.py`:
  - Update `check_portfolio()` to return not just (pass: bool) but (pass: bool, constraint_util: Dict, recommended_weights: np.ndarray).
  - Add method `compute_constraint_utilization()` that returns {sector_cap_util, correlation_util, gross_util, ...}.
  - If any constraint utilization > 70%, call `size_with_backoff()` to reduce weights.
- Update call sites to handle new return signature.

**Implementation notes:**
- Constraint utilization = (current / limit). Example: if current sector weight = 0.38 and limit = 0.40, util = 0.95.
- Backoff factors are cumulative: if sector and correlation both trigger backoff, apply product of factors.
- Recommend weights rather than fail hard; allow trader discretion but warn.

**Verify:**
- Build a portfolio with sector weights at 95% of cap; verify backoff weights are lower.
- Run end-to-end: positions recommended, then sized with backoff, then passed to paper trader.

---

### T6: Constraint Tightening Replay for Stress Testing

**What:** Given historical stress events (from `stress_test.py`), replay portfolio construction and re-optimize under stress constraints to measure robustness.

**Files:**
- Create `/mnt/quant_engine/engine/constraint_tightening_replay.py` (new):
  - Function `replay_with_stress_constraints(portfolio_history: List[Dict], stress_scenarios: List[StressScenario], regime_conditional: bool = True) -> pd.DataFrame`:
    - For each portfolio in history, check what constraints would have been violated if stress were active.
    - Return DataFrame with columns: date, scenario, sector_cap_violated, correlation_violated, gross_violated, turnover_violated, constraint_util.
  - Integrate with `/mnt/quant_engine/engine/stress_test.py` to use existing scenarios (2008, COVID, 2022, flash crash, stagflation).
- Tests: verify that stress constraints catch portfolios that would have suffered in 2008, COVID.

**Implementation notes:**
- Replay requires historical position data (from paper trader logs or backtest).
- For each date, compute constraint multipliers as-if regime were stress, and re-check all constraints.
- Output: summary table showing max constraint utilization per scenario and overall robustness score.

**Verify:**
- Run replay on 2007–2009 backtest window; verify that stress constraints would have reduced position sizes before 2008 crisis.
- Compare portfolio PnL with/without stress constraints activated.

---

### T7: Smooth Constraint Transitions Across Regime Changes

**What:** Implement exponential smoothing for constraint multipliers when regime changes to avoid abrupt position liquidations.

**Files:**
- Modify `/mnt/quant_engine/engine/portfolio_risk.py` (ConstraintMultiplier class):
  - Add `_smoothed_multipliers: Dict[str, float]` instance variable.
  - Add method `get_multipliers_smoothed(regime_state, alpha=0.3) -> Dict`: Apply exponential smoothing: `smoothed = alpha * current + (1 - alpha) * previous`.
  - Track regime change timestamp and apply smoothing over 1–2 days.
- Modify call sites to use `get_multipliers_smoothed()` instead of `get_multipliers()`.

**Implementation notes:**
- Smoothing parameter alpha: default 0.3 (half-life ~2 days).
- On regime change day 1, constraint multiplier is 30% of new, 70% of old. Day 2, ~51% new. Day 3, ~65% new.
- Can be controlled via environment variable: `CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA`.

**Verify:**
- Simulate regime transition (0→2) and verify that constraint multipliers transition smoothly.
- Verify that position sizes do not drop abruptly (gradual reduction over 1–2 days).

---

### T8: Documentation and Integration Testing

**What:** Write detailed documentation for the regime-conditioned portfolio layer, including config reference, API documentation, and integration test suite.

**Files:**
- Create `/mnt/quant_engine/docs/portfolio_layer_guide.md`:
  - Architecture overview (PortfolioRiskManager → ConstraintMultiplier → sizing backoff).
  - Config reference (universe.yaml, constraint_base, stress_multipliers, factor_limits).
  - API reference: all public methods of PortfolioRiskManager, UniverseConfig, FactorExposureManager.
  - Examples: how to check a portfolio, interpret constraint utilization, handle sizing backoff.
- Create `/mnt/quant_engine/tests/test_portfolio_integration.py`:
  - End-to-end test: generate random prices, fit regime, compute weights, check portfolio, apply sizing backoff, verify no constraints violated.
  - Stress test: run constraint replay on 2008 scenario, verify robustness.
  - Regime change test: transition from normal to stress regime, verify smooth constraint scaling.
- Update `/mnt/quant_engine/docs/README.md` with link to portfolio_layer_guide.

**Implementation notes:**
- Integration tests should be parametrized over multiple scenarios (normal, stress, regime transitions).
- Document the backoff_policy config thoroughly; it's a critical knob.

**Verify:**
- All integration tests pass.
- Documentation is clear and example code runs without error.
- `pytest /mnt/quant_engine/tests/test_portfolio_integration.py -v` shows green.

---

## Validation

### Acceptance criteria

1. **Centralized universe metadata:** All sector, liquidity, borrowability data loaded from `/mnt/quant_engine/config/universe.yaml`. No hardcoded GICS_SECTORS in code.
2. **Regime-conditioned constraints:** All checks (sector, correlation, gross, turnover, factor) respect regime-dependent multipliers from `ConstraintMultiplier`.
3. **Regime-conditional covariance:** `check_correlation()` uses covariance matrix computed per-regime via `compute_regime_covariance()`, not global.
4. **Factor exposure checks:** Portfolio beta, volatility bounded per config; size, value, momentum monitored and reported.
5. **Sizing backoff:** When constraint utilization > 70%, `PositionSizer.size_with_backoff()` reduces weights (continuous, not binary gating).
6. **Smooth transitions:** Constraint multipliers transition smoothly (exponential smoothing) across regime changes over 1–2 days.
7. **Stress replay:** `/mnt/quant_engine/engine/constraint_tightening_replay.py` successfully replays historical portfolios under stress constraints; output shows no catastrophic constraint violations.
8. **Tests pass:** All unit and integration tests in `/mnt/quant_engine/tests/test_portfolio_*.py` pass. Coverage > 85%.
9. **Documentation:** `/mnt/quant_engine/docs/portfolio_layer_guide.md` is comprehensive and all code examples run.

### Verification steps

1. Load `universe.yaml` and instantiate `UniverseConfig(path)`. Verify `get_sector("AAPL")` returns "Technology".
2. Instantiate `PortfolioRiskManager(universe_config)` in normal regime and stress regime. Verify constraint thresholds differ by expected multipliers.
3. Create a portfolio with 95% sector utilization in Technology. Run `check_portfolio()` with stress regime; verify sizing backoff is recommended.
4. Compute regime-conditional covariance and verify it's different from global covariance in stress regime. Correlation limit should be tighter.
5. Build portfolio with factor exposures (beta=1.1, volatility=1.3). Check that beta passes but volatility fails in stress regime.
6. Simulate regime change (0→2→0) over 5 days. Verify constraint multipliers smoothly transition (not step change).
7. Run `replay_with_stress_constraints()` on 2007–2009 window. Verify output shows high constraint utilization in Sept 2008 (Lehman).
8. Run pytest on test suite. Verify coverage > 85%.

### Rollback plan

- **If universe.yaml is malformed:** Code throws `ConfigError` on startup with clear message; fall back to interactive prompt asking user to fix config or provide path.
- **If regime-conditional covariance has insufficient samples:** Fall back to global covariance with warning log.
- **If sizing backoff causes excessive position reduction:** Disable smoothing (set `CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA=1.0`) to revert to binary gating.
- **If constraint replay is slow:** Implement parallelization over scenarios and dates.
- **If tests fail:** Revert to previous version of `portfolio_risk.py`, `PositionSizer`, and rerun baseline tests.

---

## Notes

- **Sector cap tightening to 0.6x in stress is significant** — this assumes tail-risk contagion within sectors. May need empirical validation (e.g., measure sector-wise correlation increase during stress).
- **Correlation limit of 0.85 is already permissive** — academic evidence (Ledoit, Wolf 2004) suggests correlation > 0.6 provides minimal diversification. Consider stricter baseline.
- **SLSQP solver may not converge** if constraints become too tight. Consider adding fallback to cvxpy for convex relaxation if SLSQP fails.
- **Cross-regime consistency:** Portfolio optimizer uses global covariance (in objective) but risk manager uses regime-conditional. Consider also conditioning optimizer on regime (out of scope for this spec, but noted).
- **Liquidity feedback:** Factor exposures should eventually include realized slippage from actual execution; this spec computes factor loads but doesn't feed back execution cost to sizing.
