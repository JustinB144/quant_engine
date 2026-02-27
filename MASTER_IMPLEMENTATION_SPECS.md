# MASTER IMPLEMENTATION SPECS — Quant Engine v2.0.0
## Complete LLM Implementation Guide (Consolidated)

**Generated**: 2026-02-26
**Last Merged**: 2026-02-26 (combined MASTER_IMPLEMENTATION_SPECS.md + SPECS_STARTUP_AND_ROADMAP.md)
**Scope**: All verified improvements, audit findings, startup fixes, and integration wiring gaps
**Verification Method**: Every spec was verified against actual source code — no assumptions
**Total Specs**: 36 specs across 8 phases

---

## HOW TO USE THIS DOCUMENT

Each spec follows this format:
- **STATUS**: What exists now (verified against actual code)
- **PROBLEM**: Why this needs fixing
- **SPEC**: Exact changes to make
- **FILES**: Every file that needs modification with line references
- **VERIFICATION**: How to confirm the fix works
- **PRIORITY**: CRITICAL / HIGH / MEDIUM / LOW

Specs are ordered by implementation dependency — earlier specs must be completed before later ones that depend on them.

---

# PHASE 0: CRITICAL BUGS ✅ COMPLETE

All 8 critical bugs in this phase have been resolved. They are preserved here for reference and audit trail.

<details>
<summary>Click to expand Phase 0 specs (completed)</summary>

## SPEC-B01: Fix XGBoost callbacks API incompatibility [CRITICAL] ✅ COMPLETE

**STATUS**: `autopilot/meta_labeler.py` line 275 passes `callbacks` parameter to `XGBClassifier.fit()`. This parameter was removed in recent XGBoost versions. All 4 meta-labeling tests fail.

**PROBLEM**: Meta-labeling is completely broken. No signals get confidence filtering.

**SPEC**:
- In `autopilot/meta_labeler.py`, around line 270-280, find the `fit_params` dict construction
- Remove `callbacks` from `fit_params`
- Use `eval_set` and `early_stopping_rounds` as constructor params instead (XGBoost 2.0+ API)
- If early stopping is needed, pass `early_stopping_rounds` to the XGBClassifier constructor, not fit()

**FILES**: `autopilot/meta_labeler.py` (line ~275)

**VERIFICATION**: All 4 tests in `tests/test_signal_meta_labeling.py` must pass.

---

## SPEC-B02: Fix look-ahead bias in earnings data [CRITICAL] ✅ COMPLETE

**STATUS**: `data/alternative.py` `get_earnings_surprise()` (lines 115-116) uses `datetime.now()` for data extraction window. Need to verify `as_of_date` parameter is threaded through all call sites.

**PROBLEM**: During backtests, future earnings announcements can leak into historical predictions if `as_of_date` isn't passed.

**SPEC**:
- Verify that everywhere `get_earnings_surprise()`, `get_options_flow()`, `get_short_interest()`, `get_insider_transactions()`, `get_institutional_ownership()` are called during backtests, the current bar date is passed as `as_of_date`
- Search all call sites in `features/pipeline.py`, `features/research_factors.py`, `backtest/engine.py`
- Fix any call site missing `as_of_date`

**FILES**: `data/alternative.py`, `backtest/engine.py`, `features/pipeline.py`

**VERIFICATION**: `get_earnings_surprise("AAPL", as_of_date=datetime(2024,1,1))` should return NO data after 2024-01-01.

---

## SPEC-B03: Fix broken import in alternative.py [CRITICAL] ✅ COMPLETE

**STATUS**: `data/alternative.py` `_get_wrds()` imports `get_wrds_provider()` which does NOT exist in `data/wrds_provider.py`.

**PROBLEM**: All alternative data (earnings, fundamentals) silently fails to load.

**SPEC**:
- Check what the correct function name is in `data/wrds_provider.py` (likely `WRDSProvider` class or a factory function)
- Fix the import in `data/alternative.py` to match the actual export
- Add error handling that raises explicitly if WRDS provider is unavailable rather than silently returning empty data

**FILES**: `data/alternative.py`, `data/wrds_provider.py`

**VERIFICATION**: `from data.alternative import get_earnings_surprise` should not raise ImportError.

---

## SPEC-B04: Fix volume column case mismatch [CRITICAL] ✅ COMPLETE

**STATUS**: `data/cross_source_validator.py` uses lowercase column names but all OHLCV data uses titlecase ('Volume' not 'volume').

**PROBLEM**: Volume mismatches between data sources are NEVER detected. Corrupted volume data passes validation.

**SPEC**:
- In `cross_source_validator.py`, standardize all column references to titlecase: 'Open', 'High', 'Low', 'Close', 'Volume'
- Add a column normalization step at the top of validation that maps any casing to standard

**FILES**: `data/cross_source_validator.py`

**VERIFICATION**: Feed synthetic data with deliberately wrong volume data and confirm the validator catches it.

---

## SPEC-B05: Fix package installation structure [CRITICAL] ✅ COMPLETE

**STATUS**: `pyproject.toml` line 60-61 defines `packages.find.include = ["quant_engine*", "api*", "models*"]` but there is no `quant_engine/` subdirectory. The project root IS the package. `pip install -e .` installs but `import quant_engine` fails.

**PROBLEM**: Cannot deploy via standard Python packaging. Tests only work because pytest adds cwd to sys.path.

**SPEC**: Two options (choose one):
1. **Option A (recommended)**: Create a `src/quant_engine/` directory and move all source modules into it. Update all relative imports.
2. **Option B (minimal)**: Fix `pyproject.toml` to use the current flat layout correctly.

**FILES**: `pyproject.toml`, potentially all `__init__.py` files

**VERIFICATION**: `pip install -e .` followed by `python -c "from quant_engine.data.loader import load_ohlcv"` should work.

---

## SPEC-B06: Fix API broken imports [CRITICAL] ✅ COMPLETE

**STATUS**: Multiple API files use non-relative imports that only work with specific sys.path:
- `api/routers/iv_surface.py` line 17: imports from `models.iv.models` (wrong path)
- `api/routers/benchmark.py`, `api/services/backtest_service.py`, `api/services/model_service.py`, `api/routers/dashboard.py`: use `api.services.data_helpers` instead of relative imports

**PROBLEM**: API endpoints crash with ModuleNotFoundError depending on how the server is started.

**SPEC**:
- Convert all absolute imports in api/ to relative imports
- Fix `iv_surface.py` to use correct path for IV models
- Verify all api/ imports resolve correctly when run via `run_server.py`

**FILES**: `api/routers/iv_surface.py`, `api/routers/benchmark.py`, `api/services/backtest_service.py`, `api/services/model_service.py`, `api/routers/dashboard.py`

**VERIFICATION**: `python run_server.py` starts without import errors.

---

## SPEC-B07: Fix SPA fallback in run_server.py [HIGH] ✅ COMPLETE

**STATUS**: `run_server.py` SPA fallback returns None instead of raising 404 for non-API, non-static routes.

**PROBLEM**: Ambiguous behavior — returns 500 error instead of clean 404.

**SPEC**:
- If the path doesn't match a static file or API route, return `FileResponse("frontend/dist/index.html")` for SPA routes or raise `HTTPException(404)` for truly unknown paths

**FILES**: `run_server.py`

**VERIFICATION**: `GET /nonexistent-api-path` returns 404, `GET /dashboard` returns the SPA HTML.

---

## SPEC-B08: Fix MIN_REGIME_SAMPLES config value [HIGH] ✅ COMPLETE

**STATUS**: `config.py` sets `MIN_REGIME_SAMPLES = 50`. Test `test_regime_model_min_samples` asserts it should be >= 100.

**PROBLEM**: 50 samples is insufficient for regime model training.

**SPEC**:
- Increase `MIN_REGIME_SAMPLES` to 100 in `config.py`
- Update `config_structured.py` if it has a corresponding value

**FILES**: `config.py` (line ~214), `config_structured.py`

**VERIFICATION**: `test_regime_model_min_samples` passes.

</details>

---

# PHASE 1: CRITICAL WIRING + STARTUP FIXES

This is the highest-ROI work. These modules already exist and are tested individually, but their outputs aren't passed to consumers. Also includes two startup warning fixes.

## SPEC-W01: Wire structural state into execution simulator [CRITICAL] ✅ COMPLETE

**STATUS**: `backtest/execution.py` lines 255-258: `simulate()` accepts `break_probability`, `structure_uncertainty`, `drift_score`, `systemic_stress` parameters. These feed into a structural multiplier that scales spreads and impact (lines 399-416). But `backtest/engine.py` NEVER passes these when calling `simulate()` (lines 325-336, 399-410). Only passes `realized_vol`, `overnight_gap`, `intraday_range`, `urgency_type`, `volume_trend`.

**PROBLEM**: The execution cost model's most sophisticated feature — structural state conditioning — is completely unused. Costs are computed without regime/uncertainty awareness despite the code being ready to accept it.

**SPEC**:
In `backtest/engine.py`, before the main trade loop, pre-compute structural state for each bar:

```python
# At the top of run() or _run_backtest(), after regime detection:
from regime.shock_vector import compute_shock_vectors

# Pre-compute shock vectors for the entire backtest period
shock_vectors = compute_shock_vectors(
    prices=prices,
    regime_output=regime_output,
)
# shock_vectors is a dict: date → ShockVector
```

Then at both call sites for `simulate()` (lines 325-336 and 399-410), add the structural params:

```python
# Get shock vector for current bar date
shock = shock_vectors.get(bar_date)

fill = self.execution_model.simulate(
    side=side,
    reference_price=ref_price,
    daily_volume=vol,
    desired_notional_usd=desired_notional,
    force_full=force_full,
    realized_vol=context["realized_vol"],
    overnight_gap=context["overnight_gap"],
    intraday_range=context["intraday_range"],
    urgency_type=urgency,
    volume_trend=volume_trend,
    # NEW: structural state from shock vector
    break_probability=shock.bocpd_changepoint_prob if shock else None,
    structure_uncertainty=shock.hmm_uncertainty if shock else None,
    drift_score=getattr(shock, 'drift_score', None) if shock else None,
    systemic_stress=getattr(shock, 'systemic_stress', None) if shock else None,
)
```

Pre-compute shock vectors for the entire backtest period to avoid redundant regime detection per bar. Ensure `EXEC_STRUCTURAL_STRESS_ENABLED = True` in config.py.

**FILES**: `backtest/engine.py` (2 simulate() call sites + pre-computation block), `regime/shock_vector.py` (verify compute_shock_vectors API), `config.py` (verify EXEC_STRUCTURAL_STRESS_ENABLED)

**VERIFICATION**: Run two backtests: one with structural params (default) and one with all structural params = None. Compare TCA report costs — structural version should show higher costs during high-uncertainty periods.

---

## SPEC-W02: Wire cost calibrator into backtest execution [HIGH] ✅ COMPLETE

**STATUS**: `backtest/cost_calibrator.py` has a complete `CostCalibrator` class with per-segment (micro/small/mid/large) impact coefficients. NEVER imported in `backtest/engine.py`. `backtest/execution.py` line 75 uses flat `EXEC_IMPACT_COEFF_BPS = 25.0` for all securities. `simulate()` accepts `impact_coeff_override` parameter (line 263) — already designed for this.

**PROBLEM**: Micro-cap stocks use the same impact model as large-caps. Real impact is ~40 bps/sqrt(participation) for micro vs ~15 for large.

**SPEC**:
In `backtest/engine.py` `__init__()`:

```python
from backtest.cost_calibrator import CostCalibrator
self._cost_calibrator = CostCalibrator()
```

Before each trade execution, get the calibrated coefficient:

```python
# Determine market cap segment for the ticker (from universe data or price × shares)
segment = self._get_market_cap_segment(ticker)  # "micro", "small", "mid", "large"
impact_coeff = self._cost_calibrator.get_impact_coeff(segment)

fill = self.execution_model.simulate(
    ...,
    impact_coeff_override=impact_coeff,
)
```

Add `_get_market_cap_segment()` helper that maps ticker to cap segment using OHLCV volume × price or a precomputed lookup.

After each trade, feed actual cost data back: `calibrator.record_trade(...)` for online recalibration.

**FILES**: `backtest/engine.py`, `backtest/cost_calibrator.py` (verify `get_impact_coeff()` API)

**VERIFICATION**: Run backtest with mixed-cap universe. TCA report should show different impact costs for different market cap segments.

---

## SPEC-W03: Wire uncertainty gate into position sizing and portfolio weights [CRITICAL] ✅ COMPLETE

**STATUS**: `regime/uncertainty_gate.py` has a complete `UncertaintyGate` class with `compute_size_multiplier()` and `apply_uncertainty_gate()`. Exported from `regime/__init__.py` line 18. NEVER imported in `backtest/engine.py` or `autopilot/engine.py`. Only used in test files.

**PROBLEM**: Position sizes don't shrink when regime is uncertain. Full-size positions during regime transitions (where most losses occur).

**SPEC**:
**In `backtest/engine.py`**, after computing position size, apply uncertainty scaling:

```python
from regime.uncertainty_gate import UncertaintyGate

# In __init__:
self._uncertainty_gate = UncertaintyGate()

# In trade entry logic, after computing raw position size:
if shock is not None and shock.hmm_uncertainty is not None:
    size_mult = self._uncertainty_gate.compute_size_multiplier(shock.hmm_uncertainty)
    position_size *= size_mult
```

**In `autopilot/engine.py`**, after calling `optimize_portfolio()` (~line 1246):

```python
from regime.uncertainty_gate import UncertaintyGate

gate = UncertaintyGate()
current_uncertainty = regime_output.uncertainty.iloc[-1] if hasattr(regime_output, 'uncertainty') else 0.0
weights = gate.apply_uncertainty_gate(weights, current_uncertainty)
```

**FILES**: `backtest/engine.py` (position sizing section), `autopilot/engine.py` (after optimize_portfolio call, ~line 1248), `regime/uncertainty_gate.py`

**VERIFICATION**: During high-uncertainty bars (entropy > 0.8), position sizes should be reduced by 15-20%. Check trade log for size_multiplier < 1.0 during regime transitions.

---

## SPEC-W04: Wire regime covariance into portfolio optimizer [HIGH] ✅ COMPLETE

**STATUS**: `risk/covariance.py` line 230: `compute_regime_covariance()` exists — takes returns + regime labels, returns `Dict[int, pd.DataFrame]`. `risk/covariance.py` line 317: `get_regime_covariance()` also exists. But `autopilot/engine.py` (lines 1230-1248) uses generic `CovarianceEstimator().estimate(returns_df)` — NOT regime-aware. Line 1231 even has a comment: "regime-conditional covariance requires aligned regime labels" — acknowledging the gap.

**PROBLEM**: Portfolio optimization uses full-sample covariance. During stress regimes, correlations increase dramatically. The optimizer underestimates risk.

**SPEC**:
In `autopilot/engine.py`, replace lines 1230-1248:

```python
# Compute regime-conditional covariance
from risk.covariance import compute_regime_covariance, get_regime_covariance, CovarianceEstimator

current_regime = regime_output.regime.iloc[-1] if hasattr(regime_output, 'regime') else None

if current_regime is not None and regime_output.regime is not None:
    try:
        regime_covs = compute_regime_covariance(returns_df, regime_output.regime)
        cov_matrix = get_regime_covariance(regime_covs, current_regime)
    except (ValueError, KeyError):
        # Fall back to generic if insufficient data for regime-specific
        estimator = CovarianceEstimator()
        cov_matrix = estimator.estimate(returns_df).covariance
else:
    estimator = CovarianceEstimator()
    cov_matrix = estimator.estimate(returns_df).covariance

weights = optimize_portfolio(
    expected_returns=expected_returns[common],
    covariance=cov_matrix.loc[common, common],
)
```

**FILES**: `autopilot/engine.py` (lines 1230-1248), `risk/covariance.py` (verify API)

**VERIFICATION**: During a high-vol regime, portfolio should be less concentrated (more diversified) than with generic covariance.

---

## SPEC-W05: Wire reproducibility manifests into all entry points [HIGH] ✅ COMPLETE

**STATUS**: `reproducibility.py` has `build_run_manifest()`, `write_run_manifest()`, and `verify_manifest()` fully implemented (lines 117-235). None of `run_backtest.py`, `run_train.py`, `run_retrain.py`, `run_autopilot.py`, `run_predict.py` call any of these functions.

**PROBLEM**: No reproducibility tracking for any run. Cannot reconstruct past results.

**SPEC**:
In each entry point (`run_backtest.py`, `run_train.py`, `run_retrain.py`, `run_autopilot.py`, `run_predict.py`):

1. At the start of `main()`:
```python
from reproducibility import build_run_manifest, write_run_manifest
manifest = build_run_manifest(command="backtest", args=vars(args))
```

2. At the end of `main()` (or in a finally block):
```python
write_run_manifest(manifest, output_dir=results_dir)
```

3. Add `--verify-manifest` flag to each entry point:
```python
parser.add_argument("--verify-manifest", type=str, default=None,
    help="Path to a manifest file to verify environment matches before running")
```

**FILES**: `run_backtest.py`, `run_train.py`, `run_retrain.py`, `run_autopilot.py`, `run_predict.py`, `reproducibility.py` (verify API)

**VERIFICATION**: Run any entry point. A `run_manifest.json` file should be created in the output directory with git hash, config snapshot, and data checksums.

---

## SPEC-W06: Populate GICS_SECTORS and add --gics flag to WRDS refresh [HIGH] ✅ COMPLETE

**STATUS**:
- `config.py` line 405: `GICS_SECTORS: Dict[str, str] = {}` — empty dict placeholder
- `config.py` lines 673-681: `validate_config()` emits warning when `GICS_SECTORS` is empty
- `risk/portfolio_optimizer.py` lines 189-195: When `GICS_SECTORS` is empty, sector exposure constraint (`MAX_SECTOR_EXPOSURE = 0.10`) is completely skipped
- `run_wrds_daily_refresh.py` lines 171-186: argparse has NO --gics flag despite the warning message referencing it
- `config_data/universe.yaml` lines 4-113: Already contains a manual sector→tickers mapping (tech: AAPL, MSFT, GOOGL, etc.)

**PROBLEM**:
1. The startup warning tells the user to run `run_wrds_daily_refresh.py --gics` but that flag doesn't exist
2. Without GICS_SECTORS populated, the portfolio optimizer allows unlimited sector concentration — a major risk control gap
3. A manual mapping exists in `config_data/universe.yaml` but is never loaded into `GICS_SECTORS`

**SPEC**:

**Part A — Load GICS_SECTORS from universe.yaml on startup:**

In `config.py`, after the `GICS_SECTORS` definition (line 405), add loading logic:

```python
# Attempt to load GICS_SECTORS from universe.yaml
import yaml
_UNIVERSE_YAML = Path(__file__).parent / "config_data" / "universe.yaml"
if _UNIVERSE_YAML.exists() and not GICS_SECTORS:
    try:
        with open(_UNIVERSE_YAML) as f:
            _universe = yaml.safe_load(f)
        if _universe and "sectors" in _universe:
            for sector_name, tickers in _universe["sectors"].items():
                for ticker in tickers:
                    GICS_SECTORS[ticker] = sector_name
    except Exception:
        pass  # Will be caught by validate_config()
```

**Part B — Add --gics flag to run_wrds_daily_refresh.py:**

In `run_wrds_daily_refresh.py`, add to argparse (after line 186):

```python
parser.add_argument(
    "--gics", action="store_true",
    help="Refresh GICS sector mapping from WRDS Compustat and write to config_data/universe.yaml"
)
```

Add a new function that queries WRDS for GICS sector codes:

```python
def refresh_gics_sectors(provider, tickers: List[str]) -> Dict[str, str]:
    """Query Compustat for GICS sector mapping."""
    query = """
        SELECT DISTINCT a.tic, b.gsector, b.ggroup, b.gind, b.gsubind
        FROM comp.security a
        JOIN comp.company b ON a.gvkey = b.gvkey
        WHERE a.tic IN ({placeholders})
        AND b.gsector IS NOT NULL
    """.format(placeholders=",".join(f"'{t}'" for t in tickers))
    df = provider.raw_sql(query)
    GICS_SECTOR_NAMES = {
        "10": "Energy", "15": "Materials", "20": "Industrials",
        "25": "Consumer Discretionary", "30": "Consumer Staples",
        "35": "Health Care", "40": "Financials", "45": "Information Technology",
        "50": "Communication Services", "55": "Utilities", "60": "Real Estate",
    }
    return {row.tic: GICS_SECTOR_NAMES.get(str(row.gsector), "Unknown") for _, row in df.iterrows()}
```

**Part C — Fix the warning message:**

In `config.py` line 679, update:

```python
"GICS_SECTORS is empty — sector exposure constraints are disabled. "
"Populate via run_wrds_daily_refresh.py --gics, or ensure config_data/universe.yaml has a 'sectors' key."
```

**FILES**: `config.py` (line 405, lines 673-681), `run_wrds_daily_refresh.py` (lines 171-186, new function), `config_data/universe.yaml` (verify sectors key exists)

**VERIFICATION**:
- Server starts without the GICS_SECTORS warning
- `from config import GICS_SECTORS; len(GICS_SECTORS) > 0` returns True
- `risk/portfolio_optimizer.py` sector constraint code path is exercised

---

## SPEC-W07: Handle WRDS_ENABLED gracefully when credentials missing [MEDIUM]

**STATUS**:
- `config.py` line 33: `WRDS_ENABLED = True`
- `config.py` lines 693-703: Warning emitted when `WRDS_ENABLED=True` but `WRDS_USERNAME` env var not set
- `data/wrds_provider.py` lines 64-70: ALSO emits a Python warning at module import time (duplicate)
- `data/loader.py` lines 372-468: Has a 3-stage fallback (WRDS → cache → None) that marks fallback data as `trusted: False`
- `data/alternative.py` lines 172-186: Has `_check_wrds()` guard that returns None when WRDS unavailable

**PROBLEM**: Two separate warnings are emitted (one from config, one from wrds_provider.py). No guidance on degradation level. Alternative data features silently return None.

**SPEC**:

**Part A — Consolidate to a single, more informative warning:**

In `config.py` `validate_config()` (line 693), enhance:

```python
if WRDS_ENABLED:
    wrds_user = os.environ.get("WRDS_USERNAME", "")
    if not wrds_user:
        issues.append({
            "level": "WARNING",
            "message": (
                "WRDS_ENABLED=True but WRDS_USERNAME env var is not set. "
                "The system will operate in degraded mode: "
                "OHLCV data falls back to local cache (may have survivorship bias), "
                "and alternative data (earnings, options, short interest, insider) will be unavailable. "
                "Set via: export WRDS_USERNAME=<your_username>"
            ),
        })
```

**Part B — Suppress duplicate warning from wrds_provider.py:**

In `data/wrds_provider.py` lines 64-70, change `warnings.warn()` to `logger.debug()`.

**Part C — Add WRDS degradation status to health endpoint:**

In `api/services/health_service.py`:

```python
def _check_wrds_status(self) -> HealthCheckResult:
    from data.wrds_provider import WRDSProvider
    provider = WRDSProvider()
    if provider.available():
        return HealthCheckResult(name="wrds", status="ok", score=1.0)
    return HealthCheckResult(
        name="wrds", status="degraded", score=0.5,
        detail="WRDS unavailable — operating on cached/local data only"
    )
```

**FILES**: `config.py` (lines 693-703), `data/wrds_provider.py` (lines 64-70), `api/services/health_service.py`

**VERIFICATION**:
- Server starts with ONE clear warning (not two)
- Health endpoint shows WRDS status
- When `WRDS_USERNAME` is set, both warnings disappear

---

## SPEC-W08: Permanent cache for delisted stock data — skip WRDS re-queries for immutable history [HIGH]

**STATUS** (verified against actual code):

The current data loading pipeline treats delisted stock cache entries identically to active stocks. Specifically:

1. **`data/loader.py` `_cache_is_usable()` (lines 172-215)**: Checks `CACHE_MAX_STALENESS_DAYS = 21` (config.py line ~95) against the elapsed trading days since the cache's last bar date. This applies uniformly — even to stocks that delisted years ago and will never have new bars.

2. **`data/loader.py` `load_with_delistings()` (lines 690-832)**: At line 711-716, it calls `_cache_is_usable()` with `require_recent=False` — this is an existing partial mitigation. With `require_recent=False`, the staleness check is skipped, BUT it still requires `require_trusted=True`, meaning the cache entry must have `source` in `CACHE_TRUSTED_SOURCES` (line 196). If metadata is missing or corrupt, the entry fails even though the data is perfectly valid and immutable.

3. **`data/loader.py` `load_ohlcv()` (lines 323-469)**: The single-ticker path at line 351-357 calls `_cache_is_usable()` with `require_recent=True` AND `require_trusted=True`. A delisted stock cached 30+ trading days ago will fail the staleness check and trigger a redundant WRDS query for data that hasn't changed since the company ceased to exist.

4. **`run_wrds_daily_refresh.py` `main()` (lines 275-490)**: The bulk refresh downloads ALL tickers in `_build_ticker_list()` (line 300) — which includes every cached ticker plus `UNIVERSE_FULL` plus `BENCHMARK`. There is no logic to skip tickers whose cache already contains a `delist_event == 1` row. Every refresh cycle re-downloads immutable delisted stock data.

5. **`data/local_cache.py` `_write_cache_meta()` (lines 168-192)**: The metadata sidecar has no `"delisted"` or `"is_terminal"` field. There's no way for downstream code to know from metadata alone that a cached file contains complete, final data for a dead company.

6. **`data/local_cache.py` `save_ohlcv()` (lines 195-228)**: No detection of whether the DataFrame being saved contains a `delist_event == 1` row. The data is saved with the same metadata structure as an active stock.

7. **`data/survivorship.py` `DelistingHandler` (lines 555-726)**: Has `get_all_delisted_symbols()` (line 719) that returns all symbols recorded as delisted in the SQLite DB. Also has `is_delisted(symbol, as_of_date)` (line 712). This infrastructure exists but is never consulted by the cache layer.

8. **`data/wrds_provider.py` `get_crsp_prices_with_delistings()` (lines 524-598)**: Queries `crsp.dsedelist` for delisting events and integrates `dlret` into the `total_ret` column at lines 586-593. Sets `delist_event = 1` on the terminal bar. This data column IS preserved in the cache parquet files — meaning the cache already contains the information needed to detect delisted stocks.

**PROBLEM**:

Every time `load_ohlcv()` is called for a delisted stock beyond the 21-day staleness window, or every time `run_wrds_daily_refresh.py` runs, the system makes a redundant WRDS API call for data that is historically immutable. A stock that went bankrupt in 2008 will never un-go-bankrupt — its price history is final. This causes:

1. **Wasted WRDS API quota**: Each refresh cycle re-downloads ~50-100+ delisted tickers (depending on PIT universe coverage). At ~50 tickers per batch with 20 years of daily data, each batch takes 5-15 seconds. Delisted stocks alone can account for 30-60 seconds of WRDS query time per refresh.
2. **Unnecessary failure points**: If WRDS is down or credentials expire, delisted stock data that was already perfectly cached becomes "stale" and falls through to the untrusted fallback path, potentially triggering survivorship bias warnings for data that has nothing to do with survivorship.
3. **Incorrect staleness semantics**: A cache entry for Lehman Brothers (delisted 2008) being marked "stale" after 21 trading days is semantically wrong. The data is complete and final, not stale.

**SPEC**:

**Part A — Mark delisted cache entries as terminal in metadata sidecar:**

In `data/local_cache.py` `save_ohlcv()` (line 195), after validating the DataFrame, detect if it contains a terminal delisting event and record it in the metadata:

```python
def save_ohlcv(
    ticker: str,
    df: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    source: str = "unknown",
    meta: Optional[Dict[str, object]] = None,
) -> Path:
    d = Path(cache_dir) if cache_dir else _ensure_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    daily = _to_daily_ohlcv(df)
    if daily is None:
        raise ValueError(f"Invalid OHLCV data for {ticker}")

    # Detect terminal delisting event in the data
    extra_meta = dict(meta) if meta else {}
    if "delist_event" in df.columns:
        has_delist = int(df["delist_event"].max()) == 1
        if has_delist:
            delist_rows = df[df["delist_event"] == 1]
            delist_date = str(pd.to_datetime(delist_rows.index.max()).date())
            extra_meta["is_terminal"] = True
            extra_meta["terminal_date"] = delist_date
            extra_meta["terminal_reason"] = "delisted"
    # If the original df (pre _to_daily_ohlcv normalization) has delist info
    # but the daily version dropped it, also check the original
    elif meta and meta.get("is_terminal"):
        pass  # Caller already set it

    parquet_path = d / f"{ticker.upper()}_1d.parquet"
    try:
        daily.to_parquet(parquet_path)
        _write_cache_meta(parquet_path, ticker=ticker, df=daily, source=source, meta=extra_meta)
        return parquet_path
    except (ImportError, OSError) as e:
        logger.debug("Parquet write failed for %s, falling back to CSV: %s", ticker, e)
        csv_path = d / f"{ticker.upper()}_1d.csv"
        daily.to_csv(csv_path, index=True)
        _write_cache_meta(csv_path, ticker=ticker, df=daily, source=source, meta=extra_meta)
        return csv_path
```

Also update `_write_cache_meta()` to preserve `is_terminal` and `terminal_date` in the sidecar JSON (no change needed since line 185-186 already does `payload.update(meta)` — just verify this works).

**Part B — Bypass staleness check for terminal cache entries:**

In `data/loader.py` `_cache_is_usable()` (line 172), add a fast-path for terminal/delisted entries:

```python
def _cache_is_usable(
    cached: Optional[pd.DataFrame],
    meta: Optional[Dict[str, object]],
    years: int,
    require_recent: bool,
    require_trusted: bool,
) -> bool:
    if cached is None or len(cached) < MIN_BARS:
        return False

    if DATA_QUALITY_ENABLED:
        quality = assess_ohlcv_quality(cached)
        if not quality.passed:
            return False

    source = _cache_source(meta)
    if require_trusted and source not in TRUSTED_CACHE_SOURCES:
        return False

    idx = pd.to_datetime(cached.index)
    if len(idx) == 0:
        return False

    required_start = pd.Timestamp(date.today() - timedelta(days=int(years * 365.25)))
    if idx.min() > required_start + pd.Timedelta(days=90):
        return False

    # NEW: Terminal entries (delisted stocks) skip staleness check entirely.
    # A stock that delisted is historically immutable — its data is complete.
    if meta and meta.get("is_terminal") is True:
        return True

    # Also detect terminal state from the data itself if metadata is missing
    if "delist_event" in cached.columns:
        if int(cached["delist_event"].max()) == 1:
            return True

    if require_recent:
        last_trading_day = _get_last_trading_day()
        cache_end = pd.Timestamp(idx.max().date())
        trading_days_elapsed = _trading_days_between(cache_end, last_trading_day)
        if trading_days_elapsed > int(CACHE_MAX_STALENESS_DAYS):
            return False

    return True
```

**Part C — Skip delisted tickers in bulk refresh:**

In `run_wrds_daily_refresh.py`, modify `_build_ticker_list()` (line 33) to accept an option to exclude tickers with terminal cache entries. Then in `main()` (line 300), filter them out:

```python
def _build_ticker_list(tickers_arg, skip_terminal=True):
    """Build the full ticker list from cached + UNIVERSE_FULL + BENCHMARK.

    If skip_terminal=True, exclude tickers whose cache metadata has
    is_terminal=True (delisted stocks with complete, immutable data).
    """
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    cached = set(list_cached_tickers())
    universe = set(UNIVERSE_FULL)
    benchmark = {BENCHMARK}
    combined = sorted(cached | universe | benchmark)

    if not skip_terminal:
        return combined

    # Filter out tickers with terminal (delisted) cache entries
    from quant_engine.data.local_cache import load_ohlcv_with_meta
    active = []
    skipped_terminal = []
    for ticker in combined:
        _, meta, _ = load_ohlcv_with_meta(ticker)
        if meta and meta.get("is_terminal") is True:
            skipped_terminal.append(ticker)
            continue
        active.append(ticker)
    if skipped_terminal:
        print(f"  Skipping {len(skipped_terminal)} terminal/delisted tickers "
              f"(immutable cache): {', '.join(skipped_terminal[:10])}"
              f"{'...' if len(skipped_terminal) > 10 else ''}")
    return active
```

Add a `--include-terminal` CLI flag (line ~293) that overrides this behavior for cases where the user wants to force-refresh delisted data (e.g., if they suspect data corruption):

```python
parser.add_argument("--include-terminal", action="store_true",
    help="Include delisted/terminal tickers in refresh (normally skipped as immutable)")
```

In `main()`, pass it through:

```python
tickers = _build_ticker_list(args.tickers, skip_terminal=not args.include_terminal)
```

**Part D — Backfill is_terminal metadata for existing cache entries:**

Many delisted stocks are already cached with `source="wrds_delisting"` but lack the `is_terminal` metadata flag. Add a one-time migration function in `data/local_cache.py`:

```python
def backfill_terminal_metadata(cache_dir: Optional[Path] = None, dry_run: bool = False) -> Dict[str, int]:
    """Scan cache for delisted stocks and set is_terminal in their metadata.

    Reads each parquet file, checks for delist_event == 1, and updates
    the metadata sidecar if found. Idempotent — safe to run multiple times.

    Returns:
        Summary dict with counts of files scanned, updated, and skipped.
    """
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    summary = {"scanned": 0, "updated": 0, "already_terminal": 0, "active": 0, "errors": 0}

    for path in sorted(d.glob("*_1d.parquet")):
        summary["scanned"] += 1
        ticker = path.stem.replace("_1d", "").upper()

        meta = _read_cache_meta(path, ticker)
        if meta.get("is_terminal") is True:
            summary["already_terminal"] += 1
            continue

        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError, ImportError):
            summary["errors"] += 1
            continue

        has_delist = False
        if "delist_event" in df.columns:
            has_delist = int(df["delist_event"].max()) == 1

        if has_delist:
            delist_rows = df[df["delist_event"] == 1]
            terminal_date = str(pd.to_datetime(delist_rows.index.max()).date())
            if not dry_run:
                meta["is_terminal"] = True
                meta["terminal_date"] = terminal_date
                meta["terminal_reason"] = "delisted"
                _write_cache_meta(path, ticker=ticker, df=df, source=meta.get("source", "unknown"), meta=meta)
            summary["updated"] += 1
        else:
            summary["active"] += 1

    return summary
```

Add a CLI entry point to `run_wrds_daily_refresh.py`:

```python
parser.add_argument("--backfill-terminal", action="store_true",
    help="Scan cache and mark delisted stocks as terminal (one-time migration)")
```

In `main()`, before the download logic:

```python
if args.backfill_terminal:
    print(f"\n── Backfill Terminal Metadata ──")
    from quant_engine.data.local_cache import backfill_terminal_metadata
    result = backfill_terminal_metadata(DATA_CACHE_DIR)
    print(f"  Scanned: {result['scanned']}")
    print(f"  Newly marked terminal: {result['updated']}")
    print(f"  Already terminal: {result['already_terminal']}")
    print(f"  Active (no delist): {result['active']}")
    print(f"  Errors: {result['errors']}")
    return
```

**Part E — Cross-reference with DelistingHandler for cache entries missing delist_event column:**

Some older cache files from IBKR or pre-delisting WRDS downloads may not have a `delist_event` column at all. For these, cross-reference against the `DelistingHandler` SQLite DB:

In `backfill_terminal_metadata()`, add a fallback check:

```python
# If no delist_event column, check DelistingHandler DB
if not has_delist:
    try:
        from data.survivorship import DelistingHandler
        from config import SURVIVORSHIP_DB
        delist_handler = DelistingHandler(db_path=str(SURVIVORSHIP_DB))
        event = delist_handler.get_delisting_event(ticker)
        if event is not None:
            # Verify the cache data's last bar is on or before the delisting date
            last_bar_date = pd.to_datetime(df.index.max()).date()
            if last_bar_date <= event.delisting_date + timedelta(days=5):
                has_delist = True
                terminal_date = str(event.delisting_date)
    except (ImportError, OSError):
        pass  # DelistingHandler not available, skip this check
```

This ensures that even cache files without explicit `delist_event` markers can be identified as terminal if the survivorship DB knows about the delisting.

**FILES**:
- `data/local_cache.py` — `save_ohlcv()` (line 195-228): Add delist detection + terminal metadata. Add `backfill_terminal_metadata()` function.
- `data/loader.py` — `_cache_is_usable()` (lines 172-215): Add terminal fast-path before staleness check.
- `run_wrds_daily_refresh.py` — `_build_ticker_list()` (lines 33-41): Add terminal skip logic. `main()`: Add `--include-terminal` and `--backfill-terminal` flags.
- `data/survivorship.py` — No changes needed, only read (DelistingHandler is the cross-reference source).

**DEPENDENCIES**: None — this spec can be implemented independently of all other specs. It does not depend on W01-W07 or any other phase.

**VERIFICATION**:
1. **Metadata backfill**: Run `python run_wrds_daily_refresh.py --backfill-terminal`. Confirm it reports the correct count of delisted tickers marked terminal. Verify a known delisted ticker's `.meta.json` now contains `"is_terminal": true` and `"terminal_date": "YYYY-MM-DD"`.
2. **Staleness bypass**: Call `load_ohlcv("LEHMAN_PERMNO")` (or any known delisted permno in your cache) after 21+ trading days without refresh. Confirm it returns the cached data immediately without attempting a WRDS query. Add logging to verify: `logger.debug("Terminal cache hit for %s — skipping staleness check", ticker)`.
3. **Refresh skip**: Run `python run_wrds_daily_refresh.py --dry-run`. Confirm the ticker list is shorter than before — delisted tickers should be reported as "skipped (immutable)". Run with `--include-terminal` and confirm they reappear.
4. **Data integrity**: After backfill, run the existing test suite (`pytest tests/ -k "not integration"`) to confirm no regressions. Specifically verify `test_delisting_total_return.py` still passes.
5. **Edge case — new delisting**: Simulate a stock that was active last refresh but has since delisted. On next refresh, WRDS returns data with `delist_event = 1`. Confirm `save_ohlcv()` auto-sets `is_terminal: true` in metadata. On the FOLLOWING refresh, confirm this ticker is skipped.
6. **Edge case — force refresh**: Run `python run_wrds_daily_refresh.py --include-terminal --tickers DELISTED_TICKER`. Confirm the terminal ticker is re-downloaded and its cache is updated.

---

# PHASE 2: EXECUTION LAYER IMPROVEMENTS

## SPEC-E01: Add edge-after-costs trade gating [CRITICAL]

**STATUS**: The backtest engine has NO logic gating trades by whether predicted edge exceeds expected cost. Trades enter regardless of cost.

**PROBLEM**: System "wins" in backtests on trades that would never clear costs in live trading, especially during unstable states.

**SPEC**:
In `backtest/engine.py`, in the trade entry logic (around line 600-620), BEFORE entering a trade:

```python
# Compute expected cost for this trade
expected_cost_bps = self.execution_model.estimate_cost(
    daily_volume=volume,
    desired_notional=notional,
    realized_vol=realized_vol,
    structure_uncertainty=uncertainty,
)

# Predicted edge from model
predicted_edge_bps = abs(predicted_return) * 10000

# Buffer scales with regime uncertainty
cost_buffer_bps = EDGE_COST_BUFFER_BASE_BPS * (1 + structure_uncertainty)

# Gate: only trade if edge exceeds cost + buffer
if predicted_edge_bps <= expected_cost_bps + cost_buffer_bps:
    continue  # Skip this trade
```

Also add `estimate_cost()` method to `ExecutionModel` that returns expected cost without executing:
```python
def estimate_cost(self, daily_volume, desired_notional, realized_vol, structure_uncertainty=0):
    """Estimate expected round-trip cost in bps without executing."""
    # Use same spread + impact model as simulate() but return cost only
```

Config additions:
```python
EDGE_COST_GATE_ENABLED = True
EDGE_COST_BUFFER_BASE_BPS = 5.0  # Additional buffer beyond expected cost
```

**FILES**: `backtest/engine.py`, `backtest/execution.py`, `config.py`

**VERIFICATION**: Run backtest with and without edge gating. Gated version should have fewer trades but higher average net return per trade.

---

## SPEC-E02: Generalize regime suppression beyond "regime 2" [HIGH] ✅ COMPLETE

**STATUS**: `backtest/engine.py` line 603 hardcodes `regime == 2`:
```python
if not REGIME_2_TRADE_ENABLED and regime == 2 and confidence > REGIME_2_SUPPRESSION_MIN_CONFIDENCE:
    continue
```

**PROBLEM**: Only regime 2 can be suppressed. No way to suppress regime 3 (high vol) or apply different confidence thresholds per regime.

**SPEC**:
Replace hardcoded check with configurable per-regime policy:

```python
# config.py
REGIME_TRADE_POLICY = {
    0: {"enabled": True, "min_confidence": 0.0},    # trending_bull: always trade
    1: {"enabled": True, "min_confidence": 0.0},    # trending_bear: always trade
    2: {"enabled": False, "min_confidence": 0.70},   # mean_reverting: suppress unless high conf
    3: {"enabled": True, "min_confidence": 0.60},    # high_volatility: trade only with confidence
}
```

In `backtest/engine.py`, replace the hardcoded check:
```python
policy = REGIME_TRADE_POLICY.get(regime, {"enabled": True, "min_confidence": 0.0})
if not policy["enabled"] and confidence >= policy["min_confidence"]:
    continue
```

**FILES**: `config.py`, `backtest/engine.py`

**VERIFICATION**: Set regime 3 to `enabled: False` and run backtest. No trades should occur during regime 3.

---

## SPEC-E03: Add shock-mode execution policy [HIGH]

**STATUS**: `execution.py` has structural stress logic (lines 310-327) with a no-trade gate at `systemic_stress > 0.95`, and structural multipliers that increase costs. But this only works IF structural params are passed (see SPEC-W01). There's no generalized "shock mode" policy.

**PROBLEM**: Shock handling is spread across multiple ad-hoc checks. Need a unified shock policy that affects execution, sizing, and entry thresholds.

**SPEC**:
Create a `ShockModePolicy` class (can live in `backtest/execution.py` or new file):

```python
@dataclass
class ShockModePolicy:
    """Unified shock response for execution layer."""
    is_active: bool
    max_participation_override: float  # e.g., 0.005 vs normal 0.02
    spread_multiplier: float           # e.g., 2.0x during shock
    min_confidence_override: float     # e.g., 0.80 vs normal 0.50

    @classmethod
    def from_shock_vector(cls, shock: ShockVector) -> 'ShockModePolicy':
        if shock.is_shock_event():
            return cls(is_active=True, max_participation_override=0.005,
                       spread_multiplier=2.0, min_confidence_override=0.80)
        elif shock.hmm_uncertainty > 0.7:
            return cls(is_active=True, max_participation_override=0.01,
                       spread_multiplier=1.5, min_confidence_override=0.65)
        return cls(is_active=False, max_participation_override=0.02,
                   spread_multiplier=1.0, min_confidence_override=0.50)
```

Wire into engine trade loop:
- Apply max_participation_override to execution model
- Apply min_confidence_override to entry filter
- Log when shock mode activates

**FILES**: `backtest/execution.py`, `backtest/engine.py`, `config.py`

**VERIFICATION**: During a known shock period (e.g., March 2020), shock mode should activate, reducing participation and increasing entry threshold.

---

## SPEC-E04: Build calibration feedback loop for execution costs [MEDIUM] ✅ COMPLETE

**STATUS**: `cost_calibrator.py` can calibrate from historical trades. But there's no loop that compares simulated fills vs later actual fills and updates coefficients.

**PROBLEM**: Cost model parameters are static. Real costs drift over time.

**SPEC**:
1. Add `record_actual_fill()` to `CostCalibrator` that stores IBKR paper trade fill data
2. Add `compute_cost_surprise()` that compares predicted vs actual costs by regime bucket
3. Add monthly recalibration job that:
   - Loads recent paper trade fills
   - Computes cost surprise distribution per regime
   - Updates impact coefficients using EMA smoothing
   - Persists updated model

Config:
```python
EXEC_CALIBRATION_FEEDBACK_ENABLED = True
EXEC_CALIBRATION_FEEDBACK_INTERVAL_DAYS = 30
```

**FILES**: `backtest/cost_calibrator.py`, `autopilot/paper_trader.py`, `config.py`

**VERIFICATION**: After 100+ paper trades, calibrated coefficients should differ from initial defaults. Cost surprise distribution should center near zero.

---

# PHASE 3: PORTFOLIO LAYER IMPROVEMENTS

## SPEC-P01: Replace hardcoded regime_stats with learned statistics [HIGH]

**STATUS**: `risk/position_sizer.py` lines 108-113 initializes regime_stats with fixed win_rate/avg_win/avg_loss values and `n_trades: 0`. These are never updated from actual performance data.

**PROBLEM**: Position sizing uses made-up win rates. Real regime performance may differ dramatically.

**SPEC**:
1. Add `update_regime_stats()` method that accepts walk-forward backtest results:
```python
def update_regime_stats(self, backtest_trades: List[Trade]):
    """Compute regime-conditioned win rate/payoff from actual trades."""
    for regime_name, regime_code in REGIME_NAMES.items():
        regime_trades = [t for t in backtest_trades if t.regime == regime_code]
        if len(regime_trades) >= MIN_REGIME_TRADES_FOR_STATS:
            wins = [t for t in regime_trades if t.net_return > 0]
            losses = [t for t in regime_trades if t.net_return <= 0]
            self.regime_stats[regime_name] = {
                "win_rate": len(wins) / len(regime_trades),
                "avg_win": np.mean([t.net_return for t in wins]) if wins else 0,
                "avg_loss": np.mean([t.net_return for t in losses]) if losses else 0,
                "n_trades": len(regime_trades),
            }
```

2. Call this from `autopilot/engine.py` after each backtest cycle
3. Persist learned stats to disk (alongside Bayesian counters)
4. Only use learned stats when `n_trades >= MIN_REGIME_TRADES_FOR_STATS` (default 30); otherwise use Bayesian prior

Config:
```python
MIN_REGIME_TRADES_FOR_STATS = 30
REGIME_STATS_PERSIST_PATH = "trained_models/regime_trade_stats.json"
```

**FILES**: `risk/position_sizer.py`, `autopilot/engine.py`, `config.py`

**VERIFICATION**: After a backtest with 200+ trades, `regime_stats` should contain non-zero `n_trades` and win rates that differ from the hardcoded defaults.

---

## SPEC-P02: Wire uncertainty into Bayesian Kelly sizing [HIGH]

**STATUS**: `risk/position_sizer.py` has `kelly_bayesian()` (lines 843-880) with per-regime posteriors. `regime/uncertainty_gate.py` has `compute_size_multiplier()`. They are never connected.

**PROBLEM**: Kelly sizes don't shrink during uncertain regime transitions.

**SPEC**:
1. Add `uncertainty` parameter to `compute_size()` in position_sizer.py:
```python
def compute_size(self, ..., regime_uncertainty: float = 0.0) -> float:
    # ... existing Kelly computation ...
    raw_size = kelly_result

    # Apply uncertainty scaling
    if regime_uncertainty > 0:
        from regime.uncertainty_gate import UncertaintyGate
        gate = UncertaintyGate()
        uncertainty_mult = gate.compute_size_multiplier(regime_uncertainty)
        raw_size *= uncertainty_mult

    return raw_size
```

2. Thread regime uncertainty from detector through backtest engine to position sizer

**FILES**: `risk/position_sizer.py`, `backtest/engine.py`

**VERIFICATION**: During high-uncertainty bars, Kelly position sizes should be 80-85% of normal (per REGIME_UNCERTAINTY_SIZING_MAP defaults).

---

## SPEC-P03: Dynamic correlation-based constraint tightening [MEDIUM]

**STATUS**: `risk/portfolio_risk.py` tightens constraints by regime (stress multipliers for regimes 2 & 3). But it does NOT tighten based on observed correlation rising.

**PROBLEM**: Regime classification can lag. Correlations may spike before the regime detector catches up.

**SPEC**:
In `risk/portfolio_risk.py` `_check_correlations()`:

```python
# Compute current average pairwise correlation
avg_corr = np.mean(np.abs(recent_corr_matrix[np.triu_indices_from(recent_corr_matrix, k=1)]))

# Dynamic tightening based on actual correlation level
corr_stress_multiplier = 1.0
if avg_corr > 0.6:
    corr_stress_multiplier = 0.85  # 15% tighter
if avg_corr > 0.7:
    corr_stress_multiplier = 0.70  # 30% tighter
if avg_corr > 0.8:
    corr_stress_multiplier = 0.50  # 50% tighter

# Apply on top of regime multiplier
effective_limit *= corr_stress_multiplier
```

Config:
```python
CORRELATION_STRESS_THRESHOLDS = {0.6: 0.85, 0.7: 0.70, 0.8: 0.50}
```

**FILES**: `risk/portfolio_risk.py`, `config.py`

**VERIFICATION**: When pairwise correlation > 0.7, single-name and sector caps should be noticeably tighter.

---

## SPEC-P04: Make turnover penalty configurable and cost-aware [MEDIUM] ✅ COMPLETE

**STATUS**: `risk/portfolio_optimizer.py` line 33: `turnover_penalty: float = 0.001` hardcoded default. `autopilot/engine.py` line 1246 calls `optimize_portfolio()` without passing `turnover_penalty`. No config variable exists.

**PROBLEM**: Turnover penalty is locked at 0.1% regardless of actual trading costs. During high-cost regimes, the optimizer may suggest trades that cost more than the penalty.

**SPEC**:

**Part A — Add config variable:**

In `config.py`:
```python
PORTFOLIO_TURNOVER_PENALTY = 0.001  # Penalty per unit turnover in optimization (decimal)
```

**Part B — Pass from autopilot:**

In `autopilot/engine.py` at the optimize_portfolio call:
```python
from config import PORTFOLIO_TURNOVER_PENALTY

weights = optimize_portfolio(
    expected_returns=expected_returns[common],
    covariance=cov_matrix.loc[common, common],
    turnover_penalty=PORTFOLIO_TURNOVER_PENALTY,
    current_weights=current_weights,
)
```

**Part C (optional) — Dynamic penalty from execution costs:**

```python
# Estimate current cost environment
avg_cost_bps = execution_model.estimate_cost(avg_volume, avg_notional, current_vol)
dynamic_penalty = max(PORTFOLIO_TURNOVER_PENALTY, avg_cost_bps / 10000 * 2)  # 2× expected cost
```

**FILES**: `config.py`, `autopilot/engine.py` (~line 1246), `risk/portfolio_optimizer.py` (no change needed — already accepts the param)

**VERIFICATION**: Changing `PORTFOLIO_TURNOVER_PENALTY` from 0.001 to 0.01 should reduce portfolio turnover.

---

# PHASE 4: EVALUATION LAYER IMPROVEMENTS

## SPEC-V01: Add uncertainty quantile slicing [HIGH]

**STATUS**: `evaluation/slicing.py` has regime-based slicing (5 primary + 4 individual regime slices). Missing: uncertainty quantile slicing, regime transition window slicing.

**PROBLEM**: Cannot evaluate strategy performance specifically during high-uncertainty periods or regime transitions — exactly when most losses occur.

**SPEC**:
In `evaluation/slicing.py`, add two new slice creators:

```python
def create_uncertainty_slices(self, uncertainty: pd.Series, n_quantiles: int = 3):
    """Slice by regime uncertainty quantiles."""
    quantile_edges = np.linspace(0, 1, n_quantiles + 1)
    thresholds = uncertainty.quantile(quantile_edges)

    for i in range(n_quantiles):
        low, high = thresholds.iloc[i], thresholds.iloc[i+1]
        label = f"uncertainty_q{i+1}"  # q1=low, q2=mid, q3=high uncertainty
        mask = (uncertainty >= low) & (uncertainty < high)
        self.add_slice(label, mask)

def create_transition_slices(self, regimes: pd.Series, window: int = 5):
    """Slice by proximity to regime transitions."""
    changes = regimes.diff().abs() > 0
    near_transition = changes.rolling(window, center=True).max().fillna(0) > 0
    self.add_slice("near_transition", near_transition)
    self.add_slice("stable_regime", ~near_transition)
```

Wire into `evaluation/engine.py`:
- Make uncertainty slices MANDATORY when regime_states provided
- Make transition slices MANDATORY

**FILES**: `evaluation/slicing.py`, `evaluation/engine.py`

**VERIFICATION**: EvaluationResult should contain `uncertainty_q1`, `uncertainty_q2`, `uncertainty_q3`, `near_transition`, `stable_regime` slice metrics.

---

## SPEC-V02: Add worst-case bucket promotion gates [HIGH] ✅ COMPLETE

**STATUS**: `autopilot/promotion_gate.py` checks `regime_positive_fraction` and `worst_event_loss` but does NOT check max drawdown within specific stress buckets.

**PROBLEM**: A strategy can pass overall Sharpe but have catastrophic losses during shock periods.

**SPEC**:
Add to `promotion_gate.py`:

```python
# In evaluate() method, after basic gates:
if self.require_advanced_contract:
    regime_metrics = backtest_result.regime_performance
    for regime_code, metrics in regime_metrics.items():
        if regime_code in [2, 3]:  # stress regimes
            if metrics.get('max_drawdown', 0) > PROMOTION_MAX_STRESS_DRAWDOWN:
                gates.append(Gate(
                    name=f"stress_regime_{regime_code}_drawdown",
                    passed=False,
                    reason=f"Regime {regime_code} drawdown {metrics['max_drawdown']:.1%} exceeds {PROMOTION_MAX_STRESS_DRAWDOWN:.1%}"
                ))
            if metrics.get('sharpe_ratio', 0) < PROMOTION_MIN_STRESS_SHARPE:
                gates.append(Gate(
                    name=f"stress_regime_{regime_code}_sharpe",
                    passed=False,
                    reason=f"Regime {regime_code} Sharpe {metrics['sharpe_ratio']:.2f} below {PROMOTION_MIN_STRESS_SHARPE:.2f}"
                ))
```

Config:
```python
PROMOTION_MAX_STRESS_DRAWDOWN = 0.15     # Max 15% drawdown in stress regimes
PROMOTION_MIN_STRESS_SHARPE = -0.50      # At least not deeply negative in stress
PROMOTION_MAX_TRANSITION_DRAWDOWN = 0.10 # Max 10% drawdown near transitions
```

**FILES**: `autopilot/promotion_gate.py`, `config.py`

**VERIFICATION**: A strategy with overall Sharpe 1.5 but -25% drawdown in regime 3 should fail promotion.

---

## SPEC-V03: Make capacity analysis regime-dependent [MEDIUM]

**STATUS**: `backtest/advanced_validation.py` `capacity_analysis()` uses uniform volume and impact assumptions regardless of regime.

**PROBLEM**: Liquidity collapses in stress. A strategy with $50M capacity in calm might only handle $10M in stress.

**SPEC**:
Add regime parameter to `capacity_analysis()`:

```python
def capacity_analysis(returns_series, trades, regime_labels=None, stress_regimes=[2, 3]):
    overall_capacity = _compute_capacity(trades)

    if regime_labels is not None:
        stress_trades = [t for t in trades if t.regime in stress_regimes]
        if len(stress_trades) >= 10:
            stress_capacity = _compute_capacity(stress_trades)
            result.stress_capacity = stress_capacity
            result.capacity_constrained = stress_capacity < MIN_CAPACITY_USD
```

Gate on stress capacity in promotion:
```python
if capacity_result.stress_capacity < PROMOTION_MIN_STRESS_CAPACITY:
    gates.append(Gate(name="stress_capacity", passed=False, ...))
```

**FILES**: `backtest/advanced_validation.py`, `autopilot/promotion_gate.py`, `config.py`

**VERIFICATION**: A strategy trading illiquid stocks should show lower capacity during stress regime bars.

---

# PHASE 5: DATA & FEATURE INTEGRITY

## SPEC-D01: Add OHLC relationship validation [HIGH]

**STATUS**: `data/quality.py` daily quality checks do NOT verify High >= max(Open, Close), Low <= min(Open, Close), High >= Low.

**PROBLEM**: Corrupted OHLC data can propagate into features and signals without detection.

**SPEC**:
Add to quality check pipeline:
```python
def check_ohlc_relationships(df: pd.DataFrame) -> List[str]:
    violations = []
    bad_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
    bad_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
    bad_hl = (df['High'] < df['Low']).sum()
    if bad_high > 0: violations.append(f"{bad_high} bars where High < max(Open, Close)")
    if bad_low > 0: violations.append(f"{bad_low} bars where Low > min(Open, Close)")
    if bad_hl > 0: violations.append(f"{bad_hl} bars where High < Low")
    return violations
```

**FILES**: `data/quality.py`

**VERIFICATION**: Feed synthetic data with High < Low and confirm it's flagged.

---

## SPEC-D02: Add feature causality enforcement at runtime [HIGH]

**STATUS**: `features/pipeline.py` has `FEATURE_METADATA` marking features as CAUSAL (lines 87-200) and `TRUTH_LAYER_ENFORCE_CAUSALITY = True` in config. But there's NO runtime check preventing RESEARCH_ONLY features from reaching live predictions.

**PROBLEM**: Research features (which may use future data) can leak into live predictions.

**SPEC**:
In `models/predictor.py` predict method:
```python
if TRUTH_LAYER_ENFORCE_CAUSALITY:
    from features.pipeline import FEATURE_METADATA
    allowed = {f for f, meta in FEATURE_METADATA.items() if meta.get('type') == 'CAUSAL'}
    used = set(X.columns)
    research_only = used - allowed
    if research_only:
        raise ValueError(f"RESEARCH_ONLY features in prediction: {research_only}")
```

**FILES**: `models/predictor.py`, `features/pipeline.py`

**VERIFICATION**: Create a feature set with a RESEARCH_ONLY feature and try to predict. Should raise ValueError.

---

## SPEC-D03: Fix data/local_cache.py race condition [MEDIUM]

**STATUS**: No file locking on parquet writes. Concurrent processes can corrupt cache.

**SPEC**:
Use atomic write pattern:
```python
import tempfile, os

def write_cache(path: str, df: pd.DataFrame):
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=dir_name, suffix='.tmp', delete=False) as f:
        tmp_path = f.name
        df.to_parquet(f.name)
    os.replace(tmp_path, path)  # Atomic on POSIX
```

**FILES**: `data/local_cache.py`, `data/feature_store.py`

**VERIFICATION**: Run two concurrent cache writes to the same key. Neither should produce a corrupted file.

---

# PHASE 6: CONFIG & ARCHITECTURE

## SPEC-A01: Consolidate flat and structured configs [HIGH]

**STATUS**: `config.py` (flat, 300+ constants) and `config_structured.py` (dataclasses) have overlapping but inconsistent values. Example: `PROMOTION_MAX_PBO=0.45` in config.py but may differ in structured config.

**PROBLEM**: No single source of truth. Changes in one config aren't reflected in the other.

**SPEC**:
1. Make `config_structured.py` the authoritative source
2. Have `config.py` import from structured config for backward compatibility:
```python
# config.py
from config_structured import get_config
_cfg = get_config()
PROMOTION_MAX_PBO = _cfg.promotion.max_pbo
MIN_REGIME_SAMPLES = _cfg.regime.min_samples
# etc.
```
3. Validate that structured config values match flat config values (add a test)

**FILES**: `config.py`, `config_structured.py`, `tests/test_config_consistency.py` (new)

**VERIFICATION**: Change a value in structured config and confirm flat config reflects it.

---

## SPEC-A02: Add config validation completeness [HIGH]

**STATUS**: `config.py` `validate_config()` does not check: ensemble weights sum to 1.0, negative cost multipliers, regime threshold ranges, label_h > 0.

**SPEC**:
Add to `validate_config()`:
```python
# Ensemble weights must sum to 1.0
weights = REGIME_ENSEMBLE_DEFAULT_WEIGHTS
assert abs(sum(weights.values()) - 1.0) < 1e-6, f"Ensemble weights sum to {sum(weights.values())}"

# No negative cost multipliers
assert TRANSACTION_COST_BPS >= 0, "Negative transaction cost"
assert EXEC_SPREAD_BPS >= 0, "Negative spread"

# Regime threshold ranges
assert 0 < REGIME_UNCERTAINTY_ENTROPY_THRESHOLD < 2.0, "Entropy threshold out of range"

# Label horizon positive
assert LABEL_HORIZONS and all(h > 0 for h in LABEL_HORIZONS), "Invalid label horizons"
```

**FILES**: `config.py`

**VERIFICATION**: Set ensemble weights to [0.5, 0.3, 0.3] (sum 1.1) and confirm validate_config() raises.

---

# PHASE 7: HEALTH & MONITORING

## SPEC-H01: Add IC tracking to health system [MEDIUM]

**STATUS**: Health service monitors data quality, model freshness, and general system health. It does NOT track rolling IC (information coefficient) over time.

**SPEC**:
Add a health check that:
1. Retrieves last N autopilot cycle reports
2. Extracts IC values from each
3. Computes rolling IC mean and trend
4. Flags WARNING if IC < 0.01, CRITICAL if IC < 0

**FILES**: `api/services/health_service.py`

---

## SPEC-H02: Add ensemble disagreement monitoring [MEDIUM]

**STATUS**: The ensemble predictor combines XGBoost, LightGBM, and regime-specific models. No monitoring of when members disagree.

**SPEC**:
After ensemble prediction, compute:
```python
disagreement = np.std([model_1_pred, model_2_pred, model_3_pred])
```
Track this in health service. High disagreement = high uncertainty = signal unreliable.

**FILES**: `models/predictor.py`, `api/services/health_service.py`

---

## SPEC-H03: Add execution quality monitoring [MEDIUM]

**STATUS**: TCA report exists per backtest but no live monitoring of paper trade fill quality.

**SPEC**:
Compare paper trade fills to model predictions:
```python
cost_surprise = actual_cost_bps - predicted_cost_bps
```
Track rolling cost surprise distribution. Alert if systematically positive (model underestimates costs).

**FILES**: `autopilot/paper_trader.py`, `api/services/health_service.py`

---

# APPENDIX A: PREVIOUSLY RECOMMENDED BUT ALREADY IMPLEMENTED

These were recommended in COMPREHENSIVE_IMPROVEMENT_INSTRUCTIONS.md but our verification shows they ALREADY EXIST. Do NOT re-implement:

| Recommendation | Already Implemented | Location |
|---|---|---|
| Add Spearman IC to autopilot | YES — inline in run_statistical_tests() | backtest/validation.py:461-486, called from autopilot/engine.py:828 |
| Add CPCV to autopilot | YES — combinatorial_purged_cv() | backtest/validation.py:547-687, called from autopilot/engine.py:842-849 |
| Add SPA bootstrap to autopilot | YES — superior_predictive_ability() | backtest/validation.py:716-781, called from autopilot/engine.py:853-865 |
| Add DSR to promotion | YES — deflated_sharpe_ratio() | advanced_validation.py:94-174, called from autopilot/engine.py:800-809 |
| Add PBO to promotion | YES — probability_of_backtest_overfitting() | advanced_validation.py:176-288, checked in promotion_gate.py |
| Add Monte Carlo validation | YES — monte_carlo_validation() | advanced_validation.py:291-364, checked in promotion_gate.py |
| Add capacity analysis | YES — capacity_analysis() | advanced_validation.py:367-457, checked in promotion_gate.py |
| Wire calibration into ensemble | YES — ConfidenceCalibrator | models/calibration.py, fitted in trainer.py:1450-1546 |
| Add Bayesian Kelly | YES — kelly_bayesian() | risk/position_sizer.py:843-880 with per-regime posteriors |
| Implement regime ensemble | YES — detect_ensemble() | regime/detector.py:390-509 (HMM + Jump + Rule) |
| Add regime uncertainty/entropy | YES — get_regime_uncertainty() | regime/detector.py:622-636, output in RegimeOutput.uncertainty |
| Add uncertainty gate | YES — UncertaintyGate class | regime/uncertainty_gate.py (exists but NOT WIRED — see SPEC-W03) |
| Add ShockVector | YES — full ShockVector dataclass + detection | regime/shock_vector.py |
| Add cost calibrator | YES — CostCalibrator class | backtest/cost_calibrator.py (exists but NOT WIRED — see SPEC-W02) |
| Add regime covariance | YES — compute_regime_covariance() | risk/covariance.py:230-314 (exists but NOT WIRED — see SPEC-W04) |
| Add drawdown-adjusted sizing | YES — multi-tier circuit breaker | risk/drawdown.py (quadratic recovery ramp) |
| Add factor exposure tracking | YES — FactorExposureMonitor | risk/factor_exposures.py + factor_monitor.py |
| Add portfolio-level risk checks | YES — PortfolioRiskManager | risk/portfolio_risk.py (correlation + vol + dynamic stress) |
| Config dataclasses | YES — config_structured.py | config_structured.py (exists, needs consolidation — see SPEC-A01) |
| Feature versioning | YES — features/version.py | features/version.py (exists) |

---

# APPENDIX B: IMPROVEMENT ROADMAP — ALL 30 ITEMS VERIFIED AS COMPLETE

The following items from `docs/plans/IMPROVEMENT_ROADMAP.md` are ALL fully implemented. Do NOT re-implement:

| Item | Evidence |
|------|----------|
| 1.1 Fresh training cycle | Models dated 2026-02-26 in trained_models/ |
| 1.2 Regime 2 gating | config.py:537 REGIME_2_TRADE_ENABLED=False + predictor.py:362-369 |
| 2.1 IBES earnings | alternative.py:196-294 get_earnings_surprise() |
| 2.2 OptionMetrics | alternative.py:298-371 get_options_flow() |
| 2.3 Short interest | alternative.py:375-460 get_short_interest() |
| 2.4 Insider txns | alternative.py:464-583 get_insider_transactions() |
| 2.5 13F ownership | alternative.py:587-658 get_institutional_ownership() |
| 2.6 TAQmsec intraday | features/intraday.py:24-195 all 6 metrics |
| 3.1 DTW lead-lag | research_factors.py:703-860 compute_dtw_lead_lag() |
| 3.2 Path signatures | research_factors.py:370+ compute_signature_path_features() |
| 3.3 OFI calibration | research_factors.py:66-131 true OFI, not proxy |
| 3.4 FRED macro | features/macro.py:50-244 MacroFeatureProvider, 5 series |
| 4.1 XGBoost | models/trainer.py:1081-1102 in ensemble |
| 4.2 Neural net | models/neural_net.py:29-199 TabularNet (PyTorch) |
| 4.3 Predictor regime 2 | models/predictor.py:362-369 confidence=0 when regime==2 |
| 4.4 Walk-forward select | models/walk_forward.py:93-235 multi-config + DSR |
| 4.5 Cross-sectional rank | cross_sectional.py:18-136, called in autopilot/engine.py:401,541 |
| 4.6 Portfolio optimizer | portfolio_optimizer.py:27-100, called from autopilot |
| 5.1 UI dashboard | React + TypeScript, 10 pages, 50+ components |
| 5.2 Integration tests | 73 test files, full pipeline coverage |
| 5.3 Logging | utils/logging.py:22-43 StructuredFormatter with JSON |
| 5.4 Reproducibility | reproducibility.py:117-235 verify_manifest() implemented |
| 6.1 HARX spillovers | features/harx_spillovers.py:54+ full implementation |
| 6.2 Markov LOB | features/lob_features.py:88+ 7 microstructure proxies |
| 6.3 Vol-scaled momentum | research_factors.py normalized by 60d vol |
| 6.4 Wave-flow decomp | features/wave_flow.py:26-144 FFT spectral analysis |

All SPEC 1-10 from docs/specs/ are also verified as implemented in the codebase.

---

# APPENDIX C: EVALUATION VERDICTS FOR USER'S NEW PROPOSALS

## 1A. Make execution costs regime & uncertainty conditioned
**VERDICT: BENEFICIAL — but it's a WIRING task, not a new build**
The execution model ALREADY has structural state conditioning (SPEC-W01). Just needs to be called with the right params.

## 1B. Shock override that changes order aggressiveness
**VERDICT: BENEFICIAL — partially exists, needs generalization (SPEC-E03)**
Structural stress multipliers exist. Need unified ShockModePolicy.

## 1C. Calibration loop
**VERDICT: BENEFICIAL — calibrator exists, needs feedback loop (SPEC-E04)**
CostCalibrator is real and production-ready. Missing: actual fill comparison loop.

## 2A. Replace hard regime priors in PositionSizer
**VERDICT: BENEFICIAL (SPEC-P01)**
Hardcoded defaults with n_trades=0 are indeed placeholders. Need update_regime_stats().

## 2B. Regime-uncertainty-aware scaling
**VERDICT: BENEFICIAL (SPEC-P02 + SPEC-W03)**
UncertaintyGate exists but isn't wired. High-ROI connection.

## 2C. Dynamic correlation-based constraints
**VERDICT: BENEFICIAL (SPEC-P03)**
Current tightening is regime-based only. Adding observed-correlation trigger adds a faster response.

## 2D. Optimizer: regime covariance + turnover realism
**VERDICT: BENEFICIAL (SPEC-W04 + SPEC-P04)**
compute_regime_covariance() exists, isn't called. Turnover penalty is hardcoded 0.001.

## 3A. Mandatory slicing reports
**VERDICT: BENEFICIAL — partially exists, needs uncertainty + transition slices (SPEC-V01)**
5 regime slices already mandatory. Missing: uncertainty quantile and transition window slices.

## 3B. Worst-case bucket promotion gates
**VERDICT: BENEFICIAL (SPEC-V02)**
regime_positive_fraction exists but no max-drawdown-per-regime gate.

## 3C. Regime-dependent capacity analysis
**VERDICT: BENEFICIAL (SPEC-V03)**
Current capacity analysis is regime-agnostic. Stress-regime capacity is the binding constraint.

## The "single most important wiring improvement"
**VERDICT: STRONGLY AGREE — this is Phase 1 in our specs**
The user correctly identified that regime state (probabilities + uncertainty + shock flags) should be a required input everywhere. Our SPEC-W01 through SPEC-W05 address exactly this.

---

# IMPLEMENTATION ORDER

**Phase 0 — Critical Bugs** ✅ COMPLETE (8 specs done)

**Phase 1 — Critical Wiring + Startup Fixes (~25 hours)**:
SPEC-W01 through SPEC-W08 (8 specs — 6 done, 2 remaining)
- W01: Wire structural state → execution [CRITICAL] ✅ DONE (3h)
- W02: Wire cost calibrator → backtest [HIGH] ✅ DONE (2h)
- W03: Wire uncertainty gate → sizing [CRITICAL] ✅ DONE (2h)
- W04: Wire regime covariance → optimizer [HIGH] ✅ DONE (2h)
- W05: Wire reproducibility manifests → entry points [HIGH] ✅ DONE (2h)
- W06: Populate GICS_SECTORS + --gics flag [HIGH] ✅ DONE (2h)
- W07: Handle WRDS warning gracefully [MEDIUM] **← NEXT** (1h)
- W08: Permanent cache for delisted stock data [HIGH] (3h)

**Phase 2 — Execution Layer (~12 hours)**:
SPEC-E01 through SPEC-E04 (4 specs)
- E01: Edge-after-costs trade gating [CRITICAL] (4h)
- E02: Generalize regime suppression [HIGH] (2h) ✅ COMPLETE
- E03: Shock-mode execution policy [HIGH] (3h)
- E04: Calibration feedback loop [MEDIUM] (3h)

**Phase 3 — Portfolio Layer (~12 hours)**:
SPEC-P01 through SPEC-P04 (4 specs)
- P01: Replace hardcoded regime_stats [HIGH] (4h)
- P02: Wire uncertainty into Kelly sizing [HIGH] (2h)
- P03: Dynamic correlation constraints [MEDIUM] (3h)
- P04: Configurable turnover penalty [MEDIUM] (3h)

**Phase 4 — Evaluation Layer (~10 hours)**:
SPEC-V01 through SPEC-V03 (3 specs)
- V01: Uncertainty quantile slicing [HIGH] (4h)
- V02: Worst-case bucket gates [HIGH] (3h)
- V03: Regime-dependent capacity [MEDIUM] (3h)

**Phase 5 — Data & Feature Integrity (~8 hours)**:
SPEC-D01 through SPEC-D03 (3 specs)
- D01: OHLC relationship validation [HIGH] (2h)
- D02: Feature causality enforcement [HIGH] (3h)
- D03: Cache race condition fix [MEDIUM] (3h)

**Phase 6 — Config & Architecture (~8 hours)**:
SPEC-A01 through SPEC-A02 (2 specs)
- A01: Consolidate flat + structured configs [HIGH] (5h)
- A02: Config validation completeness [HIGH] (3h)

**Phase 7 — Health & Monitoring (~8 hours)**:
SPEC-H01 through SPEC-H03 (3 specs)
- H01: IC tracking [MEDIUM] (3h)
- H02: Ensemble disagreement monitoring [MEDIUM] (2h)
- H03: Execution quality monitoring [MEDIUM] (3h)

---

## SUMMARY

| Phase | Specs | Hours | Status |
|-------|-------|-------|--------|
| 0 — Critical Bugs | 8 | ~16h | ✅ COMPLETE |
| 1 — Critical Wiring + Startup | 8 (6 done) | ~25h | W07-W08 remaining |
| 2 — Execution Layer | 4 | ~12h | Pending |
| 3 — Portfolio Layer | 4 | ~12h | Pending |
| 4 — Evaluation Layer | 3 | ~10h | Pending |
| 5 — Data & Features | 3 | ~8h | Pending |
| 6 — Config & Architecture | 2 | ~8h | Pending |
| 7 — Health & Monitoring | 3 | ~8h | Pending |
| **TOTAL** | **22 remaining + 14 done** | **~99h** | W07 next |

**Note**: Phase 0 (8 specs) and W01-W06 (6 specs) are complete. Active work continues at W07 with 22 remaining specs (~69 hours).

---

*Last consolidated: 2026-02-26 — Merged from MASTER_IMPLEMENTATION_SPECS.md + SPECS_STARTUP_AND_ROADMAP.md*
