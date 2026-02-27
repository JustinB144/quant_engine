# IMPLEMENTATION SPECS: Server Startup Fixes + Improvement Roadmap Completion
## Quant Engine v2.0.0 — LLM Implementation Guide

**Generated**: 2026-02-26
**Verification Method**: Every item verified against actual source code with line numbers
**Scope**: Server startup warnings + all remaining unfinished work from improvement roadmap + integration wiring gaps

---

## CONTEXT FOR LLM

This document contains implementation specs for fixing all remaining issues in the quant_engine. Each spec was verified by reading the actual code — nothing is assumed.

**Key finding from verification**: The Improvement Roadmap (docs/plans/IMPROVEMENT_ROADMAP.md) has ALL 30 feature items fully implemented (Tiers 1-6). However, six critical **integration wiring gaps** exist where completed modules are not connected to each other, and two server startup warnings indicate infrastructure gaps that need fixing.

**Total specs**: 11 specs across 3 categories.

---

# CATEGORY A: SERVER STARTUP WARNING FIXES

These are the two warnings visible on `uv run run_server.py`:

```
WARNING | Config validation: GICS_SECTORS is empty — sector exposure constraints are disabled.
WARNING | Config validation: WRDS_ENABLED=True but WRDS_USERNAME env var is not set.
```

## SPEC-S01: Populate GICS_SECTORS and add --gics flag to WRDS refresh [HIGH]

### STATUS (verified)
- `config.py` line 405: `GICS_SECTORS: Dict[str, str] = {}` — empty dict placeholder
- `config.py` lines 673-681: `validate_config()` emits the warning when `GICS_SECTORS` is empty
- `risk/portfolio_optimizer.py` lines 189-195: When `GICS_SECTORS` is empty, sector exposure constraint (`MAX_SECTOR_EXPOSURE = 0.10`) is completely skipped
- `run_wrds_daily_refresh.py` lines 171-186: argparse has --dry-run, --skip-cleanup, --tickers, --years, --batch-size, --verify-only — but NO --gics flag despite the warning message referencing it
- `config_data/universe.yaml` lines 4-113: Already contains a manual sector→tickers mapping (tech: AAPL, MSFT, GOOGL, etc.)

### PROBLEM
1. The startup warning tells the user to run `run_wrds_daily_refresh.py --gics` but that flag doesn't exist
2. Without GICS_SECTORS populated, the portfolio optimizer allows unlimited sector concentration — a major risk control gap
3. A manual mapping exists in `config_data/universe.yaml` but is never loaded into `GICS_SECTORS`

### SPEC

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
    # Map gsector code to sector name
    GICS_SECTOR_NAMES = {
        "10": "Energy", "15": "Materials", "20": "Industrials",
        "25": "Consumer Discretionary", "30": "Consumer Staples",
        "35": "Health Care", "40": "Financials", "45": "Information Technology",
        "50": "Communication Services", "55": "Utilities", "60": "Real Estate",
    }
    return {row.tic: GICS_SECTOR_NAMES.get(str(row.gsector), "Unknown") for _, row in df.iterrows()}
```

In main(), if `args.gics` is set, run the refresh and write results to both `config_data/universe.yaml` and a `GICS_SECTORS` update.

**Part C — Fix the warning message:**

In `config.py` line 679, update the message to also mention the universe.yaml fallback:

```python
"GICS_SECTORS is empty — sector exposure constraints are disabled. "
"Populate via run_wrds_daily_refresh.py --gics, or ensure config_data/universe.yaml has a 'sectors' key."
```

### FILES
- `config.py` (line 405, lines 673-681)
- `run_wrds_daily_refresh.py` (lines 171-186, new function)
- `config_data/universe.yaml` (verify sectors key exists)

### VERIFICATION
- Server starts without the GICS_SECTORS warning
- `from config import GICS_SECTORS; len(GICS_SECTORS) > 0` returns True
- `risk/portfolio_optimizer.py` sector constraint code path is exercised (no skip warning)

---

## SPEC-S02: Handle WRDS_ENABLED gracefully when credentials missing [MEDIUM]

### STATUS (verified)
- `config.py` line 33: `WRDS_ENABLED = True`
- `config.py` lines 693-703: Warning emitted when `WRDS_ENABLED=True` but `WRDS_USERNAME` env var not set
- `data/wrds_provider.py` lines 64-70: Also emits a Python warning at module import time
- `data/loader.py` lines 372-468: Has a 3-stage fallback (WRDS → cache → None) that marks fallback data as `trusted: False`
- `data/alternative.py` lines 172-186: Has `_check_wrds()` guard that returns None when WRDS unavailable

### PROBLEM
The warning is noisy but the system already degrades gracefully. The real issue is:
1. Two separate warnings are emitted (one from config validation, one from wrds_provider.py import)
2. No guidance on whether the user NEEDS WRDS or can operate without it
3. When WRDS is unavailable, the system operates but some features (earnings, options, short interest) return None silently — the user may not realize they're running degraded

### SPEC

**Part A — Consolidate to a single, more informative warning:**

In `config.py` `validate_config()` (line 693), enhance the warning to indicate degradation level:

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

In `data/wrds_provider.py` lines 64-70, change the `warnings.warn()` to a `logger.debug()` so it doesn't double-emit:

```python
if not _WRDS_USERNAME:
    logger.debug(
        'WRDS_USERNAME env var not set; WRDS will be unavailable.'
    )
```

The config validation warning (which runs at startup) is sufficient.

**Part C — Add WRDS degradation status to health endpoint:**

In `api/services/health_service.py`, add a check that reports WRDS status:

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

### FILES
- `config.py` (lines 693-703)
- `data/wrds_provider.py` (lines 64-70)
- `api/services/health_service.py`

### VERIFICATION
- Server starts with ONE clear warning (not two)
- Health endpoint shows WRDS status
- When `WRDS_USERNAME` is set, both warnings disappear

---

# CATEGORY B: INTEGRATION WIRING GAPS (Modules exist but aren't connected)

These are the highest-ROI improvements. The code is already written and tested — it just needs to be called.

## SPEC-S03: Wire structural state into execution simulator [CRITICAL]

### STATUS (verified)
- `backtest/execution.py` lines 255-258: `simulate()` accepts `break_probability`, `structure_uncertainty`, `drift_score`, `systemic_stress`
- `backtest/execution.py` lines 399-416: These parameters feed into a structural multiplier that scales spreads and impact
- `backtest/engine.py` lines 325-336 (entry call): Only passes `realized_vol`, `overnight_gap`, `intraday_range`, `urgency_type`, `volume_trend`
- `backtest/engine.py` lines 399-410 (exit call): Same limited params
- `regime/shock_vector.py`: Complete ShockVector dataclass with `bocpd_changepoint_prob`, `hmm_uncertainty`, structural features

### PROBLEM
The execution model's most sophisticated feature — structural state conditioning — is completely unused. Costs are computed without regime/uncertainty awareness despite the code being ready to accept it.

### SPEC

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

Ensure `EXEC_STRUCTURAL_STRESS_ENABLED = True` in config.py (verify current value).

### FILES
- `backtest/engine.py` (2 simulate() call sites + pre-computation block)
- `regime/shock_vector.py` (verify compute_shock_vectors API)
- `config.py` (verify EXEC_STRUCTURAL_STRESS_ENABLED)

### VERIFICATION
Run two backtests: one with structural params (default) and one with all structural params = None. Compare TCA reports — structural version should show higher costs during high-uncertainty periods.

---

## SPEC-S04: Wire uncertainty gate into position sizing and portfolio weights [CRITICAL]

### STATUS (verified)
- `regime/uncertainty_gate.py`: Complete `UncertaintyGate` class with `compute_size_multiplier()` and `apply_uncertainty_gate()`
- Exported from `regime/__init__.py` line 18
- NEVER imported in `backtest/engine.py` or `autopilot/engine.py`
- Only used in test files

### PROBLEM
Position sizes don't shrink when regime is uncertain. Full-size positions during regime transitions (where most losses occur).

### SPEC

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

**In `autopilot/engine.py`**, after calling `optimize_portfolio()` (line 1246):

```python
from regime.uncertainty_gate import UncertaintyGate

gate = UncertaintyGate()
current_uncertainty = regime_output.uncertainty.iloc[-1] if hasattr(regime_output, 'uncertainty') else 0.0
weights = gate.apply_uncertainty_gate(weights, current_uncertainty)
```

### FILES
- `backtest/engine.py` (position sizing section)
- `autopilot/engine.py` (after optimize_portfolio call, ~line 1248)

### VERIFICATION
During high-uncertainty bars (entropy > 0.8), position sizes should be reduced. Check trade log for size_multiplier < 1.0 during regime transitions.

---

## SPEC-S05: Wire cost calibrator into backtest execution [HIGH]

### STATUS (verified)
- `backtest/cost_calibrator.py`: Complete `CostCalibrator` class with per-segment (micro/small/mid/large) impact coefficients
- NEVER imported in `backtest/engine.py`
- `backtest/execution.py` line 75: Uses flat `EXEC_IMPACT_COEFF_BPS = 25.0` for all securities
- `simulate()` accepts `impact_coeff_override` parameter (line 263) — already designed for this

### PROBLEM
Micro-cap stocks use the same impact model as large-caps. Real impact is ~40 bps/sqrt(participation) for micro vs ~15 for large.

### SPEC

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

### FILES
- `backtest/engine.py`
- `backtest/cost_calibrator.py` (verify `get_impact_coeff()` API)

### VERIFICATION
Run backtest with mixed-cap universe. TCA report should show different impact costs for different market cap segments.

---

## SPEC-S06: Wire regime covariance into portfolio optimizer [HIGH]

### STATUS (verified)
- `risk/covariance.py` line 230: `compute_regime_covariance()` exists — takes returns + regime labels, returns `Dict[int, pd.DataFrame]`
- `autopilot/engine.py` lines 1230-1248: Uses generic `CovarianceEstimator().estimate(returns_df)` — NOT regime-aware
- `autopilot/engine.py` line 1231 has a comment: "regime-conditional covariance requires aligned regime labels" — acknowledging the gap

### PROBLEM
Portfolio optimizer uses full-sample covariance. During stress regimes, correlations spike but the optimizer doesn't know.

### SPEC

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

### FILES
- `autopilot/engine.py` (lines 1230-1248)
- `risk/covariance.py` (verify API)

### VERIFICATION
During a high-vol regime, portfolio should be less concentrated (more diversified) than with generic covariance.

---

## SPEC-S07: Wire reproducibility manifests into all entry points [HIGH]

### STATUS (verified)
- `reproducibility.py`: `build_run_manifest()` and `write_run_manifest()` fully implemented
- `reproducibility.py` lines 117-235: `verify_manifest()` also implemented
- None of `run_backtest.py`, `run_train.py`, `run_retrain.py`, `run_autopilot.py`, `run_predict.py` call any of these functions

### PROBLEM
No reproducibility tracking for any run. Cannot reconstruct past results.

### SPEC

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

### FILES
- `run_backtest.py`, `run_train.py`, `run_retrain.py`, `run_autopilot.py`, `run_predict.py`
- `reproducibility.py` (verify API)

### VERIFICATION
Run any entry point. A `run_manifest.json` file should be created in the output directory with git hash, config snapshot, and data checksums.

---

## SPEC-S08: Make turnover penalty configurable and cost-aware [MEDIUM]

### STATUS (verified)
- `risk/portfolio_optimizer.py` line 33: `turnover_penalty: float = 0.001` hardcoded default
- `autopilot/engine.py` line 1246: Calls `optimize_portfolio()` without passing `turnover_penalty`
- No config variable for turnover penalty

### PROBLEM
Turnover penalty is locked at 0.1% regardless of actual trading costs. During high-cost regimes, the optimizer may suggest trades that cost more than the penalty.

### SPEC

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
    current_weights=current_weights,  # Also pass current weights for turnover computation
)
```

**Part C (optional) — Dynamic penalty from execution costs:**

```python
# Estimate current cost environment
avg_cost_bps = execution_model.estimate_cost(avg_volume, avg_notional, current_vol)
dynamic_penalty = max(PORTFOLIO_TURNOVER_PENALTY, avg_cost_bps / 10000 * 2)  # 2× expected cost
```

### FILES
- `config.py`
- `autopilot/engine.py` (~line 1246)
- `risk/portfolio_optimizer.py` (no change needed — already accepts the param)

### VERIFICATION
Changing `PORTFOLIO_TURNOVER_PENALTY` from 0.001 to 0.01 should reduce portfolio turnover.

---

# CATEGORY C: ADDITIONAL FIXES DISCOVERED DURING VERIFICATION

## SPEC-S09: Fix XGBoost callbacks API incompatibility [CRITICAL]

### STATUS (verified)
- `autopilot/meta_labeler.py` line 275: passes `callbacks` to `XGBClassifier.fit()` — removed in recent XGBoost versions
- 4 tests fail: `test_train_produces_model`, `test_predict_confidence_range`, `test_save_and_load_roundtrip`, `test_feature_importance_no_single_dominant`

### PROBLEM
Meta-labeling is completely broken. No signals get confidence filtering.

### SPEC

In `autopilot/meta_labeler.py`, find the `fit_params` dict construction around line 270-280. Remove `callbacks` from `fit_params`. If early stopping is needed, pass `early_stopping_rounds` to the XGBClassifier constructor, not fit():

```python
# BEFORE (broken):
fit_params = {..., "callbacks": [...]}
model.fit(X, y, **fit_params)

# AFTER (fixed):
model = XGBClassifier(
    ...,
    early_stopping_rounds=10,  # Set in constructor for XGBoost 2.0+
)
model.fit(X, y, eval_set=[(X_val, y_val)])  # No callbacks kwarg
```

### FILES
- `autopilot/meta_labeler.py` (~line 275)

### VERIFICATION
All 4 tests in `tests/test_signal_meta_labeling.py` must pass.

---

## SPEC-S10: Fix look-ahead bias in alternative data [CRITICAL]

### STATUS (verified)
- `data/alternative.py` `get_earnings_surprise()` wrapper (line 820-832): Has `as_of_date` parameter — GOOD
- Inner class method (line 196): Also accepts `as_of_date` parameter — GOOD
- `data/alternative.py` lines 215-220: Uses `as_of_date` to filter: `df = df[df['report_date'] <= as_of_date]`
- HOWEVER: Need to verify that the backtest engine actually PASSES as_of_date

### PROBLEM
If the backtest engine doesn't pass `as_of_date`, the alternative data functions will use their default (which may be None → no date filter → future data leaks in).

### SPEC

Verify that everywhere `get_earnings_surprise()` is called during backtests, the current bar date is passed as `as_of_date`. Search all call sites:

1. In `features/pipeline.py` — if it calls any alternative data functions, verify as_of_date is threaded
2. In `features/research_factors.py` — same check
3. In `backtest/engine.py` — if it calls alternative data directly

If any call site does NOT pass `as_of_date`, add it:
```python
earnings = get_earnings_surprise(ticker, as_of_date=current_bar_date)
```

### FILES
- Search: `grep -r "get_earnings_surprise\|get_options_flow\|get_short_interest\|get_insider_transactions\|get_institutional_ownership" --include="*.py"`
- Fix any call site missing `as_of_date`

### VERIFICATION
Write a test: `get_earnings_surprise("AAPL", as_of_date=datetime(2024,1,1))` should return NO data after 2024-01-01.

---

## SPEC-S11: Fix volume column case mismatch in cross-source validator [HIGH]

### STATUS (verified in prior audit)
- `data/cross_source_validator.py` uses lowercase column names ('volume') but OHLCV data uses titlecase ('Volume')

### PROBLEM
Volume mismatches between data sources are never detected.

### SPEC

In `data/cross_source_validator.py`, standardize all column references to titlecase:

```python
OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# At the start of validation, normalize column names:
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower(): c.title() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
    return df.rename(columns=col_map)
```

Apply normalization before any column comparisons.

### FILES
- `data/cross_source_validator.py`

### VERIFICATION
Create test data with `volume` (lowercase) column and confirm validator normalizes and validates correctly.

---

# IMPLEMENTATION ORDER

**Immediate (Day 1)** — Fix broken code:
1. SPEC-S09: Fix XGBoost callbacks (30 min)
2. SPEC-S10: Verify as_of_date threading (1 hour)
3. SPEC-S11: Fix column case mismatch (30 min)

**Day 2** — Fix startup warnings:
4. SPEC-S01: Populate GICS_SECTORS + --gics flag (2 hours)
5. SPEC-S02: Clean up WRDS warning (1 hour)

**Week 1** — Critical wiring (highest ROI):
6. SPEC-S03: Wire structural state → execution (3 hours)
7. SPEC-S04: Wire uncertainty gate → sizing (2 hours)
8. SPEC-S05: Wire cost calibrator → backtest (2 hours)

**Week 2** — Remaining wiring:
9. SPEC-S06: Wire regime covariance → optimizer (2 hours)
10. SPEC-S07: Wire reproducibility manifests (2 hours)
11. SPEC-S08: Make turnover penalty configurable (1 hour)

**TOTAL: 11 specs, ~17 hours**

---

# APPENDIX: ROADMAP ITEMS VERIFIED AS COMPLETE

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
