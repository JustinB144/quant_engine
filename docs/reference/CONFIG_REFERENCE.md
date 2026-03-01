# Configuration Reference

Source-derived configuration reference for `config.py`, `api/config.py`, and `config_structured.py` (current working tree).

## `config.py` (Flat Runtime Configuration)

Notes:
- Values are source expressions (not evaluated runtime values).
- `Status` / `Notes` columns are parsed from inline `# STATUS:` annotations when present.

## Paths

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `ROOT_DIR` | `Path(__file__).parent` | `ACTIVE` | base path for all relative references | 26 |
| `FRAMEWORK_DIR` | `ROOT_DIR.parent` | `ACTIVE` | parent directory of quant_engine | 27 |
| `MODEL_DIR` | `ROOT_DIR / "trained_models"` | `ACTIVE` | models/versioning.py, models/trainer.py | 28 |
| `RESULTS_DIR` | `ROOT_DIR / "results"` | `ACTIVE` | backtest output, autopilot reports, alerts | 29 |

## Data Sources

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `DATA_CACHE_DIR` | `ROOT_DIR / "data" / "cache"` | `ACTIVE` | data/loader.py, data/local_cache.py | 32 |
| `WRDS_ENABLED` | `True` | `ACTIVE` | data/loader.py; try WRDS first, fall back to local cache / IBKR | 33 |
| `OPTIONMETRICS_ENABLED` | `False` | `PLACEHOLDER` | data/loader.py gates on this but pipeline incomplete | 41 |

## Execution Contract (Truth Layer)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `RET_TYPE` | `"log"` | `ACTIVE` | "log" (log returns) or "simple" (pct returns) | 44 |
| `LABEL_H` | `5` | `ACTIVE` | label horizon in trading days | 45 |
| `PX_TYPE` | `"close"` | `ACTIVE` | "close" or "open" for price baseline | 46 |
| `ENTRY_PRICE_TYPE` | `"next_bar_open"` | `ACTIVE` | "next_bar_open" (no look-ahead) | 47 |

## Truth Layer Feature Flags

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `TRUTH_LAYER_STRICT_PRECONDITIONS` | `True` | `ACTIVE` | raise on invalid execution contract | 50 |
| `TRUTH_LAYER_FAIL_ON_CORRUPT` | `True` | `ACTIVE` | block corrupt OHLCV from pipeline | 51 |
| `TRUTH_LAYER_ENFORCE_CAUSALITY` | `True` | `ACTIVE` | enforce feature causality at runtime | 52 |
| `TRUTH_LAYER_COMPUTE_NULL_BASELINES` | `False` | `ACTIVE` | compute null baselines per backtest (adds ~4x time) | 53 |
| `TRUTH_LAYER_COST_STRESS_ENABLED` | `False` | `ACTIVE` | run cost stress sweep per backtest (adds ~4x time) | 54 |

## Cost Stress Testing

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `COST_STRESS_MULTIPLIERS` | `[0.5, 1.0, 2.0, 5.0]` | `ACTIVE` | cost sweep factors | 57 |
| `KALSHI_ENABLED` | `False` | `ACTIVE` | kalshi/provider.py, run_kalshi_event_pipeline.py; disabled by design | 59 |
| `KALSHI_ENV` | `"demo"` | `ACTIVE` | selects demo vs prod API URL; "demo" (safety) or "prod" | 60 |
| `KALSHI_DEMO_API_BASE_URL` | `"https://demo-api.kalshi.co/trade-api/v2"` | `ACTIVE` | used to compute KALSHI_API_BASE_URL | 61 |
| `KALSHI_PROD_API_BASE_URL` | `"https://api.elections.kalshi.com/trade-api/v2"` | `ACTIVE` | used when KALSHI_ENV="prod" | 62 |
| `KALSHI_API_BASE_URL` | `KALSHI_DEMO_API_BASE_URL` | `ACTIVE` | kalshi/provider.py base URL | 63 |
| `KALSHI_HISTORICAL_API_BASE_URL` | `KALSHI_API_BASE_URL` | `ACTIVE` | kalshi historical data endpoint | 64 |
| `KALSHI_HISTORICAL_CUTOFF_TS` | `None` | `ACTIVE` | optional cutoff for historical fetches | 65 |
| `KALSHI_RATE_LIMIT_RPS` | `6.0` | `ACTIVE` | kalshi/provider.py rate limiter | 66 |
| `KALSHI_RATE_LIMIT_BURST` | `2` | `ACTIVE` | kalshi/provider.py burst allowance | 67 |
| `KALSHI_DB_PATH` | `ROOT_DIR / "data" / "kalshi.duckdb"` | `ACTIVE` | DuckDB path for Kalshi snapshots | 68 |
| `KALSHI_SNAPSHOT_HORIZONS` | `["7d", "1d", "4h", "1h", "15m", "5m"]` | `ACTIVE` | kalshi/provider.py, run_kalshi_event_pipeline.py | 69 |
| `KALSHI_DISTRIBUTION_FREQ` | `"5min"` | `ACTIVE` | kalshi/provider.py distribution resampling | 70 |
| `KALSHI_STALE_AFTER_MINUTES` | `30` | `ACTIVE` | kalshi/provider.py staleness detection | 71 |
| `KALSHI_NEAR_EVENT_MINUTES` | `30.0` | `ACTIVE` | kalshi/provider.py near-event window | 72 |
| `KALSHI_NEAR_EVENT_STALE_MINUTES` | `2.0` | `ACTIVE` | tighter staleness near events | 73 |
| `KALSHI_FAR_EVENT_MINUTES` | `24.0 * 60.0` | `ACTIVE` | far-event window (24h) | 74 |
| `KALSHI_FAR_EVENT_STALE_MINUTES` | `60.0` | `ACTIVE` | relaxed staleness far from events | 75 |
| `KALSHI_STALE_MARKET_TYPE_MULTIPLIERS` | `{           # STATUS: ACTIVE — kalshi/provider.py per-event-type staleness
    "CPI": 0.80,
    "UNEMPLOYMENT": 0.90,...` | `ACTIVE` | kalshi/provider.py per-event-type staleness | 76 |
| `KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD` | `2.0` | `ACTIVE` | kalshi/provider.py low-liquidity threshold | 82 |
| `KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD` | `6.0` | `ACTIVE` | kalshi/provider.py high-liquidity threshold | 83 |
| `KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER` | `1.35` | `ACTIVE` | tighten staleness for illiquid markets | 84 |
| `KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER` | `0.80` | `ACTIVE` | relax staleness for liquid markets | 85 |
| `KALSHI_DISTANCE_LAGS` | `["1h", "1d"]` | `ACTIVE` | kalshi/provider.py distance lag features | 86 |
| `KALSHI_TAIL_THRESHOLDS` | `{                         # STATUS: ACTIVE — kalshi/provider.py tail-risk thresholds
    "CPI": [3.0, 3.5, 4.0],
    ...` | `ACTIVE` | kalshi/provider.py tail-risk thresholds | 87 |
| `DEFAULT_UNIVERSE_SOURCE` | `"wrds"` | `PLACEHOLDER` | defined but never imported; "wrds", "static", or "ibkr" | 93 |
| `CACHE_TRUSTED_SOURCES` | `["wrds", "wrds_delisting", "ibkr"]` | `ACTIVE` | data/local_cache.py source ranking | 94 |
| `CACHE_MAX_STALENESS_DAYS` | `21` | `ACTIVE` | data/local_cache.py max cache age | 95 |
| `CACHE_WRDS_SPAN_ADVANTAGE_DAYS` | `180` | `ACTIVE` | data/local_cache.py WRDS preference window | 96 |
| `REQUIRE_PERMNO` | `True` | `ACTIVE` | data/loader.py, backtest/engine.py PERMNO validation | 97 |

## Survivorship

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `SURVIVORSHIP_DB` | `ROOT_DIR / "data" / "universe_history.db"` | `ACTIVE` | data/survivorship.py, data/loader.py | 100 |
| `SURVIVORSHIP_UNIVERSE_NAME` | `"SP500"` | `ACTIVE` | data/loader.py, autopilot/engine.py | 101 |
| `SURVIVORSHIP_SNAPSHOT_FREQ` | `"quarterly"` | `ACTIVE` | data/loader.py; "annual" or "quarterly" | 102 |

## Model Versioning

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `MODEL_REGISTRY` | `MODEL_DIR / "registry.json"` | `PLACEHOLDER` | defined but never imported; CHAMPION_REGISTRY is used instead | 105 |
| `MAX_MODEL_VERSIONS` | `5` | `ACTIVE` | models/versioning.py; keep last 5 versions for rollback | 106 |
| `CHAMPION_REGISTRY` | `MODEL_DIR / "champion_registry.json"` | `ACTIVE` | models/governance.py | 107 |

## Retraining

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `RETRAIN_MAX_DAYS` | `30` | `ACTIVE` | schedule-based retrain trigger (30 calendar days) | 110 |
| `RETRAIN_MIN_TRADES` | `50` | `PLACEHOLDER` | defined but never imported by running code | 111 |
| `RETRAIN_MIN_WIN_RATE` | `0.45` | `PLACEHOLDER` | defined but never imported by running code | 112 |
| `RETRAIN_MIN_CORRELATION` | `0.05` | `PLACEHOLDER` | minimum OOS Spearman; defined but never imported | 113 |
| `RETRAIN_REGIME_CHANGE_DAYS` | `10` | `ACTIVE` | run_retrain.py; trigger retrain if regime changed for 10+ consecutive days | 114 |
| `RECENCY_DECAY` | `0.003` | `ACTIVE` | models/trainer.py; exponential recency weighting (1yr weight ~ 0.33) | 115 |

## Universe

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `UNIVERSE_FULL` | `[                                  # STATUS: ACTIVE — data/loader.py, run_*.py, autopilot/engine.py
    # Large cap t...` | `ACTIVE` | data/loader.py, run_*.py, autopilot/engine.py | 118 |
| `UNIVERSE_QUICK` | `[                                 # STATUS: ACTIVE — run_*.py quick-test universe
    "AAPL", "MSFT", "GOOGL", "NVDA"...` | `ACTIVE` | run_*.py quick-test universe | 135 |
| `UNIVERSE_INTRADAY` | `[                              # STATUS: ACTIVE — 128-ticker intraday universe (all IBKR-downloaded timeframes)
    "...` | `ACTIVE` | 128-ticker intraday universe (all IBKR-downloaded timeframes) | 140 |
| `BENCHMARK` | `"SPY"` | `ACTIVE` | backtest/engine.py, api/routers/benchmark.py | 156 |

## Data

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `LOOKBACK_YEARS` | `15` | `ACTIVE` | data/loader.py; years of historical data to load | 159 |
| `MIN_BARS` | `500` | `ACTIVE` | data/loader.py; minimum bars needed for feature warm-up | 160 |

## Intraday Data

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `INTRADAY_TIMEFRAMES` | `["4h", "1h", "30m", "15m", "5m", "1m"]` | `ACTIVE` | run_wrds_taq_intraday_download.py; all supported intraday bar sizes | 177 |
| `INTRADAY_CACHE_SOURCE` | `"wrds_taq"` | `ACTIVE` | run_wrds_taq_intraday_download.py; NYSE TAQ Daily Product source | 178 |
| `TAQ_START_DATE` | `"2003-09-10"` | `ACTIVE` | run_wrds_taq_intraday_download.py; earliest NYSE TAQ Daily Product date | 179 |
| `INTRADAY_MIN_BARS` | `100` | `ACTIVE` | features/pipeline.py; minimum intraday bars for feature computation | 165 |
| `MARKET_OPEN` | `"09:30"` | `ACTIVE` | features/intraday.py; US equity regular-session open (ET) | 166 |
| `MARKET_CLOSE` | `"16:00"` | `ACTIVE` | features/intraday.py; US equity regular-session close (ET) | 167 |

### Intraday Indicators (SPEC_INTRADAY_02)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `INTRADAY_MIN_BARS` | `100` | `ACTIVE` | run_intraday_indicators.py; minimum 1m bars per ticker to compute indicators | 165 |
| `REQUIRE_PERMNO` | `True` | `ACTIVE` | run_intraday_indicators.py; resolve ticker → CRSP PERMNO for feature store keys | 111 |
| Feature version | `intraday_v1` | n/a | Feature store version tag for intraday indicators | n/a |

## Intraday Data Integrity (SPEC_11)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `INTRADAY_VALIDATION_ENABLED` | `True` | `ACTIVE` | scripts/alpaca_intraday_download.py; enable IBKR cross-validation | 170 |
| `INTRADAY_CLOSE_TOLERANCE_PCT` | `0.15` | `ACTIVE` | data/cross_source_validator.py; max close price diff vs IBKR (%) | 171 |
| `INTRADAY_OPEN_TOLERANCE_PCT` | `0.20` | `ACTIVE` | data/cross_source_validator.py; max open price diff vs IBKR (%) | 172 |
| `INTRADAY_HIGHLOW_TOLERANCE_PCT` | `0.25` | `ACTIVE` | data/cross_source_validator.py; max H/L price diff vs IBKR (%) | 173 |
| `INTRADAY_VOLUME_TOLERANCE_PCT` | `5.0` | `ACTIVE` | data/cross_source_validator.py; max volume diff vs IBKR (%) | 174 |
| `INTRADAY_MAX_REJECTED_BAR_PCT` | `5.0` | `ACTIVE` | data/intraday_quality.py; quarantine if >N% bars rejected | 175 |
| `INTRADAY_MAX_MISMATCH_RATE_PCT` | `5.0` | `ACTIVE` | data/cross_source_validator.py; quarantine if >N% bars mismatch | 176 |
| `INTRADAY_VALIDATION_SAMPLE_WINDOWS` | `10` | `ACTIVE` | data/cross_source_validator.py; stratified date sampling windows | 177 |
| `INTRADAY_VALIDATION_DAYS_PER_WINDOW` | `2` | `ACTIVE` | data/cross_source_validator.py; days sampled per window | 178 |
| `INTRADAY_QUARANTINE_DIR` | `DATA_CACHE_DIR / "quarantine"` | `ACTIVE` | data/intraday_quality.py; quarantined data location | 179 |

## Targets

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `FORWARD_HORIZONS` | `[5, 10, 20]` | `ACTIVE` | models/trainer.py, run_*.py; days ahead to predict | 182 |

## Features

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `INTERACTION_PAIRS` | `[                              # STATUS: ACTIVE — features/pipeline.py; regime-conditional interaction features
    #...` | `ACTIVE` | features/pipeline.py; regime-conditional interaction features | 185 |

## Regime

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `REGIME_NAMES` | `{                                   # STATUS: ACTIVE — used in 15+ files; canonical regime label mapping
    0: "tren...` | `ACTIVE` | used in 15+ files; canonical regime label mapping | 208 |
| `MIN_REGIME_SAMPLES` | `50` | `ACTIVE` | models/trainer.py; minimum training samples per regime model (reduced from 100 for short regimes, SPEC_10 T7) | 222 |
| `REGIME_MODEL_TYPE` | `"jump"` | `ACTIVE` | regime/detector.py; "jump", "hmm", or "rule" | 215 |
| `REGIME_HMM_STATES` | `4` | `ACTIVE` | regime/hmm.py; number of hidden states | 216 |
| `REGIME_HMM_MAX_ITER` | `60` | `ACTIVE` | regime/hmm.py; EM iteration limit | 217 |
| `REGIME_HMM_STICKINESS` | `0.92` | `ACTIVE` | regime/hmm.py; diagonal prior bias for sticky transitions | 218 |
| `REGIME_MIN_DURATION` | `3` | `ACTIVE` | regime/detector.py; minimum regime duration in days | 219 |
| `REGIME_SOFT_ASSIGNMENT_THRESHOLD` | `0.35` | `ACTIVE` | models/trainer.py; probability threshold for soft regime labels | 220 |
| `REGIME_HMM_PRIOR_WEIGHT` | `0.3` | `ACTIVE` | regime/hmm.py; shrinkage weight toward sticky prior in M-step | 221 |
| `REGIME_HMM_COVARIANCE_TYPE` | `"full"` | `ACTIVE` | regime/hmm.py; "full" (captures return-vol correlation) or "diag" | 222 |
| `REGIME_HMM_AUTO_SELECT_STATES` | `True` | `ACTIVE` | regime/hmm.py; use BIC to select optimal number of states | 223 |
| `REGIME_HMM_MIN_STATES` | `2` | `ACTIVE` | regime/hmm.py; BIC state search lower bound | 224 |
| `REGIME_HMM_MAX_STATES` | `6` | `ACTIVE` | regime/hmm.py; BIC state search upper bound | 225 |
| `REGIME_JUMP_MODEL_ENABLED` | `True` | `ACTIVE` | regime/detector.py; statistical jump model alongside HMM | 226 |
| `REGIME_JUMP_PENALTY` | `0.02` | `ACTIVE` | regime/jump_model.py; jump penalty lambda (higher = fewer transitions) | 227 |
| `REGIME_EXPECTED_CHANGES_PER_YEAR` | `4` | `ACTIVE` | regime/jump_model.py; calibrate jump penalty from expected regime changes/yr | 228 |
| `REGIME_ENSEMBLE_ENABLED` | `True` | `ACTIVE` | regime/detector.py; combine HMM + JM + rule-based via majority vote | 229 |
| `REGIME_ENSEMBLE_CONSENSUS_THRESHOLD` | `2` | `ACTIVE` | regime/detector.py; require N of 3 methods to agree for transition | 230 |
| `REGIME_JUMP_USE_PYPI_PACKAGE` | `True` | `ACTIVE` | regime/jump_model_pypi.py; True=PyPI jumpmodels, False=legacy custom | 233 |
| `REGIME_JUMP_CV_FOLDS` | `5` | `ACTIVE` | regime/jump_model_pypi.py; time-series CV folds for lambda selection | 234 |
| `REGIME_JUMP_LAMBDA_RANGE` | `(0.005, 0.15)` | `ACTIVE` | regime/jump_model_pypi.py; search range for jump penalty | 235 |
| `REGIME_JUMP_LAMBDA_STEPS` | `20` | `ACTIVE` | regime/jump_model_pypi.py; grid points for lambda search | 236 |
| `REGIME_JUMP_MAX_ITER` | `50` | `ACTIVE` | regime/jump_model_pypi.py; coordinate descent iterations | 237 |
| `REGIME_JUMP_TOL` | `1e-6` | `ACTIVE` | regime/jump_model_pypi.py; convergence tolerance | 238 |
| `REGIME_JUMP_USE_CONTINUOUS` | `True` | `ACTIVE` | regime/jump_model_pypi.py; continuous JM for soft probabilities | 239 |
| `REGIME_JUMP_MODE_LOSS_WEIGHT` | `0.1` | `ACTIVE` | regime/jump_model_pypi.py; mode loss penalty (continuous JM) | 240 |

## Regime Detection Upgrade (SPEC_10)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `REGIME_ENSEMBLE_DEFAULT_WEIGHTS` | `{               # STATUS: ACTIVE — regime/detector.py; default component weights before calibration
    "hmm": 0.5,
 ...` | `ACTIVE` | regime/detector.py; default component weights before calibration | 244 |
| `REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD` | `0.40` | `ACTIVE` | regime/detector.py; if max weighted vote < this, regime is uncertain | 249 |
| `REGIME_ENSEMBLE_UNCERTAIN_FALLBACK` | `3` | `ACTIVE` | regime/detector.py; regime to assume when uncertain (3=high_volatility) | 250 |
| `REGIME_UNCERTAINTY_ENTROPY_THRESHOLD` | `0.50` | `ACTIVE` | regime/uncertainty_gate.py; flag if normalized entropy > this | 253 |
| `REGIME_UNCERTAINTY_STRESS_THRESHOLD` | `0.80` | `ACTIVE` | regime/uncertainty_gate.py; assume stress if entropy > this | 254 |
| `REGIME_UNCERTAINTY_SIZING_MAP` | `{                 # STATUS: ACTIVE — regime/uncertainty_gate.py; entropy->sizing multiplier
    0.0: 1.0,
    0.5: 0....` | `ACTIVE` | regime/uncertainty_gate.py; entropy->sizing multiplier | 255 |
| `REGIME_UNCERTAINTY_MIN_MULTIPLIER` | `0.80` | `ACTIVE` | regime/uncertainty_gate.py; floor for sizing multiplier | 260 |
| `REGIME_CONSENSUS_THRESHOLD` | `0.80` | `ACTIVE` | regime/consensus.py; high confidence consensus threshold | 263 |
| `REGIME_CONSENSUS_EARLY_WARNING` | `0.60` | `ACTIVE` | regime/consensus.py; early warning if consensus drops below | 264 |
| `REGIME_CONSENSUS_DIVERGENCE_WINDOW` | `20` | `ACTIVE` | regime/consensus.py; window for trend detection | 265 |
| `REGIME_CONSENSUS_DIVERGENCE_SLOPE` | `-0.01` | `ACTIVE` | regime/consensus.py; slope threshold for divergence | 266 |
| `REGIME_ONLINE_UPDATE_ENABLED` | `True` | `ACTIVE` | regime/online_update.py; enable incremental HMM updates | 269 |
| `REGIME_ONLINE_REFIT_DAYS` | `30` | `ACTIVE` | regime/online_update.py; full refit every N days | 270 |
| `REGIME_EXPANDED_FEATURES_ENABLED` | `True` | `ACTIVE` | regime/hmm.py; add spectral/SSA/BOCPD to observation matrix | 273 |
| `MIN_REGIME_DAYS` | `10` | `ACTIVE` | models/trainer.py; minimum days in regime before training | 276 |

## Bayesian Online Change-Point Detection

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `BOCPD_ENABLED` | `True` | `ACTIVE` | regime/detector.py, regime/bocpd.py; enable BOCPD alongside HMM | 279 |
| `BOCPD_HAZARD_FUNCTION` | `"constant"` | `ACTIVE` | regime/bocpd.py; "constant" or "geometric" | 280 |
| `BOCPD_HAZARD_LAMBDA` | `1.0 / 60` | `ACTIVE` | regime/bocpd.py; constant hazard rate (1 change per 60 bars) | 281 |
| `BOCPD_LIKELIHOOD_TYPE` | `"gaussian"` | `ACTIVE` | regime/bocpd.py; observation model type (only "gaussian" for v1) | 282 |
| `BOCPD_RUNLENGTH_DEPTH` | `200` | `ACTIVE` | regime/bocpd.py; max run-length to track (older hypotheses pruned) | 283 |
| `BOCPD_CHANGEPOINT_THRESHOLD` | `0.50` | `ACTIVE` | regime/detector.py; flag changepoint if P(cp) > threshold | 284 |

## Shock Vector Schema

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `SHOCK_VECTOR_SCHEMA_VERSION` | `"1.0"` | `ACTIVE` | regime/shock_vector.py; schema version for backward compatibility | 287 |
| `SHOCK_VECTOR_INCLUDE_STRUCTURAL` | `True` | `ACTIVE` | regime/shock_vector.py; include structural features in vector | 288 |

## Kalshi Purge/Embargo by Event Type (E3)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `KALSHI_PURGE_WINDOW_BY_EVENT` | `{"CPI": 14, "FOMC": 21, "NFP": 14, "GDP": 14}` | `PLACEHOLDER` | defined but never imported | 291 |
| `KALSHI_DEFAULT_PURGE_WINDOW` | `10` | `PLACEHOLDER` | defined but never imported; companion to KALSHI_PURGE_WINDOW_BY_EVENT | 292 |

## Model

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `MODEL_PARAMS` | `{                                   # STATUS: ACTIVE — models/trainer.py; GBR hyperparameters
    "n_estimators": 500...` | `ACTIVE` | models/trainer.py; GBR hyperparameters | 295 |
| `MAX_FEATURES_SELECTED` | `30` | `ACTIVE` | models/trainer.py; after permutation importance | 303 |
| `MAX_IS_OOS_GAP` | `0.05` | `ACTIVE` | models/trainer.py; max allowed IS-OOS degradation (R^2 or correlation) | 304 |
| `CV_FOLDS` | `5` | `ACTIVE` | models/trainer.py; cross-validation folds | 305 |
| `HOLDOUT_FRACTION` | `0.15` | `ACTIVE` | models/trainer.py; holdout set fraction | 306 |
| `ENSEMBLE_DIVERSIFY` | `True` | `ACTIVE` | models/trainer.py; train GBR + ElasticNet + RandomForest and average | 307 |

## Walk-Forward Rolling Window

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `WF_MAX_TRAIN_DATES` | `1260` | `ACTIVE` | backtest/engine.py, models/trainer.py; rolling walk-forward window | 313 |

## Backtest

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `TRANSACTION_COST_BPS` | `20` | `ACTIVE` | backtest/engine.py; 20 bps round-trip | 316 |
| `ENTRY_THRESHOLD` | `0.005` | `ACTIVE` | backtest/engine.py; minimum predicted return to enter (0.5%) | 317 |
| `CONFIDENCE_THRESHOLD` | `0.6` | `ACTIVE` | backtest/engine.py; minimum model confidence | 318 |
| `MAX_POSITIONS` | `20` | `ACTIVE` | backtest/engine.py; max simultaneous positions | 319 |
| `POSITION_SIZE_PCT` | `0.05` | `ACTIVE` | backtest/engine.py; 5% of capital per position | 320 |
| `BACKTEST_ASSUMED_CAPITAL_USD` | `1_000_000.0` | `ACTIVE` | backtest/engine.py; initial capital | 321 |
| `EXEC_SPREAD_BPS` | `3.0` | `ACTIVE` | backtest/engine.py; base spread cost | 322 |
| `EXEC_MAX_PARTICIPATION` | `0.02` | `ACTIVE` | backtest/engine.py; max 2% of daily volume | 323 |
| `EXEC_IMPACT_COEFF_BPS` | `25.0` | `ACTIVE` | backtest/engine.py; market impact coefficient | 324 |
| `EXEC_MIN_FILL_RATIO` | `0.20` | `ACTIVE` | backtest/engine.py; minimum fill ratio | 325 |
| `EXEC_DYNAMIC_COSTS` | `True` | `ACTIVE` | backtest/engine.py; condition costs on market state | 326 |
| `EXEC_DOLLAR_VOLUME_REF_USD` | `25_000_000.0` | `ACTIVE` | backtest/engine.py; dollar volume reference | 327 |
| `EXEC_VOL_REF` | `0.20` | `ACTIVE` | backtest/engine.py; reference volatility | 328 |
| `EXEC_VOL_SPREAD_BETA` | `1.0` | `ACTIVE` | backtest/engine.py; vol-spread sensitivity | 329 |
| `EXEC_GAP_SPREAD_BETA` | `4.0` | `ACTIVE` | backtest/engine.py; gap-spread sensitivity | 330 |
| `EXEC_RANGE_SPREAD_BETA` | `2.0` | `ACTIVE` | backtest/engine.py; range-spread sensitivity | 331 |
| `EXEC_VOL_IMPACT_BETA` | `1.0` | `ACTIVE` | backtest/engine.py; vol-impact sensitivity | 332 |

## Structural State-Aware Costs (Spec 06)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `EXEC_STRUCTURAL_STRESS_ENABLED` | `True` | `ACTIVE` | backtest/execution.py; enable structural state cost multipliers | 335 |
| `EXEC_BREAK_PROB_COST_MULT` | `{                     # STATUS: ACTIVE — backtest/execution.py; break probability → cost multiplier tiers
    "low": ...` | `ACTIVE` | backtest/execution.py; break probability → cost multiplier tiers | 336 |
| `EXEC_STRUCTURE_UNCERTAINTY_COST_MULT` | `0.50` | `ACTIVE` | backtest/execution.py; +50% per 1.0 increase in uncertainty | 341 |
| `EXEC_DRIFT_SCORE_COST_REDUCTION` | `0.20` | `ACTIVE` | backtest/execution.py; -20% per 1.0 increase in drift (strong trend = cheaper) | 342 |
| `EXEC_SYSTEMIC_STRESS_COST_MULT` | `0.30` | `ACTIVE` | backtest/execution.py; +30% per 1.0 increase in systemic stress | 343 |

## ADV Computation (Spec 06)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `ADV_LOOKBACK_DAYS` | `20` | `ACTIVE` | backtest/adv_tracker.py; window for ADV calculation | 346 |
| `ADV_EMA_SPAN` | `20` | `ACTIVE` | backtest/adv_tracker.py; EMA smoothing parameter | 347 |
| `EXEC_VOLUME_TREND_ENABLED` | `True` | `ACTIVE` | backtest/execution.py; adjust costs based on volume trend | 348 |
| `EXEC_LOW_VOLUME_COST_MULT` | `1.5` | `ACTIVE` | backtest/adv_tracker.py; +50% on below-average volume days | 349 |

## Entry/Exit Urgency (Spec 06)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `EXEC_EXIT_URGENCY_COST_LIMIT_MULT` | `1.5` | `ACTIVE` | backtest/execution.py; exits tolerate 1.5x higher costs | 352 |
| `EXEC_ENTRY_URGENCY_COST_LIMIT_MULT` | `1.0` | `ACTIVE` | backtest/execution.py; entries use standard cost limits | 353 |
| `EXEC_STRESS_PULLBACK_MIN_SIZE` | `0.10` | `ACTIVE` | backtest/execution.py; reduce order size by 10% per urgency level | 354 |

## Cost Calibration (Spec 06)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `EXEC_CALIBRATION_ENABLED` | `True` | `ACTIVE` | backtest/cost_calibrator.py; enable per-segment calibration | 357 |
| `EXEC_CALIBRATION_MIN_TRADES` | `100` | `ACTIVE` | backtest/cost_calibrator.py; min trades before calibration | 358 |
| `EXEC_CALIBRATION_MIN_SEGMENT_TRADES` | `20` | `ACTIVE` | backtest/cost_calibrator.py; min trades per segment | 359 |
| `EXEC_CALIBRATION_SMOOTHING` | `0.30` | `ACTIVE` | backtest/cost_calibrator.py; new-coefficient weight in EMA update | 360 |
| `EXEC_COST_IMPACT_COEFF_BY_MARKETCAP` | `{           # STATUS: ACTIVE — backtest/cost_calibrator.py; default coefficients by segment
    "micro": 40.0,       ...` | `ACTIVE` | backtest/cost_calibrator.py; default coefficients by segment | 361 |
| `EXEC_MARKETCAP_THRESHOLDS` | `{                     # STATUS: ACTIVE — backtest/cost_calibrator.py; segment boundaries in USD
    "micro": 300e6,
 ...` | `ACTIVE` | backtest/cost_calibrator.py; segment boundaries in USD | 367 |

## No-Trade Gate (Spec 06)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `EXEC_NO_TRADE_STRESS_THRESHOLD` | `0.95` | `ACTIVE` | backtest/execution.py; VIX percentile above which low-urgency orders blocked | 374 |
| `MAX_PORTFOLIO_VOL` | `0.30` | `ACTIVE` | backtest/engine.py, risk/portfolio_optimizer.py; max annualized vol | 376 |
| `REGIME_RISK_MULTIPLIER` | `{                         # STATUS: ACTIVE — backtest/engine.py; regime-conditional position sizing multipliers
    0...` | `ACTIVE` | backtest/engine.py; regime-conditional position sizing multipliers | 377 |
| `REGIME_STOP_MULTIPLIER` | `{                         # STATUS: ACTIVE — backtest/engine.py; regime-conditional stop loss multipliers
    0: 1.0,...` | `ACTIVE` | backtest/engine.py; regime-conditional stop loss multipliers | 383 |
| `MAX_ANNUALIZED_TURNOVER` | `500.0` | `ACTIVE` | backtest/engine.py; 500% annualized turnover warning threshold | 389 |
| `MAX_SECTOR_EXPOSURE` | `0.10` | `ACTIVE` | risk/portfolio_optimizer.py; but INACTIVE when GICS_SECTORS is empty | 393 |
| `GICS_SECTORS` | `{}` | `PLACEHOLDER` | empty; sector constraints not enforced | 405 |

## Almgren-Chriss Optimal Execution

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `ALMGREN_CHRISS_ENABLED` | `False` | `PLACEHOLDER` | backtest/engine.py gates on this but not fully integrated | 412 |
| `ALMGREN_CHRISS_ADV_THRESHOLD` | `0.05` | `ACTIVE` | backtest/engine.py; positions > 5% of ADV use AC cost model | 413 |

## Validation

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `CPCV_PARTITIONS` | `8` | `ACTIVE` | backtest/validation.py; combinatorial purged CV partitions | 416 |
| `CPCV_TEST_PARTITIONS` | `4` | `ACTIVE` | backtest/validation.py; CPCV test partitions | 417 |
| `SPA_BOOTSTRAPS` | `400` | `ACTIVE` | backtest/validation.py; SPA bootstrap trials | 418 |
| `IC_ROLLING_WINDOW` | `60` | `ACTIVE` | backtest/validation.py; rolling window for Information Coefficient | 582 |

## Data Quality

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `DATA_QUALITY_ENABLED` | `True` | `ACTIVE` | data/loader.py; OHLCV quality checks | 421 |
| `MAX_MISSING_BAR_FRACTION` | `0.05` | `ACTIVE` | data/quality.py; max fraction of missing bars | 422 |
| `MAX_ZERO_VOLUME_FRACTION` | `0.25` | `ACTIVE` | data/quality.py; max fraction of zero-volume bars | 423 |
| `MAX_ABS_DAILY_RETURN` | `0.40` | `ACTIVE` | data/quality.py; max absolute single-day return | 424 |

## Autopilot (discovery -> promotion -> paper trading)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `AUTOPILOT_DIR` | `RESULTS_DIR / "autopilot"` | `ACTIVE` | base path for autopilot output (used via derived paths below) | 427 |
| `STRATEGY_REGISTRY_PATH` | `AUTOPILOT_DIR / "strategy_registry.json"` | `ACTIVE` | autopilot/registry.py | 428 |
| `PAPER_STATE_PATH` | `AUTOPILOT_DIR / "paper_state.json"` | `ACTIVE` | autopilot/paper_trader.py | 429 |
| `AUTOPILOT_CYCLE_REPORT` | `AUTOPILOT_DIR / "latest_cycle.json"` | `ACTIVE` | autopilot/engine.py | 430 |
| `AUTOPILOT_FEATURE_MODE` | `"core"` | `ACTIVE` | autopilot/engine.py; "core" (reduced) or "full" | 431 |
| `DISCOVERY_ENTRY_MULTIPLIERS` | `[0.8, 1.0, 1.2]` | `ACTIVE` | autopilot/strategy_discovery.py | 433 |
| `DISCOVERY_CONFIDENCE_OFFSETS` | `[-0.10, 0.0, 0.10]` | `ACTIVE` | autopilot/strategy_discovery.py | 434 |
| `DISCOVERY_RISK_VARIANTS` | `[False, True]` | `ACTIVE` | autopilot/strategy_discovery.py | 435 |
| `DISCOVERY_MAX_POSITIONS_VARIANTS` | `[10, 20]` | `ACTIVE` | autopilot/strategy_discovery.py | 436 |
| `PROMOTION_MIN_TRADES` | `80` | `ACTIVE` | autopilot/promotion_gate.py | 438 |
| `PROMOTION_MIN_WIN_RATE` | `0.50` | `ACTIVE` | autopilot/promotion_gate.py | 439 |
| `PROMOTION_MIN_SHARPE` | `0.75` | `ACTIVE` | autopilot/promotion_gate.py | 440 |
| `PROMOTION_MIN_PROFIT_FACTOR` | `1.10` | `ACTIVE` | autopilot/promotion_gate.py | 441 |
| `PROMOTION_MAX_DRAWDOWN` | `-0.20` | `ACTIVE` | autopilot/promotion_gate.py | 442 |
| `PROMOTION_MIN_ANNUAL_RETURN` | `0.05` | `ACTIVE` | autopilot/promotion_gate.py | 443 |
| `PROMOTION_MAX_ACTIVE_STRATEGIES` | `5` | `ACTIVE` | autopilot/registry.py | 444 |
| `PROMOTION_REQUIRE_ADVANCED_CONTRACT` | `True` | `ACTIVE` | autopilot/promotion_gate.py | 445 |
| `PROMOTION_MAX_DSR_PVALUE` | `0.05` | `ACTIVE` | autopilot/promotion_gate.py | 446 |
| `PROMOTION_MAX_PBO` | `0.45` | `ACTIVE` | autopilot/promotion_gate.py; tightened from 0.50 (Bailey et al. 2017) | 447 |
| `PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED` | `True` | `ACTIVE` | autopilot/promotion_gate.py | 448 |
| `PROMOTION_MAX_CAPACITY_UTILIZATION` | `1.0` | `ACTIVE` | autopilot/promotion_gate.py | 449 |
| `PROMOTION_MIN_WF_OOS_CORR` | `0.01` | `ACTIVE` | autopilot/promotion_gate.py | 450 |
| `PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION` | `0.60` | `ACTIVE` | autopilot/promotion_gate.py | 451 |
| `PROMOTION_MAX_WF_IS_OOS_GAP` | `0.20` | `ACTIVE` | autopilot/promotion_gate.py | 452 |
| `PROMOTION_MIN_REGIME_POSITIVE_FRACTION` | `0.50` | `ACTIVE` | autopilot/promotion_gate.py | 453 |
| `PROMOTION_EVENT_MAX_WORST_EVENT_LOSS` | `-0.08` | `ACTIVE` | autopilot/promotion_gate.py | 454 |
| `PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE` | `0.50` | `ACTIVE` | autopilot/promotion_gate.py | 455 |
| `PROMOTION_EVENT_MIN_REGIME_STABILITY` | `0.40` | `ACTIVE` | autopilot/promotion_gate.py | 456 |
| `PROMOTION_REQUIRE_STATISTICAL_TESTS` | `True` | `ACTIVE` | autopilot/promotion_gate.py; require IC/FDR tests to pass | 457 |
| `PROMOTION_REQUIRE_CPCV` | `True` | `ACTIVE` | autopilot/promotion_gate.py; require CPCV to pass | 458 |
| `PROMOTION_REQUIRE_SPA` | `False` | `ACTIVE` | autopilot/promotion_gate.py; SPA is informational by default | 459 |

## Signal Selection — Spec 04

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `SIGNAL_TOPK_QUANTILE` | `0.70` | `ACTIVE` | autopilot/engine.py; select top 70% by cross-sectional z-score | 462 |
| `SIGNAL_Z_THRESHOLD` | `1.5` | `DEPRECATED` | superseded by SIGNAL_TOPK_QUANTILE; kept for backward compat | 463 |

## Meta-Labeling — Spec 04

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `META_LABELING_ENABLED` | `True` | `ACTIVE` | autopilot/engine.py, autopilot/meta_labeler.py | 466 |
| `META_LABELING_RETRAIN_FREQ_DAYS` | `7` | `ACTIVE` | autopilot/engine.py; weekly retraining schedule | 467 |
| `META_LABELING_FOLD_COUNT` | `5` | `ACTIVE` | autopilot/meta_labeler.py; CV folds for training | 468 |
| `META_LABELING_MIN_SAMPLES` | `500` | `ACTIVE` | autopilot/meta_labeler.py; minimum samples to train | 469 |
| `META_LABELING_CONFIDENCE_THRESHOLD` | `0.55` | `ACTIVE` | autopilot/engine.py; min confidence to pass filter | 470 |
| `META_LABELING_XGB_PARAMS` | `{                     # STATUS: ACTIVE — autopilot/meta_labeler.py; XGBoost hyperparameters
    "max_depth": 5,
    "...` | `ACTIVE` | autopilot/meta_labeler.py; XGBoost hyperparameters | 471 |

## Fold-Level Metrics — Spec 04

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `FOLD_CONSISTENCY_PENALTY_WEIGHT` | `0.15` | `ACTIVE` | autopilot/promotion_gate.py; weight in composite score | 480 |

## Kelly Sizing

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `KELLY_FRACTION` | `0.50` | `ACTIVE` | risk/position_sizer.py; half-Kelly (conservative default) | 483 |
| `MAX_PORTFOLIO_DD` | `0.20` | `ACTIVE` | risk/position_sizer.py; max portfolio drawdown for governor | 484 |
| `KELLY_PORTFOLIO_BLEND` | `0.30` | `PLACEHOLDER` | defined but never imported; Kelly weight in composite blend | 485 |
| `KELLY_BAYESIAN_ALPHA` | `2.0` | `ACTIVE` | risk/position_sizer.py; Beta prior alpha for win rate | 486 |
| `KELLY_BAYESIAN_BETA` | `2.0` | `ACTIVE` | risk/position_sizer.py; Beta prior beta for win rate | 487 |
| `KELLY_REGIME_CONDITIONAL` | `True` | `PLACEHOLDER` | defined but never imported; use regime-specific parameters | 488 |
| `KELLY_MIN_SAMPLES_FOR_UPDATE` | `10` | `ACTIVE` | risk/position_sizer.py; min trades before Bayesian posterior overrides prior | 489 |

## Risk Governor — Spec 05

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `SHOCK_BUDGET_PCT` | `0.05` | `ACTIVE` | risk/position_sizer.py; reserve 5% of capital (positions capped at 95%) | 493 |
| `CONCENTRATION_LIMIT_PCT` | `0.20` | `ACTIVE` | risk/position_sizer.py; max 20% in any single position | 496 |
| `TURNOVER_BUDGET_ENFORCEMENT` | `True` | `ACTIVE` | risk/position_sizer.py; enable turnover budget constraint | 499 |
| `TURNOVER_BUDGET_LOOKBACK_DAYS` | `252` | `ACTIVE` | risk/position_sizer.py; annualized turnover lookback | 500 |
| `BLEND_WEIGHTS_STATIC` | `{                          # STATUS: ACTIVE — risk/position_sizer.py; default blend weights
    'kelly': 0.30,
    'v...` | `ACTIVE` | risk/position_sizer.py; default blend weights | 503 |
| `BLEND_WEIGHTS_BY_REGIME` | `{                       # STATUS: ACTIVE — risk/position_sizer.py; regime-conditional blend weights
    'NORMAL': {'k...` | `ACTIVE` | risk/position_sizer.py; regime-conditional blend weights | 510 |
| `UNCERTAINTY_SCALING_ENABLED` | `True` | `ACTIVE` | risk/position_sizer.py; enable uncertainty-based size reduction | 519 |
| `UNCERTAINTY_SIGNAL_WEIGHT` | `0.40` | `ACTIVE` | risk/position_sizer.py; weight of signal_uncertainty in composite | 520 |
| `UNCERTAINTY_REGIME_WEIGHT` | `0.30` | `ACTIVE` | risk/position_sizer.py; weight of regime_entropy | 521 |
| `UNCERTAINTY_DRIFT_WEIGHT` | `0.30` | `ACTIVE` | risk/position_sizer.py; weight of drift_score (inverted) | 522 |
| `UNCERTAINTY_REDUCTION_FACTOR` | `0.30` | `ACTIVE` | risk/position_sizer.py; max reduction from base size (30%) | 523 |
| `PAPER_INITIAL_CAPITAL` | `1_000_000.0` | `ACTIVE` | autopilot/paper_trader.py | 525 |
| `PAPER_MAX_TOTAL_POSITIONS` | `30` | `ACTIVE` | autopilot/paper_trader.py | 526 |
| `PAPER_USE_KELLY_SIZING` | `True` | `ACTIVE` | autopilot/paper_trader.py, api/services/backtest_service.py | 527 |
| `PAPER_KELLY_FRACTION` | `0.50` | `ACTIVE` | autopilot/paper_trader.py | 528 |
| `PAPER_KELLY_LOOKBACK_TRADES` | `200` | `ACTIVE` | autopilot/paper_trader.py | 529 |
| `PAPER_KELLY_MIN_SIZE_MULTIPLIER` | `0.25` | `ACTIVE` | autopilot/paper_trader.py | 530 |
| `PAPER_KELLY_MAX_SIZE_MULTIPLIER` | `1.50` | `ACTIVE` | autopilot/paper_trader.py | 531 |

## Feature Profiles

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `FEATURE_MODE_DEFAULT` | `"core"` | `ACTIVE` | run_backtest.py, run_train.py, run_predict.py; "full" or "core" | 534 |

## Regime Trade Gating (SPEC-E02)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `REGIME_TRADE_POLICY` | `{0: {enabled: True, ...}, 1: ..., 2: ..., 3: ...}` | `ACTIVE` | backtest/engine.py, api/routers/signals.py; per-regime trade gating with configurable confidence thresholds | 550 |
| `REGIME_2_TRADE_ENABLED` | `False` | `DEPRECATED` | Derived from REGIME_TRADE_POLICY[2]; kept for backward-compat | 561 |
| `REGIME_2_SUPPRESSION_MIN_CONFIDENCE` | `0.70` | `DEPRECATED` | Derived from REGIME_TRADE_POLICY[2]; kept for backward-compat | 562 |

## Regime Strategy Allocation

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `REGIME_STRATEGY_ALLOCATION_ENABLED` | `True` | `ACTIVE` | autopilot/strategy_allocator.py; adapt parameters by regime | 541 |

## Drawdown Tiers

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `DRAWDOWN_WARNING_THRESHOLD` | `-0.05` | `ACTIVE` | risk/stop_loss.py; -5% drawdown: reduce sizing 50% | 544 |
| `DRAWDOWN_CAUTION_THRESHOLD` | `-0.10` | `ACTIVE` | risk/stop_loss.py; -10% drawdown: no new entries, 25% sizing | 545 |
| `DRAWDOWN_CRITICAL_THRESHOLD` | `-0.15` | `ACTIVE` | risk/stop_loss.py; -15% drawdown: force liquidate | 546 |
| `DRAWDOWN_DAILY_LOSS_LIMIT` | `-0.03` | `ACTIVE` | risk/stop_loss.py; -3% daily loss limit | 547 |
| `DRAWDOWN_WEEKLY_LOSS_LIMIT` | `-0.05` | `ACTIVE` | risk/stop_loss.py; -5% weekly loss limit | 548 |
| `DRAWDOWN_RECOVERY_DAYS` | `10` | `ACTIVE` | risk/stop_loss.py; days to return to full sizing after recovery | 549 |
| `DRAWDOWN_SIZE_MULT_WARNING` | `0.50` | `ACTIVE` | risk/stop_loss.py; size multiplier during WARNING tier | 550 |
| `DRAWDOWN_SIZE_MULT_CAUTION` | `0.25` | `ACTIVE` | risk/stop_loss.py; size multiplier during CAUTION tier | 551 |

## Covariance

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `COVARIANCE_HALF_LIFE` | `60` | `ACTIVE` | risk/covariance.py; EWMA half-life in trading days (60 ≈ 3 months) | 554 |

## Stop Loss

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `STOP_LOSS_SPREAD_BUFFER_BPS` | `3.0` | `ACTIVE` | risk/stop_loss.py; bid-ask spread buffer for stop prices (bps) | 557 |
| `HARD_STOP_PCT` | `-0.08` | `ACTIVE` | risk/stop_loss.py; -8% hard stop loss | 558 |
| `ATR_STOP_MULTIPLIER` | `2.0` | `ACTIVE` | risk/stop_loss.py; initial stop at 2x ATR | 559 |
| `TRAILING_ATR_MULTIPLIER` | `1.5` | `ACTIVE` | risk/stop_loss.py; trailing stop at 1.5x ATR | 560 |
| `TRAILING_ACTIVATION_PCT` | `0.02` | `ACTIVE` | risk/stop_loss.py; activate trailing stop after +2% gain | 561 |
| `MAX_HOLDING_DAYS` | `30` | `ACTIVE` | risk/stop_loss.py, backtest/engine.py; time-based stop at 30 days | 562 |

## Almgren-Chriss Parameters

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `ALMGREN_CHRISS_FALLBACK_VOL` | `0.20` | `ACTIVE` | backtest/engine.py; fallback annualized vol when realized unavailable | 565 |
| `ALMGREN_CHRISS_RISK_AVERSION` | `0.01` | `ACTIVE` | backtest/optimal_execution.py; risk aversion lambda | 572 |

## Model Governance

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `GOVERNANCE_SCORE_WEIGHTS` | `{                       # STATUS: ACTIVE — models/governance.py; champion/challenger scoring weights
    "oos_spearma...` | `ACTIVE` | models/governance.py; champion/challenger scoring weights | 575 |

## Evaluation Layer (Spec 08)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `EVAL_WF_TRAIN_WINDOW` | `250` | `ACTIVE` | evaluation/engine.py; training window in trading days | 586 |
| `EVAL_WF_EMBARGO_DAYS` | `5` | `ACTIVE` | evaluation/engine.py; embargo gap to prevent data leakage | 587 |
| `EVAL_WF_TEST_WINDOW` | `60` | `ACTIVE` | evaluation/engine.py; test window in trading days | 588 |
| `EVAL_WF_SLIDE_FREQ` | `"weekly"` | `ACTIVE` | evaluation/engine.py; "weekly" or "daily" | 589 |
| `EVAL_IC_ROLLING_WINDOW` | `60` | `ACTIVE` | evaluation/engine.py; rolling IC window | 592 |
| `EVAL_IC_DECAY_THRESHOLD` | `0.02` | `ACTIVE` | evaluation/engine.py; warn if IC falls below this | 593 |
| `EVAL_IC_DECAY_LOOKBACK` | `20` | `ACTIVE` | evaluation/engine.py; days to check for sustained low IC | 594 |
| `EVAL_DECILE_SPREAD_MIN` | `0.005` | `ACTIVE` | evaluation/metrics.py; minimum expected spread for a good predictor | 597 |
| `EVAL_CALIBRATION_BINS` | `10` | `ACTIVE` | evaluation/calibration_analysis.py; number of bins for calibration curve | 600 |
| `EVAL_OVERCONFIDENCE_THRESHOLD` | `0.2` | `ACTIVE` | evaluation/calibration_analysis.py; max gap before flagging | 601 |
| `EVAL_TOP_N_TRADES` | `[5, 10, 20]` | `ACTIVE` | evaluation/fragility.py; top N for PnL concentration | 604 |
| `EVAL_RECOVERY_WINDOW` | `60` | `ACTIVE` | evaluation/fragility.py; rolling window for recovery time | 605 |
| `EVAL_CRITICAL_SLOWING_WINDOW` | `60` | `ACTIVE` | evaluation/fragility.py; window for trend detection | 606 |
| `EVAL_CRITICAL_SLOWING_SLOPE_THRESHOLD` | `0.05` | `ACTIVE` | evaluation/fragility.py; slope above this = danger | 607 |
| `EVAL_FEATURE_DRIFT_THRESHOLD` | `0.7` | `ACTIVE` | evaluation/ml_diagnostics.py; correlation below this = drift | 610 |
| `EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD` | `0.5` | `ACTIVE` | evaluation/ml_diagnostics.py; correlation below this = disagreement | 611 |
| `EVAL_MIN_SLICE_SAMPLES` | `20` | `ACTIVE` | evaluation/slicing.py; flag slices with fewer observations | 614 |
| `EVAL_REGIME_SHARPE_DIVERGENCE` | `0.5` | `ACTIVE` | evaluation/engine.py; regime Sharpes differ > this = red flag | 617 |
| `EVAL_OVERFIT_GAP_THRESHOLD` | `0.10` | `ACTIVE` | evaluation/engine.py; IS-OOS gap > this = overfitting | 618 |
| `EVAL_PNL_CONCENTRATION_THRESHOLD` | `0.70` | `ACTIVE` | evaluation/engine.py; top-20 PnL > this = fragile | 619 |
| `EVAL_CALIBRATION_ERROR_THRESHOLD` | `0.15` | `ACTIVE` | evaluation/engine.py; calibration error > this = red flag | 620 |

## Alert System

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `ALERT_WEBHOOK_URL` | `""` | `ACTIVE` | utils/logging.py; empty = disabled; set to Slack/Discord webhook URL | 623 |
| `ALERT_HISTORY_FILE` | `RESULTS_DIR / "alerts_history.json"` | `ACTIVE` | utils/logging.py; alert history persistence | 624 |

## Log Configuration

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `LOG_LEVEL` | `"INFO"` | `ACTIVE` | api/main.py; "DEBUG", "INFO", "WARNING", "ERROR" | 627 |
| `LOG_FORMAT` | `"structured"` | `ACTIVE` | api/main.py; "structured" or "json" | 628 |

## Structural Features (Spec 02)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---|
| `STRUCTURAL_FEATURES_ENABLED` | `True` | `ACTIVE` | features/pipeline.py; master toggle for spectral/SSA/tail/eigen/OT | 634 |
| `SPECTRAL_FFT_WINDOW` | `252` | `ACTIVE` | indicators/spectral.py; rolling lookback for FFT (~1 year) | 637 |
| `SPECTRAL_CUTOFF_PERIOD` | `20` | `ACTIVE` | indicators/spectral.py; HF/LF boundary in trading days | 638 |
| `SSA_WINDOW` | `60` | `ACTIVE` | indicators/ssa.py; rolling SSA lookback | 641 |
| `SSA_EMBED_DIM` | `12` | `ACTIVE` | indicators/ssa.py; embedding dimension (< SSA_WINDOW) | 642 |
| `SSA_N_SINGULAR` | `5` | `ACTIVE` | indicators/ssa.py; signal components (rest = noise) | 643 |
| `JUMP_INTENSITY_WINDOW` | `20` | `ACTIVE` | indicators/tail_risk.py; lookback for jump detection | 646 |
| `JUMP_INTENSITY_THRESHOLD` | `2.5` | `ACTIVE` | indicators/tail_risk.py; sigma threshold for jumps | 647 |
| `EIGEN_CONCENTRATION_WINDOW` | `60` | `ACTIVE` | indicators/eigenvalue.py; rolling correlation window | 650 |
| `EIGEN_MIN_ASSETS` | `5` | `ACTIVE` | indicators/eigenvalue.py; min assets for eigenvalue features | 651 |
| `EIGEN_REGULARIZATION` | `0.01` | `ACTIVE` | indicators/eigenvalue.py; Tikhonov regularization | 652 |
| `WASSERSTEIN_WINDOW` | `30` | `ACTIVE` | indicators/ot_divergence.py; current distribution window | 655 |
| `WASSERSTEIN_REF_WINDOW` | `60` | `ACTIVE` | indicators/ot_divergence.py; reference distribution window | 656 |
| `SINKHORN_EPSILON` | `0.01` | `ACTIVE` | indicators/ot_divergence.py; entropic regularization | 657 |
| `SINKHORN_MAX_ITER` | `100` | `ACTIVE` | indicators/ot_divergence.py; max Sinkhorn iterations | 658 |

## `api/config.py` (API Runtime Settings)

- LOC: 76
- Classes: `ApiSettings`, `RuntimeConfig`
- Top-level functions: none

## `config_structured.py` (Typed Config Dataclasses)

- LOC: 301
- Exported classes/enums: `ReturnType`, `PriceType`, `EntryType`, `PreconditionsConfig`, `CostStressConfig`, `DataConfig`, `RegimeConfig`, `ModelConfig`, `BacktestConfig`, `KellyConfig`, `DrawdownConfig`, `StopLossConfig`, `ValidationConfig`, `PromotionConfig`, `HealthConfig`, `PaperTradingConfig`, `ExecutionConfig`, `SystemConfig`
