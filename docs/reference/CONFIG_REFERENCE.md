# Configuration Reference

Source-derived configuration reference for `config.py`, `api/config.py`, and `config_structured.py`.

## `config.py` (Flat Runtime Configuration)

Notes:
- `config.py` contains explicit `STATUS:` annotations (`ACTIVE`, `PLACEHOLDER`, `DEPRECATED`) in inline comments.
- Values below are shown as source expressions, not evaluated runtime values.

## Paths

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `ROOT_DIR` | `Path(__file__).parent` | `ACTIVE` | base path for all relative references | 26 |
| `FRAMEWORK_DIR` | `ROOT_DIR.parent` | `ACTIVE` | parent directory of quant_engine | 27 |
| `MODEL_DIR` | `ROOT_DIR / 'trained_models'` | `ACTIVE` | models/versioning.py, models/trainer.py | 28 |
| `RESULTS_DIR` | `ROOT_DIR / 'results'` | `ACTIVE` | backtest output, autopilot reports, alerts | 29 |

## Data Sources

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `DATA_CACHE_DIR` | `ROOT_DIR / 'data' / 'cache'` | `ACTIVE` | data/loader.py, data/local_cache.py | 32 |
| `WRDS_ENABLED` | `True` | `ACTIVE` | data/loader.py; try WRDS first, fall back to local cache / IBKR | 33 |
| `OPTIONMETRICS_ENABLED` | `False` | `PLACEHOLDER` | data/loader.py gates on this but pipeline incomplete | 41 |
| `KALSHI_ENABLED` | `False` | `ACTIVE` | kalshi/provider.py, run_kalshi_event_pipeline.py; disabled by design | 43 |
| `KALSHI_ENV` | `'demo'` | `ACTIVE` | selects demo vs prod API URL; "demo" (safety) or "prod" | 44 |
| `KALSHI_DEMO_API_BASE_URL` | `'https://demo-api.kalshi.co/trade-api/v2'` | `ACTIVE` | used to compute KALSHI_API_BASE_URL | 45 |
| `KALSHI_PROD_API_BASE_URL` | `'https://api.elections.kalshi.com/trade-api/v2'` | `ACTIVE` | used when KALSHI_ENV="prod" | 46 |
| `KALSHI_API_BASE_URL` | `KALSHI_DEMO_API_BASE_URL` | `ACTIVE` | kalshi/provider.py base URL | 47 |
| `KALSHI_HISTORICAL_API_BASE_URL` | `KALSHI_API_BASE_URL` | `ACTIVE` | kalshi historical data endpoint | 48 |
| `KALSHI_HISTORICAL_CUTOFF_TS` | `None` | `ACTIVE` | optional cutoff for historical fetches | 49 |
| `KALSHI_RATE_LIMIT_RPS` | `6.0` | `ACTIVE` | kalshi/provider.py rate limiter | 50 |
| `KALSHI_RATE_LIMIT_BURST` | `2` | `ACTIVE` | kalshi/provider.py burst allowance | 51 |
| `KALSHI_DB_PATH` | `ROOT_DIR / 'data' / 'kalshi.duckdb'` | `ACTIVE` | DuckDB path for Kalshi snapshots | 52 |
| `KALSHI_SNAPSHOT_HORIZONS` | `['7d', '1d', '4h', '1h', '15m', '5m']` | `ACTIVE` | kalshi/provider.py, run_kalshi_event_pipeline.py | 53 |
| `KALSHI_DISTRIBUTION_FREQ` | `'5min'` | `ACTIVE` | kalshi/provider.py distribution resampling | 54 |
| `KALSHI_STALE_AFTER_MINUTES` | `30` | `ACTIVE` | kalshi/provider.py staleness detection | 55 |
| `KALSHI_NEAR_EVENT_MINUTES` | `30.0` | `ACTIVE` | kalshi/provider.py near-event window | 56 |
| `KALSHI_NEAR_EVENT_STALE_MINUTES` | `2.0` | `ACTIVE` | tighter staleness near events | 57 |
| `KALSHI_FAR_EVENT_MINUTES` | `24.0 * 60.0` | `ACTIVE` | far-event window (24h) | 58 |
| `KALSHI_FAR_EVENT_STALE_MINUTES` | `60.0` | `ACTIVE` | relaxed staleness far from events | 59 |
| `KALSHI_STALE_MARKET_TYPE_MULTIPLIERS` | `{'CPI': 0.8, 'UNEMPLOYMENT': 0.9, 'FOMC': 0.7, '_default': 1.0}` | `ACTIVE` | kalshi/provider.py per-event-type staleness | 60 |
| `KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD` | `2.0` | `ACTIVE` | kalshi/provider.py low-liquidity threshold | 66 |
| `KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD` | `6.0` | `ACTIVE` | kalshi/provider.py high-liquidity threshold | 67 |
| `KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER` | `1.35` | `ACTIVE` | tighten staleness for illiquid markets | 68 |
| `KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER` | `0.8` | `ACTIVE` | relax staleness for liquid markets | 69 |
| `KALSHI_DISTANCE_LAGS` | `['1h', '1d']` | `ACTIVE` | kalshi/provider.py distance lag features | 70 |
| `KALSHI_TAIL_THRESHOLDS` | `{'CPI': [3.0, 3.5, 4.0], 'UNEMPLOYMENT': [4.0, 4.2, 4.5], 'FOMC': [0.0, 25.0, 50.0], '_default': [0.0, 0.5, 1.0]}` | `ACTIVE` | kalshi/provider.py tail-risk thresholds | 71 |
| `DEFAULT_UNIVERSE_SOURCE` | `'wrds'` | `PLACEHOLDER` | defined but never imported; "wrds", "static", or "ibkr" | 77 |
| `CACHE_TRUSTED_SOURCES` | `['wrds', 'wrds_delisting', 'ibkr']` | `ACTIVE` | data/local_cache.py source ranking | 78 |
| `CACHE_MAX_STALENESS_DAYS` | `21` | `ACTIVE` | data/local_cache.py max cache age | 79 |
| `CACHE_WRDS_SPAN_ADVANTAGE_DAYS` | `180` | `ACTIVE` | data/local_cache.py WRDS preference window | 80 |
| `REQUIRE_PERMNO` | `True` | `ACTIVE` | data/loader.py, backtest/engine.py PERMNO validation | 81 |

## Survivorship

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `SURVIVORSHIP_DB` | `ROOT_DIR / 'data' / 'universe_history.db'` | `ACTIVE` | data/survivorship.py, data/loader.py | 84 |
| `SURVIVORSHIP_UNIVERSE_NAME` | `'SP500'` | `ACTIVE` | data/loader.py, autopilot/engine.py | 85 |
| `SURVIVORSHIP_SNAPSHOT_FREQ` | `'quarterly'` | `ACTIVE` | data/loader.py; "annual" or "quarterly" | 86 |

## Model Versioning

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `MODEL_REGISTRY` | `MODEL_DIR / 'registry.json'` | `PLACEHOLDER` | defined but never imported; CHAMPION_REGISTRY is used instead | 89 |
| `MAX_MODEL_VERSIONS` | `5` | `ACTIVE` | models/versioning.py; keep last 5 versions for rollback | 90 |
| `CHAMPION_REGISTRY` | `MODEL_DIR / 'champion_registry.json'` | `ACTIVE` | models/governance.py | 91 |

## Retraining

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `RETRAIN_MAX_DAYS` | `30` | `ACTIVE` | schedule-based retrain trigger (30 calendar days) | 94 |
| `RETRAIN_MIN_TRADES` | `50` | `PLACEHOLDER` | defined but never imported by running code | 95 |
| `RETRAIN_MIN_WIN_RATE` | `0.45` | `PLACEHOLDER` | defined but never imported by running code | 96 |
| `RETRAIN_MIN_CORRELATION` | `0.05` | `PLACEHOLDER` | minimum OOS Spearman; defined but never imported | 97 |
| `RETRAIN_REGIME_CHANGE_DAYS` | `10` | `ACTIVE` | run_retrain.py; trigger retrain if regime changed for 10+ consecutive days | 98 |
| `RECENCY_DECAY` | `0.003` | `ACTIVE` | models/trainer.py; exponential recency weighting (1yr weight ~ 0.33) | 99 |

## Universe

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `UNIVERSE_FULL` | `['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'DDOG', 'NET', 'CRWD', 'ZS', 'SNOW', 'MDB', 'PANW', 'FTNT', 'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', '...` | `ACTIVE` | data/loader.py, run_*.py, autopilot/engine.py | 102 |
| `UNIVERSE_QUICK` | `['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'TSLA', 'JPM', 'UNH', 'HD', 'V', 'DDOG', 'CRWD', 'CAVA']` | `ACTIVE` | run_*.py quick-test universe | 119 |
| `BENCHMARK` | `'SPY'` | `ACTIVE` | backtest/engine.py, api/routers/benchmark.py | 124 |

## Data

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `LOOKBACK_YEARS` | `15` | `ACTIVE` | data/loader.py; years of historical data to load | 127 |
| `MIN_BARS` | `500` | `ACTIVE` | data/loader.py; minimum bars needed for feature warm-up | 128 |

## Intraday Data

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `INTRADAY_TIMEFRAMES` | `['4h', '1h', '30m', '15m', '5m', '1m']` | `PLACEHOLDER` | defined but never imported | 131 |
| `INTRADAY_CACHE_SOURCE` | `'ibkr'` | `PLACEHOLDER` | defined but never imported | 132 |
| `INTRADAY_MIN_BARS` | `100` | `ACTIVE` | features/pipeline.py; minimum intraday bars for feature computation | 133 |
| `MARKET_OPEN` | `'09:30'` | `ACTIVE` | features/intraday.py; US equity regular-session open (ET) | 134 |
| `MARKET_CLOSE` | `'16:00'` | `ACTIVE` | features/intraday.py; US equity regular-session close (ET) | 135 |

## Targets

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `FORWARD_HORIZONS` | `[5, 10, 20]` | `ACTIVE` | models/trainer.py, run_*.py; days ahead to predict | 138 |

## Features

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `INTERACTION_PAIRS` | `[('RSI_14', 'Hurst_100', 'multiply'), ('ZScore_20', 'VolTS_10_60', 'multiply'), ('MACD_12_26', 'AutoCorr_20_1', 'multiply'), ('RSI_14', 'VarRatio_100_5', 'multiply'), ('NATR_14'...` | `ACTIVE` | features/pipeline.py; regime-conditional interaction features | 141 |

## Regime

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `REGIME_NAMES` | `{0: 'trending_bull', 1: 'trending_bear', 2: 'mean_reverting', 3: 'high_volatility'}` | `ACTIVE` | used in 15+ files; canonical regime label mapping | 164 |
| `MIN_REGIME_SAMPLES` | `500` | `ACTIVE` | models/trainer.py; minimum training samples per regime model | 170 |
| `REGIME_MODEL_TYPE` | `'hmm'` | `ACTIVE` | regime/detector.py; "hmm" or "rule" | 171 |
| `REGIME_HMM_STATES` | `4` | `ACTIVE` | regime/hmm.py; number of hidden states | 172 |
| `REGIME_HMM_MAX_ITER` | `60` | `ACTIVE` | regime/hmm.py; EM iteration limit | 173 |
| `REGIME_HMM_STICKINESS` | `0.92` | `ACTIVE` | regime/hmm.py; diagonal prior bias for sticky transitions | 174 |
| `REGIME_MIN_DURATION` | `3` | `ACTIVE` | regime/detector.py; minimum regime duration in days | 175 |
| `REGIME_SOFT_ASSIGNMENT_THRESHOLD` | `0.35` | `ACTIVE` | models/trainer.py; probability threshold for soft regime labels | 176 |
| `REGIME_HMM_PRIOR_WEIGHT` | `0.3` | `ACTIVE` | regime/hmm.py; shrinkage weight toward sticky prior in M-step | 177 |
| `REGIME_HMM_COVARIANCE_TYPE` | `'full'` | `ACTIVE` | regime/hmm.py; "full" (captures return-vol correlation) or "diag" | 178 |
| `REGIME_HMM_AUTO_SELECT_STATES` | `True` | `ACTIVE` | regime/hmm.py; use BIC to select optimal number of states | 179 |
| `REGIME_HMM_MIN_STATES` | `2` | `ACTIVE` | regime/hmm.py; BIC state search lower bound | 180 |
| `REGIME_HMM_MAX_STATES` | `6` | `ACTIVE` | regime/hmm.py; BIC state search upper bound | 181 |
| `REGIME_JUMP_MODEL_ENABLED` | `True` | `ACTIVE` | regime/detector.py; statistical jump model alongside HMM | 182 |
| `REGIME_JUMP_PENALTY` | `0.02` | `ACTIVE` | regime/jump_model.py; jump penalty lambda (higher = fewer transitions) | 183 |
| `REGIME_EXPECTED_CHANGES_PER_YEAR` | `4` | `ACTIVE` | regime/jump_model.py; calibrate jump penalty from expected regime changes/yr | 184 |
| `REGIME_ENSEMBLE_ENABLED` | `True` | `ACTIVE` | regime/detector.py; combine HMM + JM + rule-based via majority vote | 185 |
| `REGIME_ENSEMBLE_CONSENSUS_THRESHOLD` | `2` | `ACTIVE` | regime/detector.py; require N of 3 methods to agree for transition | 186 |

## Kalshi Purge/Embargo by Event Type (E3)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `KALSHI_PURGE_WINDOW_BY_EVENT` | `{'CPI': 14, 'FOMC': 21, 'NFP': 14, 'GDP': 14}` | `PLACEHOLDER` | defined but never imported | 189 |
| `KALSHI_DEFAULT_PURGE_WINDOW` | `10` | `PLACEHOLDER` | defined but never imported; companion to KALSHI_PURGE_WINDOW_BY_EVENT | 190 |

## Model

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `MODEL_PARAMS` | `{'n_estimators': 500, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05, 'subsample': 0.8, 'max_features': 'sqrt'}` | `ACTIVE` | models/trainer.py; GBR hyperparameters | 193 |
| `MAX_FEATURES_SELECTED` | `30` | `ACTIVE` | models/trainer.py; after permutation importance | 201 |
| `MAX_IS_OOS_GAP` | `0.05` | `ACTIVE` | models/trainer.py; max allowed IS-OOS degradation (R^2 or correlation) | 202 |
| `CV_FOLDS` | `5` | `ACTIVE` | models/trainer.py; cross-validation folds | 203 |
| `HOLDOUT_FRACTION` | `0.15` | `ACTIVE` | models/trainer.py; holdout set fraction | 204 |
| `ENSEMBLE_DIVERSIFY` | `True` | `ACTIVE` | models/trainer.py; train GBR + ElasticNet + RandomForest and average | 205 |

## Walk-Forward Rolling Window

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `WF_MAX_TRAIN_DATES` | `1260` | `ACTIVE` | backtest/engine.py, models/trainer.py; rolling walk-forward window | 211 |

## Backtest

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `TRANSACTION_COST_BPS` | `20` | `ACTIVE` | backtest/engine.py; 20 bps round-trip | 214 |
| `ENTRY_THRESHOLD` | `0.005` | `ACTIVE` | backtest/engine.py; minimum predicted return to enter (0.5%) | 215 |
| `CONFIDENCE_THRESHOLD` | `0.6` | `ACTIVE` | backtest/engine.py; minimum model confidence | 216 |
| `MAX_POSITIONS` | `20` | `ACTIVE` | backtest/engine.py; max simultaneous positions | 217 |
| `POSITION_SIZE_PCT` | `0.05` | `ACTIVE` | backtest/engine.py; 5% of capital per position | 218 |
| `BACKTEST_ASSUMED_CAPITAL_USD` | `1000000.0` | `ACTIVE` | backtest/engine.py; initial capital | 219 |
| `EXEC_SPREAD_BPS` | `3.0` | `ACTIVE` | backtest/engine.py; base spread cost | 220 |
| `EXEC_MAX_PARTICIPATION` | `0.02` | `ACTIVE` | backtest/engine.py; max 2% of daily volume | 221 |
| `EXEC_IMPACT_COEFF_BPS` | `25.0` | `ACTIVE` | backtest/engine.py; market impact coefficient | 222 |
| `EXEC_MIN_FILL_RATIO` | `0.2` | `ACTIVE` | backtest/engine.py; minimum fill ratio | 223 |
| `EXEC_DYNAMIC_COSTS` | `True` | `ACTIVE` | backtest/engine.py; condition costs on market state | 224 |
| `EXEC_DOLLAR_VOLUME_REF_USD` | `25000000.0` | `ACTIVE` | backtest/engine.py; dollar volume reference | 225 |
| `EXEC_VOL_REF` | `0.2` | `ACTIVE` | backtest/engine.py; reference volatility | 226 |
| `EXEC_VOL_SPREAD_BETA` | `1.0` | `ACTIVE` | backtest/engine.py; vol-spread sensitivity | 227 |
| `EXEC_GAP_SPREAD_BETA` | `4.0` | `ACTIVE` | backtest/engine.py; gap-spread sensitivity | 228 |
| `EXEC_RANGE_SPREAD_BETA` | `2.0` | `ACTIVE` | backtest/engine.py; range-spread sensitivity | 229 |
| `EXEC_VOL_IMPACT_BETA` | `1.0` | `ACTIVE` | backtest/engine.py; vol-impact sensitivity | 230 |
| `MAX_PORTFOLIO_VOL` | `0.3` | `ACTIVE` | backtest/engine.py, risk/portfolio_optimizer.py; max annualized vol | 231 |
| `REGIME_RISK_MULTIPLIER` | `{0: 1.0, 1: 0.85, 2: 0.95, 3: 0.6}` | `ACTIVE` | backtest/engine.py; regime-conditional position sizing multipliers | 232 |
| `REGIME_STOP_MULTIPLIER` | `{0: 1.0, 1: 0.8, 2: 1.2, 3: 1.5}` | `ACTIVE` | backtest/engine.py; regime-conditional stop loss multipliers | 238 |
| `MAX_ANNUALIZED_TURNOVER` | `500.0` | `ACTIVE` | backtest/engine.py; 500% annualized turnover warning threshold | 244 |
| `MAX_SECTOR_EXPOSURE` | `0.1` | `ACTIVE` | risk/portfolio_optimizer.py; but INACTIVE when GICS_SECTORS is empty | 248 |
| `GICS_SECTORS` | `{}` | `PLACEHOLDER` | empty; sector constraints not enforced | 260 |

## Almgren-Chriss Optimal Execution

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `ALMGREN_CHRISS_ENABLED` | `False` | `PLACEHOLDER` | backtest/engine.py gates on this but not fully integrated | 267 |
| `ALMGREN_CHRISS_ADV_THRESHOLD` | `0.05` | `ACTIVE` | backtest/engine.py; positions > 5% of ADV use AC cost model | 268 |

## Validation

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `CPCV_PARTITIONS` | `8` | `ACTIVE` | backtest/validation.py; combinatorial purged CV partitions | 271 |
| `CPCV_TEST_PARTITIONS` | `4` | `ACTIVE` | backtest/validation.py; CPCV test partitions | 272 |
| `SPA_BOOTSTRAPS` | `400` | `ACTIVE` | backtest/validation.py; SPA bootstrap trials | 273 |

## Data Quality

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `DATA_QUALITY_ENABLED` | `True` | `ACTIVE` | data/loader.py; OHLCV quality checks | 276 |
| `MAX_MISSING_BAR_FRACTION` | `0.05` | `ACTIVE` | data/quality.py; max fraction of missing bars | 277 |
| `MAX_ZERO_VOLUME_FRACTION` | `0.25` | `ACTIVE` | data/quality.py; max fraction of zero-volume bars | 278 |
| `MAX_ABS_DAILY_RETURN` | `0.4` | `ACTIVE` | data/quality.py; max absolute single-day return | 279 |

## Autopilot (discovery -> promotion -> paper trading)

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `AUTOPILOT_DIR` | `RESULTS_DIR / 'autopilot'` | `ACTIVE` | base path for autopilot output (used via derived paths below) | 282 |
| `STRATEGY_REGISTRY_PATH` | `AUTOPILOT_DIR / 'strategy_registry.json'` | `ACTIVE` | autopilot/registry.py | 283 |
| `PAPER_STATE_PATH` | `AUTOPILOT_DIR / 'paper_state.json'` | `ACTIVE` | autopilot/paper_trader.py | 284 |
| `AUTOPILOT_CYCLE_REPORT` | `AUTOPILOT_DIR / 'latest_cycle.json'` | `ACTIVE` | autopilot/engine.py | 285 |
| `AUTOPILOT_FEATURE_MODE` | `'core'` | `ACTIVE` | autopilot/engine.py; "core" (reduced) or "full" | 286 |
| `DISCOVERY_ENTRY_MULTIPLIERS` | `[0.8, 1.0, 1.2]` | `ACTIVE` | autopilot/strategy_discovery.py | 288 |
| `DISCOVERY_CONFIDENCE_OFFSETS` | `[-0.1, 0.0, 0.1]` | `ACTIVE` | autopilot/strategy_discovery.py | 289 |
| `DISCOVERY_RISK_VARIANTS` | `[False, True]` | `ACTIVE` | autopilot/strategy_discovery.py | 290 |
| `DISCOVERY_MAX_POSITIONS_VARIANTS` | `[10, 20]` | `ACTIVE` | autopilot/strategy_discovery.py | 291 |
| `PROMOTION_MIN_TRADES` | `80` | `ACTIVE` | autopilot/promotion_gate.py | 293 |
| `PROMOTION_MIN_WIN_RATE` | `0.5` | `ACTIVE` | autopilot/promotion_gate.py | 294 |
| `PROMOTION_MIN_SHARPE` | `0.75` | `ACTIVE` | autopilot/promotion_gate.py | 295 |
| `PROMOTION_MIN_PROFIT_FACTOR` | `1.1` | `ACTIVE` | autopilot/promotion_gate.py | 296 |
| `PROMOTION_MAX_DRAWDOWN` | `-0.2` | `ACTIVE` | autopilot/promotion_gate.py | 297 |
| `PROMOTION_MIN_ANNUAL_RETURN` | `0.05` | `ACTIVE` | autopilot/promotion_gate.py | 298 |
| `PROMOTION_MAX_ACTIVE_STRATEGIES` | `5` | `ACTIVE` | autopilot/registry.py | 299 |
| `PROMOTION_REQUIRE_ADVANCED_CONTRACT` | `True` | `ACTIVE` | autopilot/promotion_gate.py | 300 |
| `PROMOTION_MAX_DSR_PVALUE` | `0.05` | `ACTIVE` | autopilot/promotion_gate.py | 301 |
| `PROMOTION_MAX_PBO` | `0.45` | `ACTIVE` | autopilot/promotion_gate.py; tightened from 0.50 (Bailey et al. 2017) | 302 |
| `PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED` | `True` | `ACTIVE` | autopilot/promotion_gate.py | 303 |
| `PROMOTION_MAX_CAPACITY_UTILIZATION` | `1.0` | `ACTIVE` | autopilot/promotion_gate.py | 304 |
| `PROMOTION_MIN_WF_OOS_CORR` | `0.01` | `ACTIVE` | autopilot/promotion_gate.py | 305 |
| `PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION` | `0.6` | `ACTIVE` | autopilot/promotion_gate.py | 306 |
| `PROMOTION_MAX_WF_IS_OOS_GAP` | `0.2` | `ACTIVE` | autopilot/promotion_gate.py | 307 |
| `PROMOTION_MIN_REGIME_POSITIVE_FRACTION` | `0.5` | `ACTIVE` | autopilot/promotion_gate.py | 308 |
| `PROMOTION_EVENT_MAX_WORST_EVENT_LOSS` | `-0.08` | `ACTIVE` | autopilot/promotion_gate.py | 309 |
| `PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE` | `0.5` | `ACTIVE` | autopilot/promotion_gate.py | 310 |
| `PROMOTION_EVENT_MIN_REGIME_STABILITY` | `0.4` | `ACTIVE` | autopilot/promotion_gate.py | 311 |
| `PROMOTION_REQUIRE_STATISTICAL_TESTS` | `True` | `ACTIVE` | autopilot/promotion_gate.py; require IC/FDR tests to pass | 312 |
| `PROMOTION_REQUIRE_CPCV` | `True` | `ACTIVE` | autopilot/promotion_gate.py; require CPCV to pass | 313 |
| `PROMOTION_REQUIRE_SPA` | `False` | `ACTIVE` | autopilot/promotion_gate.py; SPA is informational by default | 314 |

## Kelly Sizing

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `KELLY_FRACTION` | `0.5` | `ACTIVE` | risk/position_sizer.py; half-Kelly (conservative default) | 317 |
| `MAX_PORTFOLIO_DD` | `0.2` | `PLACEHOLDER` | defined but never imported; max portfolio drawdown for governor | 318 |
| `KELLY_PORTFOLIO_BLEND` | `0.3` | `PLACEHOLDER` | defined but never imported; Kelly weight in composite blend | 319 |
| `KELLY_BAYESIAN_ALPHA` | `2.0` | `PLACEHOLDER` | defined but never imported; Beta prior alpha for win rate | 320 |
| `KELLY_BAYESIAN_BETA` | `2.0` | `PLACEHOLDER` | defined but never imported; Beta prior beta for win rate | 321 |
| `KELLY_REGIME_CONDITIONAL` | `True` | `PLACEHOLDER` | defined but never imported; use regime-specific parameters | 322 |
| `PAPER_INITIAL_CAPITAL` | `1000000.0` | `ACTIVE` | autopilot/paper_trader.py | 324 |
| `PAPER_MAX_TOTAL_POSITIONS` | `30` | `ACTIVE` | autopilot/paper_trader.py | 325 |
| `PAPER_USE_KELLY_SIZING` | `True` | `ACTIVE` | autopilot/paper_trader.py, api/services/backtest_service.py | 326 |
| `PAPER_KELLY_FRACTION` | `0.5` | `ACTIVE` | autopilot/paper_trader.py | 327 |
| `PAPER_KELLY_LOOKBACK_TRADES` | `200` | `ACTIVE` | autopilot/paper_trader.py | 328 |
| `PAPER_KELLY_MIN_SIZE_MULTIPLIER` | `0.25` | `ACTIVE` | autopilot/paper_trader.py | 329 |
| `PAPER_KELLY_MAX_SIZE_MULTIPLIER` | `1.5` | `ACTIVE` | autopilot/paper_trader.py | 330 |

## Feature Profiles

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `FEATURE_MODE_DEFAULT` | `'core'` | `ACTIVE` | run_backtest.py, run_train.py, run_predict.py; "full" or "core" | 333 |

## Regime Trade Gating

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `REGIME_2_TRADE_ENABLED` | `False` | `ACTIVE` | backtest/engine.py, api/routers/signals.py; suppress mean-revert regime entries (Sharpe -0.91) | 336 |
| `REGIME_2_SUPPRESSION_MIN_CONFIDENCE` | `0.5` | `ACTIVE` | backtest/engine.py; only suppress when confidence exceeds this | 337 |

## Drawdown Tiers

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `DRAWDOWN_WARNING_THRESHOLD` | `-0.05` | `ACTIVE` | risk/stop_loss.py; -5% drawdown: reduce sizing 50% | 340 |
| `DRAWDOWN_CAUTION_THRESHOLD` | `-0.1` | `ACTIVE` | risk/stop_loss.py; -10% drawdown: no new entries, 25% sizing | 341 |
| `DRAWDOWN_CRITICAL_THRESHOLD` | `-0.15` | `ACTIVE` | risk/stop_loss.py; -15% drawdown: force liquidate | 342 |
| `DRAWDOWN_DAILY_LOSS_LIMIT` | `-0.03` | `ACTIVE` | risk/stop_loss.py; -3% daily loss limit | 343 |
| `DRAWDOWN_WEEKLY_LOSS_LIMIT` | `-0.05` | `ACTIVE` | risk/stop_loss.py; -5% weekly loss limit | 344 |
| `DRAWDOWN_RECOVERY_DAYS` | `10` | `ACTIVE` | risk/stop_loss.py; days to return to full sizing after recovery | 345 |
| `DRAWDOWN_SIZE_MULT_WARNING` | `0.5` | `ACTIVE` | risk/stop_loss.py; size multiplier during WARNING tier | 346 |
| `DRAWDOWN_SIZE_MULT_CAUTION` | `0.25` | `ACTIVE` | risk/stop_loss.py; size multiplier during CAUTION tier | 347 |

## Stop Loss

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `HARD_STOP_PCT` | `-0.08` | `ACTIVE` | risk/stop_loss.py; -8% hard stop loss | 350 |
| `ATR_STOP_MULTIPLIER` | `2.0` | `ACTIVE` | risk/stop_loss.py; initial stop at 2x ATR | 351 |
| `TRAILING_ATR_MULTIPLIER` | `1.5` | `ACTIVE` | risk/stop_loss.py; trailing stop at 1.5x ATR | 352 |
| `TRAILING_ACTIVATION_PCT` | `0.02` | `ACTIVE` | risk/stop_loss.py; activate trailing stop after +2% gain | 353 |
| `MAX_HOLDING_DAYS` | `30` | `ACTIVE` | risk/stop_loss.py, backtest/engine.py; time-based stop at 30 days | 354 |

## Almgren-Chriss Parameters

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `ALMGREN_CHRISS_FALLBACK_VOL` | `0.2` | `ACTIVE` | backtest/engine.py; fallback annualized vol when realized unavailable | 357 |
| `ALMGREN_CHRISS_RISK_AVERSION` | `0.01` | `ACTIVE` | backtest/optimal_execution.py; risk aversion lambda | 364 |

## Model Governance

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `GOVERNANCE_SCORE_WEIGHTS` | `{'oos_spearman': 1.5, 'holdout_spearman': 1.0, 'cv_gap_penalty': -0.5}` | `ACTIVE` | models/governance.py; champion/challenger scoring weights | 367 |

## Validation

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `IC_ROLLING_WINDOW` | `60` | `ACTIVE` | backtest/validation.py; rolling window for Information Coefficient | 374 |

## Alert System

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `ALERT_WEBHOOK_URL` | `''` | `ACTIVE` | utils/logging.py; empty = disabled; set to Slack/Discord webhook URL | 377 |
| `ALERT_HISTORY_FILE` | `RESULTS_DIR / 'alerts_history.json'` | `ACTIVE` | utils/logging.py; alert history persistence | 378 |

## Log Configuration

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|
| `LOG_LEVEL` | `'INFO'` | `ACTIVE` | api/main.py; "DEBUG", "INFO", "WARNING", "ERROR" | 381 |
| `LOG_FORMAT` | `'structured'` | `ACTIVE` | api/main.py; "structured" or "json" | 382 |

## Config Validation

| Constant | Value (source) | Status | Notes | Line |
|---|---|---|---|---:|

## `api/config.py` (API Settings + Runtime Patching)

### `ApiSettings` (`QE_API_` environment prefix)

| Field | Default (source) | Line |
|---|---|---:|
| `host` | `'0.0.0.0'` | 30 |
| `port` | `8000` | 31 |
| `cors_origins` | `'*'` | 32 |
| `job_db_path` | `'api_jobs.db'` | 33 |
| `log_level` | `'INFO'` | 34 |

### Runtime-adjustable keys (`RuntimeConfig.patch`) 

- Count: 11
- Keys: `CONFIDENCE_THRESHOLD`, `DRAWDOWN_CAUTION_THRESHOLD`, `DRAWDOWN_CRITICAL_THRESHOLD`, `DRAWDOWN_DAILY_LOSS_LIMIT`, `DRAWDOWN_WARNING_THRESHOLD`, `DRAWDOWN_WEEKLY_LOSS_LIMIT`, `ENTRY_THRESHOLD`, `MAX_HOLDING_DAYS`, `MAX_POSITIONS`, `POSITION_SIZE_PCT`, `REGIME_2_TRADE_ENABLED`

### `RuntimeConfig` methods

- `__init__`, `get_adjustable`, `patch`

## `config_structured.py` (Typed Dataclass View)

`config_structured.py` mirrors portions of `config.py` as typed dataclasses for IDE/type-checking ergonomics. It is not the canonical runtime state for all modules.

### `DataConfig`

- Intent: Data loading and caching configuration.
- Fields: 12

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `cache_dir` | `Path` | `Path('data/cache')` | 24 |
| `wrds_enabled` | `bool` | `True` | 25 |
| `optionmetrics_enabled` | `bool` | `True` | 26 |
| `kalshi_enabled` | `bool` | `False` | 27 |
| `default_universe_source` | `str` | `'wrds'` | 28 |
| `lookback_years` | `int` | `15` | 29 |
| `min_bars` | `int` | `500` | 30 |
| `cache_max_staleness_days` | `int` | `21` | 31 |
| `cache_trusted_sources` | `List[str]` | `field(default_factory=lambda: ['wrds', 'wrds_delisting', 'ibkr'])` | 32 |
| `max_missing_bar_fraction` | `float` | `0.05` | 35 |
| `max_zero_volume_fraction` | `float` | `0.25` | 36 |
| `max_abs_daily_return` | `float` | `0.4` | 37 |

### `RegimeConfig`

- Intent: Regime detection configuration.
- Fields: 17

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `model_type` | `str` | `'hmm'` | 44 |
| `n_states` | `int` | `4` | 45 |
| `hmm_max_iter` | `int` | `60` | 46 |
| `hmm_stickiness` | `float` | `0.92` | 47 |
| `min_duration` | `int` | `3` | 48 |
| `hmm_prior_weight` | `float` | `0.3` | 49 |
| `hmm_covariance_type` | `str` | `'full'` | 50 |
| `hmm_auto_select_states` | `bool` | `True` | 51 |
| `hmm_min_states` | `int` | `2` | 52 |
| `hmm_max_states` | `int` | `6` | 53 |
| `jump_model_enabled` | `bool` | `True` | 54 |
| `jump_penalty` | `float` | `0.02` | 55 |
| `expected_changes_per_year` | `float` | `4.0` | 56 |
| `ensemble_enabled` | `bool` | `True` | 57 |
| `ensemble_consensus_threshold` | `int` | `2` | 58 |
| `risk_multiplier` | `Dict[int, float]` | `field(default_factory=lambda: {0: 1.0, 1: 0.85, 2: 0.95, 3: 0.6})` | 59 |
| `stop_multiplier` | `Dict[int, float]` | `field(default_factory=lambda: {0: 1.0, 1: 0.8, 2: 1.2, 3: 1.5})` | 62 |

### `ModelConfig`

- Intent: Model training configuration.
- Fields: 9

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `params` | `Dict[str, Any]` | `field(default_factory=lambda: {'n_estimators': 500, 'max_depth': 4, 'min_samples_leaf': 30, 'learning_rate': 0.05, 'subsample': 0.8, 'max_features': 'sqrt'})` | 71 |
| `max_features_selected` | `int` | `30` | 81 |
| `max_is_oos_gap` | `float` | `0.05` | 82 |
| `cv_folds` | `int` | `5` | 83 |
| `holdout_fraction` | `float` | `0.15` | 84 |
| `ensemble_diversify` | `bool` | `True` | 85 |
| `forward_horizons` | `List[int]` | `field(default_factory=lambda: [5, 10, 20])` | 86 |
| `max_model_versions` | `int` | `5` | 87 |
| `feature_mode` | `str` | `'core'` | 88 |

### `BacktestConfig`

- Intent: Backtesting configuration.
- Fields: 10

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `transaction_cost_bps` | `float` | `20.0` | 95 |
| `entry_threshold` | `float` | `0.005` | 96 |
| `confidence_threshold` | `float` | `0.6` | 97 |
| `max_positions` | `int` | `20` | 98 |
| `position_size_pct` | `float` | `0.05` | 99 |
| `assumed_capital_usd` | `float` | `1000000.0` | 100 |
| `max_portfolio_vol` | `float` | `0.3` | 101 |
| `max_annualized_turnover` | `float` | `500.0` | 102 |
| `max_sector_exposure` | `float` | `0.1` | 103 |
| `wf_max_train_dates` | `Optional[int]` | `1260` | 104 |

### `KellyConfig`

- Intent: Kelly criterion position sizing configuration.
- Fields: 6

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `fraction` | `float` | `0.5` | 111 |
| `max_portfolio_dd` | `float` | `0.2` | 112 |
| `portfolio_blend` | `float` | `0.3` | 113 |
| `bayesian_alpha` | `float` | `2.0` | 114 |
| `bayesian_beta` | `float` | `2.0` | 115 |
| `regime_conditional` | `bool` | `True` | 116 |

### `DrawdownConfig`

- Intent: Drawdown management configuration.
- Fields: 8

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `warning_threshold` | `float` | `-0.05` | 123 |
| `caution_threshold` | `float` | `-0.1` | 124 |
| `critical_threshold` | `float` | `-0.15` | 125 |
| `daily_loss_limit` | `float` | `-0.03` | 126 |
| `weekly_loss_limit` | `float` | `-0.05` | 127 |
| `recovery_days` | `int` | `10` | 128 |
| `size_mult_warning` | `float` | `0.5` | 129 |
| `size_mult_caution` | `float` | `0.25` | 130 |

### `StopLossConfig`

- Intent: Stop-loss configuration.
- Fields: 5

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `hard_stop_pct` | `float` | `-0.08` | 137 |
| `atr_stop_multiplier` | `float` | `2.0` | 138 |
| `trailing_atr_multiplier` | `float` | `1.5` | 139 |
| `trailing_activation_pct` | `float` | `0.02` | 140 |
| `max_holding_days` | `int` | `30` | 141 |

### `ValidationConfig`

- Intent: Statistical validation configuration.
- Fields: 4

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `cpcv_partitions` | `int` | `8` | 148 |
| `cpcv_test_partitions` | `int` | `4` | 149 |
| `spa_bootstraps` | `int` | `400` | 150 |
| `ic_rolling_window` | `int` | `60` | 151 |

### `PromotionConfig`

- Intent: Strategy promotion gate thresholds.
- Fields: 17

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `min_trades` | `int` | `80` | 158 |
| `min_win_rate` | `float` | `0.5` | 159 |
| `min_sharpe` | `float` | `0.75` | 160 |
| `min_profit_factor` | `float` | `1.1` | 161 |
| `max_drawdown` | `float` | `-0.2` | 162 |
| `min_annual_return` | `float` | `0.05` | 163 |
| `max_active_strategies` | `int` | `5` | 164 |
| `require_advanced_contract` | `bool` | `True` | 165 |
| `max_dsr_pvalue` | `float` | `0.05` | 166 |
| `max_pbo` | `float` | `0.5` | 167 |
| `require_statistical_tests` | `bool` | `True` | 168 |
| `require_cpcv` | `bool` | `True` | 169 |
| `require_spa` | `bool` | `False` | 170 |
| `min_wf_oos_corr` | `float` | `0.01` | 171 |
| `min_wf_positive_fold_fraction` | `float` | `0.6` | 172 |
| `max_wf_is_oos_gap` | `float` | `0.2` | 173 |
| `min_regime_positive_fraction` | `float` | `0.5` | 174 |

### `HealthConfig`

- Intent: Health monitoring thresholds.
- Fields: 7

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `min_ic_threshold` | `float` | `0.01` | 181 |
| `signal_decay_threshold` | `float` | `0.5` | 182 |
| `max_correlation_threshold` | `float` | `0.7` | 183 |
| `execution_quality_threshold` | `float` | `2.0` | 184 |
| `tail_ratio_threshold` | `float` | `1.0` | 185 |
| `cvar_threshold` | `float` | `-0.05` | 186 |
| `ir_threshold` | `float` | `0.5` | 187 |

### `PaperTradingConfig`

- Intent: Paper trading configuration.
- Fields: 7

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `initial_capital` | `float` | `1000000.0` | 194 |
| `max_total_positions` | `int` | `30` | 195 |
| `use_kelly_sizing` | `bool` | `True` | 196 |
| `kelly_fraction` | `float` | `0.5` | 197 |
| `kelly_lookback_trades` | `int` | `200` | 198 |
| `kelly_min_size_multiplier` | `float` | `0.25` | 199 |
| `kelly_max_size_multiplier` | `float` | `1.5` | 200 |

### `ExecutionConfig`

- Intent: Trade execution cost modeling.
- Fields: 7

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `spread_bps` | `float` | `3.0` | 207 |
| `max_participation` | `float` | `0.02` | 208 |
| `impact_coeff_bps` | `float` | `25.0` | 209 |
| `min_fill_ratio` | `float` | `0.2` | 210 |
| `dynamic_costs` | `bool` | `True` | 211 |
| `almgren_chriss_enabled` | `bool` | `True` | 212 |
| `almgren_chriss_adv_threshold` | `float` | `0.05` | 213 |

### `SystemConfig`

- Intent: Top-level system configuration aggregating all subsystems.
- Fields: 12

| Field | Type | Default (source) | Line |
|---|---|---|---:|
| `data` | `DataConfig` | `field(default_factory=DataConfig)` | 224 |
| `regime` | `RegimeConfig` | `field(default_factory=RegimeConfig)` | 225 |
| `model` | `ModelConfig` | `field(default_factory=ModelConfig)` | 226 |
| `backtest` | `BacktestConfig` | `field(default_factory=BacktestConfig)` | 227 |
| `kelly` | `KellyConfig` | `field(default_factory=KellyConfig)` | 228 |
| `drawdown` | `DrawdownConfig` | `field(default_factory=DrawdownConfig)` | 229 |
| `stop_loss` | `StopLossConfig` | `field(default_factory=StopLossConfig)` | 230 |
| `validation` | `ValidationConfig` | `field(default_factory=ValidationConfig)` | 231 |
| `promotion` | `PromotionConfig` | `field(default_factory=PromotionConfig)` | 232 |
| `health` | `HealthConfig` | `field(default_factory=HealthConfig)` | 233 |
| `paper_trading` | `PaperTradingConfig` | `field(default_factory=PaperTradingConfig)` | 234 |
| `execution` | `ExecutionConfig` | `field(default_factory=ExecutionConfig)` | 235 |
