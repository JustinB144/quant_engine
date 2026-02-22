# Configuration Reference

This document is a source-derived reference of `config.py` constants grouped by the config section headers in the file.

Notes:
- Values are shown as source expressions where needed (for paths/derived values).
- Inline comments from `config.py` are included when present.

## Paths

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `ROOT_DIR` | `Attribute` | `Path(__file__).parent` |  | 10 |
| `FRAMEWORK_DIR` | `Attribute` | `ROOT_DIR.parent` |  | 11 |
| `MODEL_DIR` | `BinOp` | `ROOT_DIR / "trained_models"` |  | 12 |
| `RESULTS_DIR` | `BinOp` | `ROOT_DIR / "results"` |  | 13 |

## Data Sources

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `DATA_CACHE_DIR` | `BinOp` | `ROOT_DIR / "data" / "cache"` |  | 16 |
| `WRDS_ENABLED` | `bool` | `True` | Try WRDS first; fall back to local cache / yfinance | 17 |
| `OPTIONMETRICS_ENABLED` | `bool` | `True` | Optional WRDS OptionMetrics enrichment | 18 |
| `KALSHI_ENABLED` | `bool` | `False` | Enable Kalshi event-market ingestion pipeline | 19 |
| `KALSHI_ENV` | `str` | `"demo"` | "demo" (default safety) or "prod" | 20 |
| `KALSHI_DEMO_API_BASE_URL` | `str` | `"https://demo-api.kalshi.co/trade-api/v2"` |  | 21 |
| `KALSHI_PROD_API_BASE_URL` | `str` | `"https://api.elections.kalshi.com/trade-api/v2"` |  | 22 |
| `KALSHI_API_BASE_URL` | `Name` | `KALSHI_DEMO_API_BASE_URL` |  | 23 |
| `KALSHI_HISTORICAL_API_BASE_URL` | `Name` | `KALSHI_API_BASE_URL` |  | 24 |
| `KALSHI_HISTORICAL_CUTOFF_TS` | `NoneType` | `None` |  | 25 |
| `KALSHI_RATE_LIMIT_RPS` | `float` | `6.0` |  | 26 |
| `KALSHI_RATE_LIMIT_BURST` | `int` | `2` |  | 27 |
| `KALSHI_DB_PATH` | `BinOp` | `ROOT_DIR / "data" / "kalshi.duckdb"` |  | 28 |
| `KALSHI_SNAPSHOT_HORIZONS` | `list` | `["7d", "1d", "4h", "1h", "15m", "5m"]` |  | 29 |
| `KALSHI_DISTRIBUTION_FREQ` | `str` | `"5min"` |  | 30 |
| `KALSHI_STALE_AFTER_MINUTES` | `int` | `30` |  | 31 |
| `KALSHI_NEAR_EVENT_MINUTES` | `float` | `30.0` |  | 32 |
| `KALSHI_NEAR_EVENT_STALE_MINUTES` | `float` | `2.0` |  | 33 |
| `KALSHI_FAR_EVENT_MINUTES` | `BinOp` | `24.0 * 60.0` |  | 34 |
| `KALSHI_FAR_EVENT_STALE_MINUTES` | `float` | `60.0` |  | 35 |
| `KALSHI_STALE_MARKET_TYPE_MULTIPLIERS` | `dict` | `{ "CPI": 0.80, "UNEMPLOYMENT": 0.90, "FOMC": 0.70, "_default": 1.00, }` |  | 36 |
| `KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD` | `float` | `2.0` |  | 42 |
| `KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD` | `float` | `6.0` |  | 43 |
| `KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER` | `float` | `1.35` |  | 44 |
| `KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER` | `float` | `0.80` |  | 45 |
| `KALSHI_DISTANCE_LAGS` | `list` | `["1h", "1d"]` |  | 46 |
| `KALSHI_TAIL_THRESHOLDS` | `dict` | `{ "CPI": [3.0, 3.5, 4.0], "UNEMPLOYMENT": [4.0, 4.2, 4.5], "FOMC": [0.0, 25.0, 50.0], "_default": [0.0, 0.5, 1.0], }` |  | 47 |
| `DEFAULT_UNIVERSE_SOURCE` | `str` | `"wrds"` | "wrds", "static", or "ibkr" | 53 |
| `CACHE_TRUSTED_SOURCES` | `list` | `["wrds", "wrds_delisting", "ibkr"]` |  | 54 |
| `CACHE_MAX_STALENESS_DAYS` | `int` | `21` |  | 55 |
| `CACHE_WRDS_SPAN_ADVANTAGE_DAYS` | `int` | `180` |  | 56 |
| `REQUIRE_PERMNO` | `bool` | `True` |  | 57 |

## Survivorship

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `SURVIVORSHIP_DB` | `BinOp` | `ROOT_DIR / "data" / "universe_history.db"` |  | 60 |
| `SURVIVORSHIP_UNIVERSE_NAME` | `str` | `"SP500"` |  | 61 |
| `SURVIVORSHIP_SNAPSHOT_FREQ` | `str` | `"quarterly"` | "annual" or "quarterly" | 62 |

## Model Versioning

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `MODEL_REGISTRY` | `BinOp` | `MODEL_DIR / "registry.json"` |  | 65 |
| `MAX_MODEL_VERSIONS` | `int` | `5` | Keep last 5 versions for rollback | 66 |
| `CHAMPION_REGISTRY` | `BinOp` | `MODEL_DIR / "champion_registry.json"` |  | 67 |

## Retraining

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `RETRAIN_MAX_DAYS` | `int` | `30` |  | 70 |
| `RETRAIN_MIN_TRADES` | `int` | `50` |  | 71 |
| `RETRAIN_MIN_WIN_RATE` | `float` | `0.45` |  | 72 |
| `RETRAIN_MIN_CORRELATION` | `float` | `0.05` | Minimum OOS Spearman | 73 |
| `RETRAIN_REGIME_CHANGE_DAYS` | `int` | `10` | Trigger retrain if regime changed for 10+ consecutive days | 74 |
| `RECENCY_DECAY` | `float` | `0.003` | λ for exponential recency weighting (1yr weight ≈ 0.33) | 75 |

## Universe

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `UNIVERSE_FULL` | `list` | `[ # Large cap tech — NOT IN CACHE: AMD, INTC, CRM, ADBE, ORCL "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "…` |  | 78 |
| `UNIVERSE_QUICK` | `list` | `[ "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "JPM", "UNH", "HD", "V", "DDOG", "CRWD", "CAVA", ]` |  | 95 |
| `BENCHMARK` | `str` | `"SPY"` |  | 100 |

## Data

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `LOOKBACK_YEARS` | `int` | `15` |  | 103 |
| `MIN_BARS` | `int` | `500` | minimum bars needed for feature warm-up | 104 |

## Intraday Data

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `INTRADAY_TIMEFRAMES` | `list` | `["4h", "1h", "30m", "15m", "5m", "1m"]` |  | 107 |
| `INTRADAY_CACHE_SOURCE` | `str` | `"ibkr"` |  | 108 |
| `INTRADAY_MIN_BARS` | `int` | `100` |  | 109 |

## Targets

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `FORWARD_HORIZONS` | `list` | `[5, 10, 20]` | days ahead to predict | 112 |

## Features

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `INTERACTION_PAIRS` | `list` | `[ # (feature_a, feature_b, operation) # Regime-conditional signals ("RSI_14", "Hurst_100", "multiply"), ("ZScore_20", "…` |  | 115 |

## Regime

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `REGIME_NAMES` | `dict` | `{ 0: "trending_bull", 1: "trending_bear", 2: "mean_reverting", 3: "high_volatility", }` |  | 138 |
| `MIN_REGIME_SAMPLES` | `int` | `500` | minimum training samples per regime model | 144 |
| `REGIME_MODEL_TYPE` | `str` | `"hmm"` | "hmm" or "rule" | 145 |
| `REGIME_HMM_STATES` | `int` | `4` |  | 146 |
| `REGIME_HMM_MAX_ITER` | `int` | `60` |  | 147 |
| `REGIME_HMM_STICKINESS` | `float` | `0.92` |  | 148 |
| `REGIME_MIN_DURATION` | `int` | `3` |  | 149 |
| `REGIME_SOFT_ASSIGNMENT_THRESHOLD` | `float` | `0.35` |  | 150 |
| `REGIME_HMM_PRIOR_WEIGHT` | `float` | `0.3` | Shrinkage weight toward sticky prior in HMM M-step | 151 |
| `REGIME_HMM_COVARIANCE_TYPE` | `str` | `"full"` | "full" (captures return-vol correlation) or "diag" | 152 |
| `REGIME_HMM_AUTO_SELECT_STATES` | `bool` | `True` | Use BIC to select optimal number of states | 153 |
| `REGIME_HMM_MIN_STATES` | `int` | `2` |  | 154 |
| `REGIME_HMM_MAX_STATES` | `int` | `6` |  | 155 |

## Kalshi Purge/Embargo by Event Type (E3)

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `KALSHI_PURGE_WINDOW_BY_EVENT` | `dict` | `{"CPI": 14, "FOMC": 21, "NFP": 14, "GDP": 14}` |  | 158 |
| `KALSHI_DEFAULT_PURGE_WINDOW` | `int` | `10` |  | 159 |

## Model

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `MODEL_PARAMS` | `dict` | `{ "n_estimators": 500, "max_depth": 4, "min_samples_leaf": 30, "learning_rate": 0.05, "subsample": 0.8, "max_features":…` |  | 162 |
| `MAX_FEATURES_SELECTED` | `int` | `30` | after permutation importance | 170 |
| `MAX_IS_OOS_GAP` | `float` | `0.05` | max allowed degradation (R² or correlation) | 171 |
| `CV_FOLDS` | `int` | `5` |  | 172 |
| `HOLDOUT_FRACTION` | `float` | `0.15` |  | 173 |
| `ENSEMBLE_DIVERSIFY` | `bool` | `True` | Train GBR + ElasticNet + RandomForest and average predictions | 174 |

## Backtest

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `TRANSACTION_COST_BPS` | `int` | `20` | 20 bps round-trip | 177 |
| `ENTRY_THRESHOLD` | `float` | `0.005` | minimum predicted return to enter (0.5%) | 178 |
| `CONFIDENCE_THRESHOLD` | `float` | `0.6` | minimum model confidence | 179 |
| `MAX_POSITIONS` | `int` | `20` |  | 180 |
| `POSITION_SIZE_PCT` | `float` | `0.05` | 5% of capital per position | 181 |
| `BACKTEST_ASSUMED_CAPITAL_USD` | `float` | `1_000_000.0` |  | 182 |
| `EXEC_SPREAD_BPS` | `float` | `3.0` |  | 183 |
| `EXEC_MAX_PARTICIPATION` | `float` | `0.02` |  | 184 |
| `EXEC_IMPACT_COEFF_BPS` | `float` | `25.0` |  | 185 |
| `EXEC_MIN_FILL_RATIO` | `float` | `0.20` |  | 186 |
| `EXEC_DYNAMIC_COSTS` | `bool` | `True` |  | 187 |
| `EXEC_DOLLAR_VOLUME_REF_USD` | `float` | `25_000_000.0` |  | 188 |
| `EXEC_VOL_REF` | `float` | `0.20` |  | 189 |
| `EXEC_VOL_SPREAD_BETA` | `float` | `1.0` |  | 190 |
| `EXEC_GAP_SPREAD_BETA` | `float` | `4.0` |  | 191 |
| `EXEC_RANGE_SPREAD_BETA` | `float` | `2.0` |  | 192 |
| `EXEC_VOL_IMPACT_BETA` | `float` | `1.0` |  | 193 |
| `MAX_PORTFOLIO_VOL` | `float` | `0.30` |  | 194 |
| `REGIME_RISK_MULTIPLIER` | `dict` | `{ 0: 1.00, # trending_bull 1: 0.85, # trending_bear 2: 0.95, # mean_reverting 3: 0.60, # high_volatility }` |  | 195 |
| `REGIME_STOP_MULTIPLIER` | `dict` | `{ 0: 1.0, # trending_bull: standard stops 1: 0.8, # trending_bear: tighter stops (cut losses faster) 2: 1.2, # mean_rev…` |  | 201 |
| `MAX_ANNUALIZED_TURNOVER` | `float` | `500.0` | 500% annualized turnover warning threshold | 207 |
| `MAX_SECTOR_EXPOSURE` | `float` | `0.10` | Maximum net weight in any single GICS sector (±10%) | 208 |
| `GICS_SECTORS` | `dict` | `{}` | Maps permno -> GICS sector name (populate from WRDS Compustat) | 209 |

## Almgren-Chriss Optimal Execution

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `ALMGREN_CHRISS_ENABLED` | `bool` | `True` | Use AC model for large positions | 212 |
| `ALMGREN_CHRISS_ADV_THRESHOLD` | `float` | `0.05` | Positions > 5% of ADV use AC cost model | 213 |

## Validation

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `CPCV_PARTITIONS` | `int` | `8` |  | 216 |
| `CPCV_TEST_PARTITIONS` | `int` | `4` |  | 217 |
| `SPA_BOOTSTRAPS` | `int` | `400` |  | 218 |

## Data Quality

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `DATA_QUALITY_ENABLED` | `bool` | `True` |  | 221 |
| `MAX_MISSING_BAR_FRACTION` | `float` | `0.05` |  | 222 |
| `MAX_ZERO_VOLUME_FRACTION` | `float` | `0.25` |  | 223 |
| `MAX_ABS_DAILY_RETURN` | `float` | `0.40` |  | 224 |

## Autopilot (discovery -> promotion -> paper trading)

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `AUTOPILOT_DIR` | `BinOp` | `RESULTS_DIR / "autopilot"` |  | 227 |
| `STRATEGY_REGISTRY_PATH` | `BinOp` | `AUTOPILOT_DIR / "strategy_registry.json"` |  | 228 |
| `PAPER_STATE_PATH` | `BinOp` | `AUTOPILOT_DIR / "paper_state.json"` |  | 229 |
| `AUTOPILOT_CYCLE_REPORT` | `BinOp` | `AUTOPILOT_DIR / "latest_cycle.json"` |  | 230 |
| `AUTOPILOT_FEATURE_MODE` | `str` | `"core"` | "core" (reduced complexity) or "full" | 231 |
| `DISCOVERY_ENTRY_MULTIPLIERS` | `list` | `[0.8, 1.0, 1.2]` |  | 233 |
| `DISCOVERY_CONFIDENCE_OFFSETS` | `list` | `[-0.10, 0.0, 0.10]` |  | 234 |
| `DISCOVERY_RISK_VARIANTS` | `list` | `[False, True]` |  | 235 |
| `DISCOVERY_MAX_POSITIONS_VARIANTS` | `list` | `[10, 20]` |  | 236 |
| `PROMOTION_MIN_TRADES` | `int` | `80` |  | 238 |
| `PROMOTION_MIN_WIN_RATE` | `float` | `0.50` |  | 239 |
| `PROMOTION_MIN_SHARPE` | `float` | `0.75` |  | 240 |
| `PROMOTION_MIN_PROFIT_FACTOR` | `float` | `1.10` |  | 241 |
| `PROMOTION_MAX_DRAWDOWN` | `float` | `-0.20` |  | 242 |
| `PROMOTION_MIN_ANNUAL_RETURN` | `float` | `0.05` |  | 243 |
| `PROMOTION_MAX_ACTIVE_STRATEGIES` | `int` | `5` |  | 244 |
| `PROMOTION_REQUIRE_ADVANCED_CONTRACT` | `bool` | `True` |  | 245 |
| `PROMOTION_MAX_DSR_PVALUE` | `float` | `0.05` |  | 246 |
| `PROMOTION_MAX_PBO` | `float` | `0.50` |  | 247 |
| `PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED` | `bool` | `True` |  | 248 |
| `PROMOTION_MAX_CAPACITY_UTILIZATION` | `float` | `1.0` |  | 249 |
| `PROMOTION_MIN_WF_OOS_CORR` | `float` | `0.01` |  | 250 |
| `PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION` | `float` | `0.60` |  | 251 |
| `PROMOTION_MAX_WF_IS_OOS_GAP` | `float` | `0.20` |  | 252 |
| `PROMOTION_MIN_REGIME_POSITIVE_FRACTION` | `float` | `0.50` |  | 253 |
| `PROMOTION_EVENT_MAX_WORST_EVENT_LOSS` | `float` | `-0.08` |  | 254 |
| `PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE` | `float` | `0.50` |  | 255 |
| `PROMOTION_EVENT_MIN_REGIME_STABILITY` | `float` | `0.40` |  | 256 |
| `PAPER_INITIAL_CAPITAL` | `float` | `1_000_000.0` |  | 258 |
| `PAPER_MAX_TOTAL_POSITIONS` | `int` | `30` |  | 259 |
| `PAPER_USE_KELLY_SIZING` | `bool` | `True` |  | 260 |
| `PAPER_KELLY_FRACTION` | `float` | `0.50` |  | 261 |
| `PAPER_KELLY_LOOKBACK_TRADES` | `int` | `200` |  | 262 |
| `PAPER_KELLY_MIN_SIZE_MULTIPLIER` | `float` | `0.25` |  | 263 |
| `PAPER_KELLY_MAX_SIZE_MULTIPLIER` | `float` | `1.50` |  | 264 |

## Feature Profiles

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `FEATURE_MODE_DEFAULT` | `str` | `"core"` | "full" or "core" | 267 |

## Regime Trade Gating

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `REGIME_2_TRADE_ENABLED` | `bool` | `False` | Model has Sharpe -0.91 in mean-revert regime; suppress entries | 270 |
| `REGIME_2_SUPPRESSION_MIN_CONFIDENCE` | `float` | `0.5` | Only suppress when confidence exceeds this | 271 |

## Drawdown Tiers

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `DRAWDOWN_WARNING_THRESHOLD` | `float` | `-0.05` | -5% drawdown: reduce sizing 50% | 274 |
| `DRAWDOWN_CAUTION_THRESHOLD` | `float` | `-0.10` | -10% drawdown: no new entries, 25% sizing | 275 |
| `DRAWDOWN_CRITICAL_THRESHOLD` | `float` | `-0.15` | -15% drawdown: force liquidate | 276 |
| `DRAWDOWN_DAILY_LOSS_LIMIT` | `float` | `-0.03` | -3% daily loss limit | 277 |
| `DRAWDOWN_WEEKLY_LOSS_LIMIT` | `float` | `-0.05` | -5% weekly loss limit | 278 |
| `DRAWDOWN_RECOVERY_DAYS` | `int` | `10` | Days to return to full sizing after recovery | 279 |
| `DRAWDOWN_SIZE_MULT_WARNING` | `float` | `0.50` | Size multiplier during WARNING tier | 280 |
| `DRAWDOWN_SIZE_MULT_CAUTION` | `float` | `0.25` | Size multiplier during CAUTION tier | 281 |

## Stop Loss

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `HARD_STOP_PCT` | `float` | `-0.08` | -8% hard stop loss | 284 |
| `ATR_STOP_MULTIPLIER` | `float` | `2.0` | Initial stop at 2x ATR | 285 |
| `TRAILING_ATR_MULTIPLIER` | `float` | `1.5` | Trailing stop at 1.5x ATR | 286 |
| `TRAILING_ACTIVATION_PCT` | `float` | `0.02` | Activate trailing stop after +2% gain | 287 |
| `MAX_HOLDING_DAYS` | `int` | `30` | Time-based stop at 30 days | 288 |

## Almgren-Chriss

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `ALMGREN_CHRISS_FALLBACK_VOL` | `float` | `0.20` | Fallback annualized vol when realized unavailable | 291 |

## Model Governance

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `GOVERNANCE_SCORE_WEIGHTS` | `dict` | `{"oos_spearman": 1.5, "holdout_spearman": 1.0, "cv_gap_penalty": -0.5}` | Weighted governance score components | 294 |

## Validation

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `IC_ROLLING_WINDOW` | `int` | `60` | Rolling window for Information Coefficient calculation | 301 |

## Alert System

| Constant | Type (best effort) | Value / Expression | Notes | Line |
|---|---|---|---|---:|
| `ALERT_WEBHOOK_URL` | `str` | `""` | Empty = disabled; set to Slack/Discord/custom webhook URL | 270 |
| `ALERT_HISTORY_FILE` | `BinOp` | `RESULTS_DIR / "alerts_history.json"` |  | 271 |
