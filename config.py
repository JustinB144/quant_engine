"""
Central configuration for the quant engine.

Self-contained — no references to automated_portfolio_system.

Config Status Legend
====================
Each constant is annotated with one of the following statuses:

  ACTIVE      — Imported and used by running code.  Changing the value
                affects live behaviour.
  PLACEHOLDER — Defined for future use.  Either never imported, or
                imported but the surrounding feature is not yet wired
                end-to-end.  Safe to change without affecting current
                behaviour.
  DEPRECATED  — Superseded by a newer mechanism.  Kept only for
                backward-compatibility; will be removed in a future
                release.

Search for ``# STATUS:`` to locate all annotations.
"""
from pathlib import Path
from typing import Dict

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent                  # STATUS: ACTIVE — base path for all relative references
FRAMEWORK_DIR = ROOT_DIR.parent                   # STATUS: ACTIVE — parent directory of quant_engine
MODEL_DIR = ROOT_DIR / "trained_models"           # STATUS: ACTIVE — models/versioning.py, models/trainer.py
RESULTS_DIR = ROOT_DIR / "results"                # STATUS: ACTIVE — backtest output, autopilot reports, alerts

# ── Data Sources ──────────────────────────────────────────────────────
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"      # STATUS: ACTIVE — data/loader.py, data/local_cache.py
WRDS_ENABLED = True                               # STATUS: ACTIVE — data/loader.py; try WRDS first, fall back to local cache / IBKR

# PLACEHOLDER — OptionMetrics IV surface data integration.
# Set to True once api/routers/iv_surface.py has a working /iv-surface/heston endpoint
# and data loader merges OptionMetrics surface into OHLCV panels.
# Note: data/loader.py does gate on this flag (lines 296, 625), but the upstream
# OptionMetrics data source is not configured in most environments, so the try/except
# blocks silently fall through.  Disabled until the full pipeline is verified.
OPTIONMETRICS_ENABLED = False                     # STATUS: PLACEHOLDER — data/loader.py gates on this but pipeline incomplete

# ── Execution Contract (Truth Layer) ─────────────────────────────────
RET_TYPE = "log"                                  # STATUS: ACTIVE — "log" (log returns) or "simple" (pct returns)
LABEL_H = 5                                       # STATUS: ACTIVE — label horizon in trading days
PX_TYPE = "close"                                 # STATUS: ACTIVE — "close" or "open" for price baseline
ENTRY_PRICE_TYPE = "next_bar_open"                # STATUS: ACTIVE — "next_bar_open" (no look-ahead)

# ── Truth Layer Feature Flags ────────────────────────────────────────
TRUTH_LAYER_STRICT_PRECONDITIONS = True           # STATUS: ACTIVE — raise on invalid execution contract
TRUTH_LAYER_FAIL_ON_CORRUPT = True                # STATUS: ACTIVE — block corrupt OHLCV from pipeline
TRUTH_LAYER_ENFORCE_CAUSALITY = True              # STATUS: ACTIVE — enforce feature causality at runtime
TRUTH_LAYER_COMPUTE_NULL_BASELINES = False        # STATUS: ACTIVE — compute null baselines per backtest (adds ~4x time)
TRUTH_LAYER_COST_STRESS_ENABLED = False           # STATUS: ACTIVE — run cost stress sweep per backtest (adds ~4x time)

# ── Cost Stress Testing ──────────────────────────────────────────────
COST_STRESS_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0]   # STATUS: ACTIVE — cost sweep factors

KALSHI_ENABLED = False                            # STATUS: ACTIVE — kalshi/provider.py, run_kalshi_event_pipeline.py; disabled by design
KALSHI_ENV = "demo"                               # STATUS: ACTIVE — selects demo vs prod API URL; "demo" (safety) or "prod"
KALSHI_DEMO_API_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"   # STATUS: ACTIVE — used to compute KALSHI_API_BASE_URL
KALSHI_PROD_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"  # STATUS: ACTIVE — used when KALSHI_ENV="prod"
KALSHI_API_BASE_URL = KALSHI_DEMO_API_BASE_URL    # STATUS: ACTIVE — kalshi/provider.py base URL
KALSHI_HISTORICAL_API_BASE_URL = KALSHI_API_BASE_URL  # STATUS: ACTIVE — kalshi historical data endpoint
KALSHI_HISTORICAL_CUTOFF_TS = None                # STATUS: ACTIVE — optional cutoff for historical fetches
KALSHI_RATE_LIMIT_RPS = 6.0                       # STATUS: ACTIVE — kalshi/provider.py rate limiter
KALSHI_RATE_LIMIT_BURST = 2                       # STATUS: ACTIVE — kalshi/provider.py burst allowance
KALSHI_DB_PATH = ROOT_DIR / "data" / "kalshi.duckdb"  # STATUS: ACTIVE — DuckDB path for Kalshi snapshots
KALSHI_SNAPSHOT_HORIZONS = ["7d", "1d", "4h", "1h", "15m", "5m"]  # STATUS: ACTIVE — kalshi/provider.py, run_kalshi_event_pipeline.py
KALSHI_DISTRIBUTION_FREQ = "5min"                 # STATUS: ACTIVE — kalshi/provider.py distribution resampling
KALSHI_STALE_AFTER_MINUTES = 30                   # STATUS: ACTIVE — kalshi/provider.py staleness detection
KALSHI_NEAR_EVENT_MINUTES = 30.0                  # STATUS: ACTIVE — kalshi/provider.py near-event window
KALSHI_NEAR_EVENT_STALE_MINUTES = 2.0             # STATUS: ACTIVE — tighter staleness near events
KALSHI_FAR_EVENT_MINUTES = 24.0 * 60.0            # STATUS: ACTIVE — far-event window (24h)
KALSHI_FAR_EVENT_STALE_MINUTES = 60.0             # STATUS: ACTIVE — relaxed staleness far from events
KALSHI_STALE_MARKET_TYPE_MULTIPLIERS = {           # STATUS: ACTIVE — kalshi/provider.py per-event-type staleness
    "CPI": 0.80,
    "UNEMPLOYMENT": 0.90,
    "FOMC": 0.70,
    "_default": 1.00,
}
KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD = 2.0        # STATUS: ACTIVE — kalshi/provider.py low-liquidity threshold
KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD = 6.0       # STATUS: ACTIVE — kalshi/provider.py high-liquidity threshold
KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER = 1.35      # STATUS: ACTIVE — tighten staleness for illiquid markets
KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER = 0.80     # STATUS: ACTIVE — relax staleness for liquid markets
KALSHI_DISTANCE_LAGS = ["1h", "1d"]               # STATUS: ACTIVE — kalshi/provider.py distance lag features
KALSHI_TAIL_THRESHOLDS = {                         # STATUS: ACTIVE — kalshi/provider.py tail-risk thresholds
    "CPI": [3.0, 3.5, 4.0],
    "UNEMPLOYMENT": [4.0, 4.2, 4.5],
    "FOMC": [0.0, 25.0, 50.0],
    "_default": [0.0, 0.5, 1.0],
}
DEFAULT_UNIVERSE_SOURCE = "wrds"                  # STATUS: PLACEHOLDER — defined but never imported; "wrds", "static", or "ibkr"
CACHE_TRUSTED_SOURCES = ["wrds", "wrds_delisting", "ibkr"]  # STATUS: ACTIVE — data/local_cache.py source ranking
CACHE_MAX_STALENESS_DAYS = 21                     # STATUS: ACTIVE — data/local_cache.py max cache age
CACHE_WRDS_SPAN_ADVANTAGE_DAYS = 180              # STATUS: ACTIVE — data/local_cache.py WRDS preference window
REQUIRE_PERMNO = True                             # STATUS: ACTIVE — data/loader.py, backtest/engine.py PERMNO validation

# ── Survivorship ──────────────────────────────────────────────────────
SURVIVORSHIP_DB = ROOT_DIR / "data" / "universe_history.db"  # STATUS: ACTIVE — data/survivorship.py, data/loader.py
SURVIVORSHIP_UNIVERSE_NAME = "SP500"              # STATUS: ACTIVE — data/loader.py, autopilot/engine.py
SURVIVORSHIP_SNAPSHOT_FREQ = "quarterly"           # STATUS: ACTIVE — data/loader.py; "annual" or "quarterly"

# ── Model Versioning ─────────────────────────────────────────────────
MODEL_REGISTRY = MODEL_DIR / "registry.json"      # STATUS: PLACEHOLDER — defined but never imported; CHAMPION_REGISTRY is used instead
MAX_MODEL_VERSIONS = 5                            # STATUS: ACTIVE — models/versioning.py; keep last 5 versions for rollback
CHAMPION_REGISTRY = MODEL_DIR / "champion_registry.json"  # STATUS: ACTIVE — models/governance.py

# ── Retraining ───────────────────────────────────────────────────────
RETRAIN_MAX_DAYS = 30                             # STATUS: ACTIVE — schedule-based retrain trigger (30 calendar days)
RETRAIN_MIN_TRADES = 50                           # STATUS: PLACEHOLDER — defined but never imported by running code
RETRAIN_MIN_WIN_RATE = 0.45                       # STATUS: PLACEHOLDER — defined but never imported by running code
RETRAIN_MIN_CORRELATION = 0.05                    # STATUS: PLACEHOLDER — minimum OOS Spearman; defined but never imported
RETRAIN_REGIME_CHANGE_DAYS = 10                   # STATUS: ACTIVE — run_retrain.py; trigger retrain if regime changed for 10+ consecutive days
RECENCY_DECAY = 0.003                             # STATUS: ACTIVE — models/trainer.py; exponential recency weighting (1yr weight ~ 0.33)

# ── Universe ───────────────────────────────────────────────────────────
UNIVERSE_FULL = [                                  # STATUS: ACTIVE — data/loader.py, run_*.py, autopilot/engine.py
    # Large cap tech — NOT IN CACHE: AMD, INTC, CRM, ADBE, ORCL
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "ORCL",
    # Mid cap tech — NOT IN CACHE: DDOG, NET, CRWD, ZS, SNOW, MDB, PANW, FTNT
    "DDOG", "NET", "CRWD", "ZS", "SNOW", "MDB", "PANW", "FTNT",
    # Healthcare — NOT IN CACHE: ABBV, TMO
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT",
    # Consumer — NOT IN CACHE: MCD, TGT, COST
    "AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD", "TGT", "COST",
    # Financial — NOT IN CACHE: GS, BLK, MA
    "JPM", "BAC", "GS", "MS", "BLK", "V", "MA",
    # Industrial — NOT IN CACHE: BA
    "CAT", "DE", "GE", "HON", "BA", "LMT",
    # Volatile / small-mid — NOT IN CACHE: CAVA, BROS, TOST, CHWY, ETSY, POOL
    "CAVA", "BROS", "TOST", "CHWY", "ETSY", "POOL",
]

UNIVERSE_QUICK = [                                 # STATUS: ACTIVE — run_*.py quick-test universe
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA",
    "JPM", "UNH", "HD", "V", "DDOG", "CRWD", "CAVA",
]

UNIVERSE_INTRADAY = [                              # STATUS: ACTIVE — 128-ticker intraday universe (all IBKR-downloaded timeframes)
    "AAPL", "ABBV", "ABT", "ADBE", "ADI", "ADM", "ADSK", "AEP", "AMD", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BBY", "BDX", "BLK", "BROS", "BSX", "CAT",
    "CAVA", "CCL", "CHWY", "CL", "CLX", "CMCSA", "COST", "CPB", "CRM", "CRWD",
    "CSCO", "CVS", "CVX", "DDOG", "DE", "DHR", "DIS", "DOV", "DRI", "DUK",
    "DVN", "EBAY", "ECL", "EMR", "EOG", "ETN", "ETSY", "FCX", "FTNT", "GD",
    "GE", "GILD", "GIS", "GOOGL", "GPC", "GS", "HAL", "HD", "HON", "HSY",
    "HUM", "IBM", "INTC", "INTU", "ITW", "JNJ", "JPM", "KMB", "KR", "LEN",
    "LLY", "LMT", "LUV", "MA", "MAR", "MCD", "MCO", "MDB", "META", "MMM",
    "MRK", "MS", "MSFT", "MSI", "MU", "NEE", "NET", "NKE", "NOC", "NSC",
    "NTAP", "NVDA", "ORCL", "PANW", "PCAR", "PFE", "PG", "PGR", "PH", "PLD",
    "PNC", "POOL", "PSA", "QCOM", "SBUX", "SCHW", "SLB", "SNOW", "SO", "SPG",
    "SPY", "SWK", "SYY", "TGT", "TMO", "TOST", "TSLA", "TXN", "UNH", "UNP",
    "USB", "V", "VLO", "VMC", "WY", "XEL", "XOM", "ZS",
]

BENCHMARK = "SPY"                                 # STATUS: ACTIVE — backtest/engine.py, api/routers/benchmark.py

# ── Data ───────────────────────────────────────────────────────────────
LOOKBACK_YEARS = 15                               # STATUS: ACTIVE — data/loader.py; years of historical data to load
MIN_BARS = 500                                    # STATUS: ACTIVE — data/loader.py; minimum bars needed for feature warm-up

# ── Intraday Data ─────────────────────────────────────────────────
INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]  # STATUS: PLACEHOLDER — defined but never imported
INTRADAY_CACHE_SOURCE = "ibkr"                    # STATUS: PLACEHOLDER — defined but never imported
INTRADAY_MIN_BARS = 100                           # STATUS: ACTIVE — features/pipeline.py; minimum intraday bars for feature computation
MARKET_OPEN = "09:30"                             # STATUS: ACTIVE — features/intraday.py; US equity regular-session open (ET)
MARKET_CLOSE = "16:00"                            # STATUS: ACTIVE — features/intraday.py; US equity regular-session close (ET)

# ── Targets ────────────────────────────────────────────────────────────
FORWARD_HORIZONS = [5, 10, 20]                    # STATUS: ACTIVE — models/trainer.py, run_*.py; days ahead to predict

# ── Features ───────────────────────────────────────────────────────────
INTERACTION_PAIRS = [                              # STATUS: ACTIVE — features/pipeline.py; regime-conditional interaction features
    # (feature_a, feature_b, operation)
    # Regime-conditional signals
    ("RSI_14", "Hurst_100", "multiply"),
    ("ZScore_20", "VolTS_10_60", "multiply"),
    ("MACD_12_26", "AutoCorr_20_1", "multiply"),
    ("RSI_14", "VarRatio_100_5", "multiply"),
    # Volatility x liquidity
    ("NATR_14", "Amihud_20", "multiply"),
    # Predictability x trend strength
    ("Entropy_20", "ADX_14", "multiply"),
    # Vol forecast x tail risk
    ("GARCH_252", "Skew_60", "multiply"),
    # Trend quality x noise
    ("Hurst_100", "FracDim_100", "multiply"),
    # Vol estimator divergence
    ("ParkVol_20", "GKVol_20", "ratio"),
    # Centered indicators
    ("RSI_14", None, "center_50"),
    ("Stoch_14", None, "center_50"),
]

# ── Regime ─────────────────────────────────────────────────────────────
REGIME_NAMES = {                                   # STATUS: ACTIVE — used in 15+ files; canonical regime label mapping
    0: "trending_bull",
    1: "trending_bear",
    2: "mean_reverting",
    3: "high_volatility",
}
MIN_REGIME_SAMPLES = 500                          # STATUS: ACTIVE — models/trainer.py; minimum training samples per regime model
REGIME_MODEL_TYPE = "jump"                        # STATUS: ACTIVE — regime/detector.py; "jump", "hmm", or "rule"
REGIME_HMM_STATES = 4                             # STATUS: ACTIVE — regime/hmm.py; number of hidden states
REGIME_HMM_MAX_ITER = 60                          # STATUS: ACTIVE — regime/hmm.py; EM iteration limit
REGIME_HMM_STICKINESS = 0.92                      # STATUS: ACTIVE — regime/hmm.py; diagonal prior bias for sticky transitions
REGIME_MIN_DURATION = 3                           # STATUS: ACTIVE — regime/detector.py; minimum regime duration in days
REGIME_SOFT_ASSIGNMENT_THRESHOLD = 0.35           # STATUS: ACTIVE — models/trainer.py; probability threshold for soft regime labels
REGIME_HMM_PRIOR_WEIGHT = 0.3                     # STATUS: ACTIVE — regime/hmm.py; shrinkage weight toward sticky prior in M-step
REGIME_HMM_COVARIANCE_TYPE = "full"               # STATUS: ACTIVE — regime/hmm.py; "full" (captures return-vol correlation) or "diag"
REGIME_HMM_AUTO_SELECT_STATES = True              # STATUS: ACTIVE — regime/hmm.py; use BIC to select optimal number of states
REGIME_HMM_MIN_STATES = 2                         # STATUS: ACTIVE — regime/hmm.py; BIC state search lower bound
REGIME_HMM_MAX_STATES = 6                         # STATUS: ACTIVE — regime/hmm.py; BIC state search upper bound
REGIME_JUMP_MODEL_ENABLED = True                  # STATUS: ACTIVE — regime/detector.py; statistical jump model alongside HMM
REGIME_JUMP_PENALTY = 0.02                        # STATUS: ACTIVE — regime/jump_model.py; jump penalty lambda (higher = fewer transitions)
REGIME_EXPECTED_CHANGES_PER_YEAR = 4              # STATUS: ACTIVE — regime/jump_model.py; calibrate jump penalty from expected regime changes/yr
REGIME_ENSEMBLE_ENABLED = True                    # STATUS: ACTIVE — regime/detector.py; combine HMM + JM + rule-based via majority vote
REGIME_ENSEMBLE_CONSENSUS_THRESHOLD = 2           # STATUS: ACTIVE — regime/detector.py; require N of 3 methods to agree for transition

# PyPI jumpmodels package configuration
REGIME_JUMP_USE_PYPI_PACKAGE = True               # STATUS: ACTIVE — regime/jump_model_pypi.py; True=PyPI jumpmodels, False=legacy custom
REGIME_JUMP_CV_FOLDS = 5                          # STATUS: ACTIVE — regime/jump_model_pypi.py; time-series CV folds for lambda selection
REGIME_JUMP_LAMBDA_RANGE = (0.005, 0.15)          # STATUS: ACTIVE — regime/jump_model_pypi.py; search range for jump penalty
REGIME_JUMP_LAMBDA_STEPS = 20                     # STATUS: ACTIVE — regime/jump_model_pypi.py; grid points for lambda search
REGIME_JUMP_MAX_ITER = 50                         # STATUS: ACTIVE — regime/jump_model_pypi.py; coordinate descent iterations
REGIME_JUMP_TOL = 1e-6                            # STATUS: ACTIVE — regime/jump_model_pypi.py; convergence tolerance
REGIME_JUMP_USE_CONTINUOUS = True                  # STATUS: ACTIVE — regime/jump_model_pypi.py; continuous JM for soft probabilities
REGIME_JUMP_MODE_LOSS_WEIGHT = 0.1                # STATUS: ACTIVE — regime/jump_model_pypi.py; mode loss penalty (continuous JM)

# ── Bayesian Online Change-Point Detection ─────────────────────────────
BOCPD_ENABLED = True                              # STATUS: ACTIVE — regime/detector.py, regime/bocpd.py; enable BOCPD alongside HMM
BOCPD_HAZARD_FUNCTION = "constant"                # STATUS: ACTIVE — regime/bocpd.py; "constant" or "geometric"
BOCPD_HAZARD_LAMBDA = 1.0 / 60                    # STATUS: ACTIVE — regime/bocpd.py; constant hazard rate (1 change per 60 bars)
BOCPD_LIKELIHOOD_TYPE = "gaussian"                # STATUS: ACTIVE — regime/bocpd.py; observation model type (only "gaussian" for v1)
BOCPD_RUNLENGTH_DEPTH = 200                       # STATUS: ACTIVE — regime/bocpd.py; max run-length to track (older hypotheses pruned)
BOCPD_CHANGEPOINT_THRESHOLD = 0.50                # STATUS: ACTIVE — regime/detector.py; flag changepoint if P(cp) > threshold

# ── Shock Vector Schema ────────────────────────────────────────────────
SHOCK_VECTOR_SCHEMA_VERSION = "1.0"               # STATUS: ACTIVE — regime/shock_vector.py; schema version for backward compatibility
SHOCK_VECTOR_INCLUDE_STRUCTURAL = True            # STATUS: ACTIVE — regime/shock_vector.py; include structural features in vector

# ── Kalshi Purge/Embargo by Event Type (E3) ─────────────────────────────
KALSHI_PURGE_WINDOW_BY_EVENT = {"CPI": 14, "FOMC": 21, "NFP": 14, "GDP": 14}  # STATUS: PLACEHOLDER — defined but never imported
KALSHI_DEFAULT_PURGE_WINDOW = 10                  # STATUS: PLACEHOLDER — defined but never imported; companion to KALSHI_PURGE_WINDOW_BY_EVENT

# ── Model ──────────────────────────────────────────────────────────────
MODEL_PARAMS = {                                   # STATUS: ACTIVE — models/trainer.py; GBR hyperparameters
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 30,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "max_features": "sqrt",
}
MAX_FEATURES_SELECTED = 30                        # STATUS: ACTIVE — models/trainer.py; after permutation importance
MAX_IS_OOS_GAP = 0.05                            # STATUS: ACTIVE — models/trainer.py; max allowed IS-OOS degradation (R^2 or correlation)
CV_FOLDS = 5                                      # STATUS: ACTIVE — models/trainer.py; cross-validation folds
HOLDOUT_FRACTION = 0.15                           # STATUS: ACTIVE — models/trainer.py; holdout set fraction
ENSEMBLE_DIVERSIFY = True                         # STATUS: ACTIVE — models/trainer.py; train GBR + ElasticNet + RandomForest and average

# ── Walk-Forward Rolling Window ───────────────────────────────────────
# When set, training windows are capped at this many unique dates so old
# data rolls off.  ``None`` means expanding windows (all history).
# A typical value of 1260 ~ 5 years of trading days.
WF_MAX_TRAIN_DATES = 1260                         # STATUS: ACTIVE — backtest/engine.py, models/trainer.py; rolling walk-forward window

# ── Backtest ───────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 20                         # STATUS: ACTIVE — backtest/engine.py; 20 bps round-trip
ENTRY_THRESHOLD = 0.005                           # STATUS: ACTIVE — backtest/engine.py; minimum predicted return to enter (0.5%)
CONFIDENCE_THRESHOLD = 0.6                        # STATUS: ACTIVE — backtest/engine.py; minimum model confidence
MAX_POSITIONS = 20                                # STATUS: ACTIVE — backtest/engine.py; max simultaneous positions
POSITION_SIZE_PCT = 0.05                          # STATUS: ACTIVE — backtest/engine.py; 5% of capital per position
BACKTEST_ASSUMED_CAPITAL_USD = 1_000_000.0        # STATUS: ACTIVE — backtest/engine.py; initial capital
EXEC_SPREAD_BPS = 3.0                            # STATUS: ACTIVE — backtest/engine.py; base spread cost
EXEC_MAX_PARTICIPATION = 0.02                     # STATUS: ACTIVE — backtest/engine.py; max 2% of daily volume
EXEC_IMPACT_COEFF_BPS = 25.0                     # STATUS: ACTIVE — backtest/engine.py; market impact coefficient
EXEC_MIN_FILL_RATIO = 0.20                       # STATUS: ACTIVE — backtest/engine.py; minimum fill ratio
EXEC_DYNAMIC_COSTS = True                         # STATUS: ACTIVE — backtest/engine.py; condition costs on market state
EXEC_DOLLAR_VOLUME_REF_USD = 25_000_000.0        # STATUS: ACTIVE — backtest/engine.py; dollar volume reference
EXEC_VOL_REF = 0.20                              # STATUS: ACTIVE — backtest/engine.py; reference volatility
EXEC_VOL_SPREAD_BETA = 1.0                       # STATUS: ACTIVE — backtest/engine.py; vol-spread sensitivity
EXEC_GAP_SPREAD_BETA = 4.0                       # STATUS: ACTIVE — backtest/engine.py; gap-spread sensitivity
EXEC_RANGE_SPREAD_BETA = 2.0                     # STATUS: ACTIVE — backtest/engine.py; range-spread sensitivity
EXEC_VOL_IMPACT_BETA = 1.0                       # STATUS: ACTIVE — backtest/engine.py; vol-impact sensitivity
MAX_PORTFOLIO_VOL = 0.30                          # STATUS: ACTIVE — backtest/engine.py, risk/portfolio_optimizer.py; max annualized vol
REGIME_RISK_MULTIPLIER = {                         # STATUS: ACTIVE — backtest/engine.py; regime-conditional position sizing multipliers
    0: 1.00,  # trending_bull
    1: 0.85,  # trending_bear
    2: 0.95,  # mean_reverting
    3: 0.60,  # high_volatility
}
REGIME_STOP_MULTIPLIER = {                         # STATUS: ACTIVE — backtest/engine.py; regime-conditional stop loss multipliers
    0: 1.0,   # trending_bull: standard stops
    1: 0.8,   # trending_bear: tighter stops (cut losses faster)
    2: 1.2,   # mean_reverting: wider stops (expect reversals)
    3: 1.5,   # high_volatility: wider stops (avoid noise stops)
}
MAX_ANNUALIZED_TURNOVER = 500.0                   # STATUS: ACTIVE — backtest/engine.py; 500% annualized turnover warning threshold

# Maximum net weight in any single GICS sector (+/-10%).
# Enforced by risk/portfolio_optimizer.py ONLY when GICS_SECTORS is populated.
MAX_SECTOR_EXPOSURE = 0.10                        # STATUS: ACTIVE — risk/portfolio_optimizer.py; but INACTIVE when GICS_SECTORS is empty

# PLACEHOLDER: Populate via `run_wrds_daily_refresh.py --gics` or manually.
# When empty, MAX_SECTOR_EXPOSURE constraint is NOT enforced in portfolio_optimizer.
# Maps ticker -> GICS sector name for sector-neutrality constraints.
# To populate from WRDS Compustat:
#
#   SELECT gvkey, gsubind AS gics_code, conm
#   FROM comp.company
#   WHERE gsubind IS NOT NULL;
#
# Then map the 2-digit sector code to a human-readable name.
GICS_SECTORS: Dict[str, str] = {}                 # STATUS: PLACEHOLDER — empty; sector constraints not enforced

# ── Almgren-Chriss Optimal Execution ─────────────────────────────────
# PLACEHOLDER — Almgren-Chriss optimal execution model.
# The AC model exists in backtest/engine.py and is gated on this flag,
# but it is not wired into the trade simulation loop end-to-end.
# Set to True once backtest/execution.py integrates AC into the fill model.
ALMGREN_CHRISS_ENABLED = False                    # STATUS: PLACEHOLDER — backtest/engine.py gates on this but not fully integrated
ALMGREN_CHRISS_ADV_THRESHOLD = 0.05               # STATUS: ACTIVE — backtest/engine.py; positions > 5% of ADV use AC cost model

# ── Validation ─────────────────────────────────────────────────────────
CPCV_PARTITIONS = 8                               # STATUS: ACTIVE — backtest/validation.py; combinatorial purged CV partitions
CPCV_TEST_PARTITIONS = 4                          # STATUS: ACTIVE — backtest/validation.py; CPCV test partitions
SPA_BOOTSTRAPS = 400                              # STATUS: ACTIVE — backtest/validation.py; SPA bootstrap trials

# ── Data Quality ───────────────────────────────────────────────────────
DATA_QUALITY_ENABLED = True                       # STATUS: ACTIVE — data/loader.py; OHLCV quality checks
MAX_MISSING_BAR_FRACTION = 0.05                   # STATUS: ACTIVE — data/quality.py; max fraction of missing bars
MAX_ZERO_VOLUME_FRACTION = 0.25                   # STATUS: ACTIVE — data/quality.py; max fraction of zero-volume bars
MAX_ABS_DAILY_RETURN = 0.40                       # STATUS: ACTIVE — data/quality.py; max absolute single-day return

# ── Autopilot (discovery -> promotion -> paper trading) ─────────────────
AUTOPILOT_DIR = RESULTS_DIR / "autopilot"         # STATUS: ACTIVE — base path for autopilot output (used via derived paths below)
STRATEGY_REGISTRY_PATH = AUTOPILOT_DIR / "strategy_registry.json"  # STATUS: ACTIVE — autopilot/registry.py
PAPER_STATE_PATH = AUTOPILOT_DIR / "paper_state.json"  # STATUS: ACTIVE — autopilot/paper_trader.py
AUTOPILOT_CYCLE_REPORT = AUTOPILOT_DIR / "latest_cycle.json"  # STATUS: ACTIVE — autopilot/engine.py
AUTOPILOT_FEATURE_MODE = "core"                   # STATUS: ACTIVE — autopilot/engine.py; "core" (reduced) or "full"

DISCOVERY_ENTRY_MULTIPLIERS = [0.8, 1.0, 1.2]    # STATUS: ACTIVE — autopilot/strategy_discovery.py
DISCOVERY_CONFIDENCE_OFFSETS = [-0.10, 0.0, 0.10] # STATUS: ACTIVE — autopilot/strategy_discovery.py
DISCOVERY_RISK_VARIANTS = [False, True]           # STATUS: ACTIVE — autopilot/strategy_discovery.py
DISCOVERY_MAX_POSITIONS_VARIANTS = [10, 20]       # STATUS: ACTIVE — autopilot/strategy_discovery.py

PROMOTION_MIN_TRADES = 80                         # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WIN_RATE = 0.50                     # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_SHARPE = 0.75                       # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_PROFIT_FACTOR = 1.10                # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_DRAWDOWN = -0.20                    # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_ANNUAL_RETURN = 0.05                # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_ACTIVE_STRATEGIES = 5               # STATUS: ACTIVE — autopilot/registry.py
PROMOTION_REQUIRE_ADVANCED_CONTRACT = True         # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_DSR_PVALUE = 0.05                   # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_PBO = 0.45                          # STATUS: ACTIVE — autopilot/promotion_gate.py; tightened from 0.50 (Bailey et al. 2017)
PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED = True    # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_CAPACITY_UTILIZATION = 1.0           # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WF_OOS_CORR = 0.01                  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION = 0.60    # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_WF_IS_OOS_GAP = 0.20               # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_REGIME_POSITIVE_FRACTION = 0.50     # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MAX_WORST_EVENT_LOSS = -0.08      # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE = 0.50      # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MIN_REGIME_STABILITY = 0.40       # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_REQUIRE_STATISTICAL_TESTS = True        # STATUS: ACTIVE — autopilot/promotion_gate.py; require IC/FDR tests to pass
PROMOTION_REQUIRE_CPCV = True                     # STATUS: ACTIVE — autopilot/promotion_gate.py; require CPCV to pass
PROMOTION_REQUIRE_SPA = False                     # STATUS: ACTIVE — autopilot/promotion_gate.py; SPA is informational by default

# ── Kelly Sizing ──────────────────────────────────────────────────────
KELLY_FRACTION = 0.50                             # STATUS: ACTIVE — risk/position_sizer.py; half-Kelly (conservative default)
MAX_PORTFOLIO_DD = 0.20                           # STATUS: PLACEHOLDER — defined but never imported; max portfolio drawdown for governor
KELLY_PORTFOLIO_BLEND = 0.30                      # STATUS: PLACEHOLDER — defined but never imported; Kelly weight in composite blend
KELLY_BAYESIAN_ALPHA = 2.0                        # STATUS: PLACEHOLDER — defined but never imported; Beta prior alpha for win rate
KELLY_BAYESIAN_BETA = 2.0                         # STATUS: PLACEHOLDER — defined but never imported; Beta prior beta for win rate
KELLY_REGIME_CONDITIONAL = True                   # STATUS: PLACEHOLDER — defined but never imported; use regime-specific parameters

PAPER_INITIAL_CAPITAL = 1_000_000.0               # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_MAX_TOTAL_POSITIONS = 30                    # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_USE_KELLY_SIZING = True                     # STATUS: ACTIVE — autopilot/paper_trader.py, api/services/backtest_service.py
PAPER_KELLY_FRACTION = 0.50                       # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_LOOKBACK_TRADES = 200                 # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_MIN_SIZE_MULTIPLIER = 0.25            # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_MAX_SIZE_MULTIPLIER = 1.50            # STATUS: ACTIVE — autopilot/paper_trader.py

# ── Feature Profiles ────────────────────────────────────────────────────
FEATURE_MODE_DEFAULT = "core"                     # STATUS: ACTIVE — run_backtest.py, run_train.py, run_predict.py; "full" or "core"

# ── Regime Trade Gating ──────────────────────────────────────────────
REGIME_2_TRADE_ENABLED = False                    # STATUS: ACTIVE — backtest/engine.py, api/routers/signals.py; suppress mean-revert regime entries (Sharpe -0.91)
REGIME_2_SUPPRESSION_MIN_CONFIDENCE = 0.5         # STATUS: ACTIVE — backtest/engine.py; only suppress when confidence exceeds this

# ── Regime Strategy Allocation ──────────────────────────────────────
REGIME_STRATEGY_ALLOCATION_ENABLED = True            # STATUS: ACTIVE — autopilot/strategy_allocator.py; adapt parameters by regime

# ── Drawdown Tiers ──────────────────────────────────────────────────
DRAWDOWN_WARNING_THRESHOLD = -0.05                # STATUS: ACTIVE — risk/stop_loss.py; -5% drawdown: reduce sizing 50%
DRAWDOWN_CAUTION_THRESHOLD = -0.10                # STATUS: ACTIVE — risk/stop_loss.py; -10% drawdown: no new entries, 25% sizing
DRAWDOWN_CRITICAL_THRESHOLD = -0.15               # STATUS: ACTIVE — risk/stop_loss.py; -15% drawdown: force liquidate
DRAWDOWN_DAILY_LOSS_LIMIT = -0.03                 # STATUS: ACTIVE — risk/stop_loss.py; -3% daily loss limit
DRAWDOWN_WEEKLY_LOSS_LIMIT = -0.05                # STATUS: ACTIVE — risk/stop_loss.py; -5% weekly loss limit
DRAWDOWN_RECOVERY_DAYS = 10                       # STATUS: ACTIVE — risk/stop_loss.py; days to return to full sizing after recovery
DRAWDOWN_SIZE_MULT_WARNING = 0.50                 # STATUS: ACTIVE — risk/stop_loss.py; size multiplier during WARNING tier
DRAWDOWN_SIZE_MULT_CAUTION = 0.25                 # STATUS: ACTIVE — risk/stop_loss.py; size multiplier during CAUTION tier

# ── Covariance ────────────────────────────────────────────────────
COVARIANCE_HALF_LIFE = 60                         # STATUS: ACTIVE — risk/covariance.py; EWMA half-life in trading days (60 ≈ 3 months)

# ── Stop Loss ───────────────────────────────────────────────────────
STOP_LOSS_SPREAD_BUFFER_BPS = 3.0                 # STATUS: ACTIVE — risk/stop_loss.py; bid-ask spread buffer for stop prices (bps)
HARD_STOP_PCT = -0.08                             # STATUS: ACTIVE — risk/stop_loss.py; -8% hard stop loss
ATR_STOP_MULTIPLIER = 2.0                         # STATUS: ACTIVE — risk/stop_loss.py; initial stop at 2x ATR
TRAILING_ATR_MULTIPLIER = 1.5                     # STATUS: ACTIVE — risk/stop_loss.py; trailing stop at 1.5x ATR
TRAILING_ACTIVATION_PCT = 0.02                    # STATUS: ACTIVE — risk/stop_loss.py; activate trailing stop after +2% gain
MAX_HOLDING_DAYS = 30                             # STATUS: ACTIVE — risk/stop_loss.py, backtest/engine.py; time-based stop at 30 days

# ── Almgren-Chriss Parameters ─────────────────────────────────────
ALMGREN_CHRISS_FALLBACK_VOL = 0.20                # STATUS: ACTIVE — backtest/engine.py; fallback annualized vol when realized unavailable

# ACTIVE — Almgren-Chriss optimal execution risk aversion.
# Higher values = more conservative execution (split orders more).
# Range: 1e-3 (passive) to 1e-1 (aggressive risk aversion).
# Default 0.01 matches moderate institutional execution.
# Academic literature: passive institutional 1e-3..1e-2, active 1e-2..1e-1.
ALMGREN_CHRISS_RISK_AVERSION = 0.01               # STATUS: ACTIVE — backtest/optimal_execution.py; risk aversion lambda

# ── Model Governance ────────────────────────────────────────────────
GOVERNANCE_SCORE_WEIGHTS = {                       # STATUS: ACTIVE — models/governance.py; champion/challenger scoring weights
    "oos_spearman": 1.5,      # Weight for OOS Spearman correlation
    "holdout_spearman": 1.0,  # Weight for holdout Spearman correlation
    "cv_gap_penalty": -0.5,   # Penalty weight for IS-OOS gap
}

# ── Validation ──────────────────────────────────────────────────────
IC_ROLLING_WINDOW = 60                            # STATUS: ACTIVE — backtest/validation.py; rolling window for Information Coefficient

# ── Alert System ──────────────────────────────────────────────────────
ALERT_WEBHOOK_URL = ""                            # STATUS: ACTIVE — utils/logging.py; empty = disabled; set to Slack/Discord webhook URL
ALERT_HISTORY_FILE = RESULTS_DIR / "alerts_history.json"  # STATUS: ACTIVE — utils/logging.py; alert history persistence

# ── Log Configuration ──────────────────────────────────────────────
LOG_LEVEL = "INFO"                                # STATUS: ACTIVE — api/main.py; "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FORMAT = "structured"                         # STATUS: ACTIVE — api/main.py; "structured" or "json"


# ── Config Validation ──────────────────────────────────────────────

def validate_config() -> list:
    """Check config for common misconfigurations.

    Returns a list of dicts: [{"level": "WARNING"|"ERROR", "message": str}].
    Called on server startup and available via /api/config/validate.
    """
    import os

    issues = []

    # 1. GICS_SECTORS empty → sector constraints disabled
    if not GICS_SECTORS:
        issues.append({
            "level": "WARNING",
            "message": (
                "GICS_SECTORS is empty — sector exposure constraints are disabled. "
                "Populate via run_wrds_daily_refresh.py --gics or manually in config.py."
            ),
        })

    # 2. OPTIONMETRICS_ENABLED but pipeline incomplete
    if OPTIONMETRICS_ENABLED:
        issues.append({
            "level": "WARNING",
            "message": (
                "OPTIONMETRICS_ENABLED=True but the OptionMetrics pipeline is not fully wired. "
                "Set to False unless the IV surface data source is configured."
            ),
        })

    # 3. WRDS_ENABLED but credentials missing
    if WRDS_ENABLED:
        wrds_user = os.environ.get("WRDS_USERNAME", "")
        if not wrds_user:
            issues.append({
                "level": "WARNING",
                "message": (
                    "WRDS_ENABLED=True but WRDS_USERNAME env var is not set. "
                    "WRDS will be unavailable. Set via: export WRDS_USERNAME=<your_username>"
                ),
            })

    # 4. DATA_CACHE_DIR doesn't exist or is empty
    if not DATA_CACHE_DIR.exists():
        issues.append({
            "level": "WARNING",
            "message": f"DATA_CACHE_DIR ({DATA_CACHE_DIR}) does not exist. No cached data available.",
        })
    elif not any(DATA_CACHE_DIR.glob("*.parquet")):
        issues.append({
            "level": "WARNING",
            "message": f"DATA_CACHE_DIR ({DATA_CACHE_DIR}) exists but contains no .parquet files.",
        })

    # 5. MODEL_DIR doesn't exist or has no models
    if not MODEL_DIR.exists():
        issues.append({
            "level": "WARNING",
            "message": f"MODEL_DIR ({MODEL_DIR}) does not exist. No trained models available.",
        })
    elif not any(MODEL_DIR.glob("*.pkl")) and not any(MODEL_DIR.glob("*.json")):
        issues.append({
            "level": "WARNING",
            "message": f"MODEL_DIR ({MODEL_DIR}) exists but has no model files (.pkl/.json).",
        })

    # 6. REGIME_MODEL_TYPE not valid
    valid_regime_types = ("hmm", "rule", "jump")
    if REGIME_MODEL_TYPE not in valid_regime_types:
        issues.append({
            "level": "ERROR",
            "message": (
                f"REGIME_MODEL_TYPE='{REGIME_MODEL_TYPE}' is invalid. "
                f"Must be one of: {valid_regime_types}"
            ),
        })

    return issues
