"""
Central configuration for the quant engine.

Self-contained — no references to automated_portfolio_system.
"""
from pathlib import Path
from typing import Dict

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
FRAMEWORK_DIR = ROOT_DIR.parent
MODEL_DIR = ROOT_DIR / "trained_models"
RESULTS_DIR = ROOT_DIR / "results"

# ── Data Sources ──────────────────────────────────────────────────────
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"
WRDS_ENABLED = True  # Try WRDS first; fall back to local cache / IBKR
OPTIONMETRICS_ENABLED = True  # Optional WRDS OptionMetrics enrichment
KALSHI_ENABLED = False  # Enable Kalshi event-market ingestion pipeline
KALSHI_ENV = "demo"  # "demo" (default safety) or "prod"
KALSHI_DEMO_API_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_BASE_URL = KALSHI_DEMO_API_BASE_URL
KALSHI_HISTORICAL_API_BASE_URL = KALSHI_API_BASE_URL
KALSHI_HISTORICAL_CUTOFF_TS = None
KALSHI_RATE_LIMIT_RPS = 6.0
KALSHI_RATE_LIMIT_BURST = 2
KALSHI_DB_PATH = ROOT_DIR / "data" / "kalshi.duckdb"
KALSHI_SNAPSHOT_HORIZONS = ["7d", "1d", "4h", "1h", "15m", "5m"]
KALSHI_DISTRIBUTION_FREQ = "5min"
KALSHI_STALE_AFTER_MINUTES = 30
KALSHI_NEAR_EVENT_MINUTES = 30.0
KALSHI_NEAR_EVENT_STALE_MINUTES = 2.0
KALSHI_FAR_EVENT_MINUTES = 24.0 * 60.0
KALSHI_FAR_EVENT_STALE_MINUTES = 60.0
KALSHI_STALE_MARKET_TYPE_MULTIPLIERS = {
    "CPI": 0.80,
    "UNEMPLOYMENT": 0.90,
    "FOMC": 0.70,
    "_default": 1.00,
}
KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD = 2.0
KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD = 6.0
KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER = 1.35
KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER = 0.80
KALSHI_DISTANCE_LAGS = ["1h", "1d"]
KALSHI_TAIL_THRESHOLDS = {
    "CPI": [3.0, 3.5, 4.0],
    "UNEMPLOYMENT": [4.0, 4.2, 4.5],
    "FOMC": [0.0, 25.0, 50.0],
    "_default": [0.0, 0.5, 1.0],
}
DEFAULT_UNIVERSE_SOURCE = "wrds"  # "wrds", "static", or "ibkr"
CACHE_TRUSTED_SOURCES = ["wrds", "wrds_delisting", "ibkr"]
CACHE_MAX_STALENESS_DAYS = 21
CACHE_WRDS_SPAN_ADVANTAGE_DAYS = 180
REQUIRE_PERMNO = True

# ── Survivorship ──────────────────────────────────────────────────────
SURVIVORSHIP_DB = ROOT_DIR / "data" / "universe_history.db"
SURVIVORSHIP_UNIVERSE_NAME = "SP500"
SURVIVORSHIP_SNAPSHOT_FREQ = "quarterly"  # "annual" or "quarterly"

# ── Model Versioning ─────────────────────────────────────────────────
MODEL_REGISTRY = MODEL_DIR / "registry.json"
MAX_MODEL_VERSIONS = 5  # Keep last 5 versions for rollback
CHAMPION_REGISTRY = MODEL_DIR / "champion_registry.json"

# ── Retraining ───────────────────────────────────────────────────────
RETRAIN_MAX_DAYS = 30
RETRAIN_MIN_TRADES = 50
RETRAIN_MIN_WIN_RATE = 0.45
RETRAIN_MIN_CORRELATION = 0.05  # Minimum OOS Spearman
RETRAIN_REGIME_CHANGE_DAYS = 10  # Trigger retrain if regime changed for 10+ consecutive days
RECENCY_DECAY = 0.003  # λ for exponential recency weighting (1yr weight ≈ 0.33)

# ── Universe ───────────────────────────────────────────────────────────
UNIVERSE_FULL = [
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

UNIVERSE_QUICK = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA",
    "JPM", "UNH", "HD", "V", "DDOG", "CRWD", "CAVA",
]

BENCHMARK = "SPY"

# ── Data ───────────────────────────────────────────────────────────────
LOOKBACK_YEARS = 15
MIN_BARS = 500  # minimum bars needed for feature warm-up

# ── Intraday Data ─────────────────────────────────────────────────
INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]
INTRADAY_CACHE_SOURCE = "ibkr"
INTRADAY_MIN_BARS = 100
MARKET_OPEN = "09:30"   # US equity regular-session open (ET)
MARKET_CLOSE = "16:00"  # US equity regular-session close (ET)

# ── Targets ────────────────────────────────────────────────────────────
FORWARD_HORIZONS = [5, 10, 20]  # days ahead to predict

# ── Features ───────────────────────────────────────────────────────────
INTERACTION_PAIRS = [
    # (feature_a, feature_b, operation)
    # Regime-conditional signals
    ("RSI_14", "Hurst_100", "multiply"),
    ("ZScore_20", "VolTS_10_60", "multiply"),
    ("MACD_12_26", "AutoCorr_20_1", "multiply"),
    ("RSI_14", "VarRatio_100_5", "multiply"),
    # Volatility × liquidity
    ("NATR_14", "Amihud_20", "multiply"),
    # Predictability × trend strength
    ("Entropy_20", "ADX_14", "multiply"),
    # Vol forecast × tail risk
    ("GARCH_252", "Skew_60", "multiply"),
    # Trend quality × noise
    ("Hurst_100", "FracDim_100", "multiply"),
    # Vol estimator divergence
    ("ParkVol_20", "GKVol_20", "ratio"),
    # Centered indicators
    ("RSI_14", None, "center_50"),
    ("Stoch_14", None, "center_50"),
]

# ── Regime ─────────────────────────────────────────────────────────────
REGIME_NAMES = {
    0: "trending_bull",
    1: "trending_bear",
    2: "mean_reverting",
    3: "high_volatility",
}
MIN_REGIME_SAMPLES = 500  # minimum training samples per regime model
REGIME_MODEL_TYPE = "hmm"  # "hmm" or "rule"
REGIME_HMM_STATES = 4
REGIME_HMM_MAX_ITER = 60
REGIME_HMM_STICKINESS = 0.92
REGIME_MIN_DURATION = 3
REGIME_SOFT_ASSIGNMENT_THRESHOLD = 0.35
REGIME_HMM_PRIOR_WEIGHT = 0.3  # Shrinkage weight toward sticky prior in HMM M-step
REGIME_HMM_COVARIANCE_TYPE = "full"  # "full" (captures return-vol correlation) or "diag"
REGIME_HMM_AUTO_SELECT_STATES = True  # Use BIC to select optimal number of states
REGIME_HMM_MIN_STATES = 2
REGIME_HMM_MAX_STATES = 6

# ── Kalshi Purge/Embargo by Event Type (E3) ─────────────────────────────
KALSHI_PURGE_WINDOW_BY_EVENT = {"CPI": 14, "FOMC": 21, "NFP": 14, "GDP": 14}
KALSHI_DEFAULT_PURGE_WINDOW = 10

# ── Model ──────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 30,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "max_features": "sqrt",
}
MAX_FEATURES_SELECTED = 30  # after permutation importance
MAX_IS_OOS_GAP = 0.05  # max allowed degradation (R² or correlation)
CV_FOLDS = 5
HOLDOUT_FRACTION = 0.15
ENSEMBLE_DIVERSIFY = True  # Train GBR + ElasticNet + RandomForest and average predictions

# ── Walk-Forward Rolling Window ───────────────────────────────────────
# When set, training windows are capped at this many unique dates so old
# data rolls off.  ``None`` means expanding windows (all history).
# A typical value of 1260 ≈ 5 years of trading days.
WF_MAX_TRAIN_DATES = 1260  # ~5 years of trading days; rolling walk-forward window

# ── Backtest ───────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 20  # 20 bps round-trip
ENTRY_THRESHOLD = 0.005  # minimum predicted return to enter (0.5%)
CONFIDENCE_THRESHOLD = 0.6  # minimum model confidence
MAX_POSITIONS = 20
POSITION_SIZE_PCT = 0.05  # 5% of capital per position
BACKTEST_ASSUMED_CAPITAL_USD = 1_000_000.0
EXEC_SPREAD_BPS = 3.0
EXEC_MAX_PARTICIPATION = 0.02
EXEC_IMPACT_COEFF_BPS = 25.0
EXEC_MIN_FILL_RATIO = 0.20
EXEC_DYNAMIC_COSTS = True
EXEC_DOLLAR_VOLUME_REF_USD = 25_000_000.0
EXEC_VOL_REF = 0.20
EXEC_VOL_SPREAD_BETA = 1.0
EXEC_GAP_SPREAD_BETA = 4.0
EXEC_RANGE_SPREAD_BETA = 2.0
EXEC_VOL_IMPACT_BETA = 1.0
MAX_PORTFOLIO_VOL = 0.30
REGIME_RISK_MULTIPLIER = {
    0: 1.00,  # trending_bull
    1: 0.85,  # trending_bear
    2: 0.95,  # mean_reverting
    3: 0.60,  # high_volatility
}
REGIME_STOP_MULTIPLIER = {
    0: 1.0,   # trending_bull: standard stops
    1: 0.8,   # trending_bear: tighter stops (cut losses faster)
    2: 1.2,   # mean_reverting: wider stops (expect reversals)
    3: 1.5,   # high_volatility: wider stops (avoid noise stops)
}
MAX_ANNUALIZED_TURNOVER = 500.0  # 500% annualized turnover warning threshold
MAX_SECTOR_EXPOSURE = 0.10  # Maximum net weight in any single GICS sector (±10%)
GICS_SECTORS: Dict[str, str] = {}
# Maps permno -> GICS sector name for sector-neutrality constraints in the
# portfolio optimizer.  Populate from WRDS Compustat ``comp.company`` table:
#
#   SELECT gvkey, gsubind AS gics_code, conm
#   FROM comp.company
#   WHERE gsubind IS NOT NULL;
#
# Then map the 2-digit sector code to a human-readable name.  When empty,
# sector constraints in optimize_portfolio() are silently skipped.

# ── Almgren-Chriss Optimal Execution ─────────────────────────────────
ALMGREN_CHRISS_ENABLED = True   # Use AC model for large positions
ALMGREN_CHRISS_ADV_THRESHOLD = 0.05  # Positions > 5% of ADV use AC cost model

# ── Validation ─────────────────────────────────────────────────────────
CPCV_PARTITIONS = 8
CPCV_TEST_PARTITIONS = 4
SPA_BOOTSTRAPS = 400

# ── Data Quality ───────────────────────────────────────────────────────
DATA_QUALITY_ENABLED = True
MAX_MISSING_BAR_FRACTION = 0.05
MAX_ZERO_VOLUME_FRACTION = 0.25
MAX_ABS_DAILY_RETURN = 0.40

# ── Autopilot (discovery -> promotion -> paper trading) ─────────────────
AUTOPILOT_DIR = RESULTS_DIR / "autopilot"
STRATEGY_REGISTRY_PATH = AUTOPILOT_DIR / "strategy_registry.json"
PAPER_STATE_PATH = AUTOPILOT_DIR / "paper_state.json"
AUTOPILOT_CYCLE_REPORT = AUTOPILOT_DIR / "latest_cycle.json"
AUTOPILOT_FEATURE_MODE = "core"  # "core" (reduced complexity) or "full"

DISCOVERY_ENTRY_MULTIPLIERS = [0.8, 1.0, 1.2]
DISCOVERY_CONFIDENCE_OFFSETS = [-0.10, 0.0, 0.10]
DISCOVERY_RISK_VARIANTS = [False, True]
DISCOVERY_MAX_POSITIONS_VARIANTS = [10, 20]

PROMOTION_MIN_TRADES = 80
PROMOTION_MIN_WIN_RATE = 0.50
PROMOTION_MIN_SHARPE = 0.75
PROMOTION_MIN_PROFIT_FACTOR = 1.10
PROMOTION_MAX_DRAWDOWN = -0.20
PROMOTION_MIN_ANNUAL_RETURN = 0.05
PROMOTION_MAX_ACTIVE_STRATEGIES = 5
PROMOTION_REQUIRE_ADVANCED_CONTRACT = True
PROMOTION_MAX_DSR_PVALUE = 0.05
PROMOTION_MAX_PBO = 0.50
PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED = True
PROMOTION_MAX_CAPACITY_UTILIZATION = 1.0
PROMOTION_MIN_WF_OOS_CORR = 0.01
PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION = 0.60
PROMOTION_MAX_WF_IS_OOS_GAP = 0.20
PROMOTION_MIN_REGIME_POSITIVE_FRACTION = 0.50
PROMOTION_EVENT_MAX_WORST_EVENT_LOSS = -0.08
PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE = 0.50
PROMOTION_EVENT_MIN_REGIME_STABILITY = 0.40
PROMOTION_REQUIRE_STATISTICAL_TESTS = True  # Require IC/FDR tests to pass
PROMOTION_REQUIRE_CPCV = True               # Require CPCV to pass
PROMOTION_REQUIRE_SPA = False               # SPA is informational by default (strict = True)

# ── Kelly Sizing ──────────────────────────────────────────────────────
KELLY_FRACTION = 0.50                   # Half-Kelly (conservative default)
MAX_PORTFOLIO_DD = 0.20                 # Max portfolio drawdown for governor
KELLY_PORTFOLIO_BLEND = 0.30            # Kelly weight in composite blend
KELLY_BAYESIAN_ALPHA = 2.0              # Beta prior alpha for win rate
KELLY_BAYESIAN_BETA = 2.0               # Beta prior beta for win rate
KELLY_REGIME_CONDITIONAL = True         # Use regime-specific parameters

PAPER_INITIAL_CAPITAL = 1_000_000.0
PAPER_MAX_TOTAL_POSITIONS = 30
PAPER_USE_KELLY_SIZING = True
PAPER_KELLY_FRACTION = 0.50
PAPER_KELLY_LOOKBACK_TRADES = 200
PAPER_KELLY_MIN_SIZE_MULTIPLIER = 0.25
PAPER_KELLY_MAX_SIZE_MULTIPLIER = 1.50

# ── Feature Profiles ────────────────────────────────────────────────────
FEATURE_MODE_DEFAULT = "core"  # "full" or "core"

# ── Regime Trade Gating ──────────────────────────────────────────────
REGIME_2_TRADE_ENABLED = False  # Model has Sharpe -0.91 in mean-revert regime; suppress entries
REGIME_2_SUPPRESSION_MIN_CONFIDENCE = 0.5  # Only suppress when confidence exceeds this

# ── Drawdown Tiers ──────────────────────────────────────────────────
DRAWDOWN_WARNING_THRESHOLD = -0.05   # -5% drawdown: reduce sizing 50%
DRAWDOWN_CAUTION_THRESHOLD = -0.10   # -10% drawdown: no new entries, 25% sizing
DRAWDOWN_CRITICAL_THRESHOLD = -0.15  # -15% drawdown: force liquidate
DRAWDOWN_DAILY_LOSS_LIMIT = -0.03    # -3% daily loss limit
DRAWDOWN_WEEKLY_LOSS_LIMIT = -0.05   # -5% weekly loss limit
DRAWDOWN_RECOVERY_DAYS = 10          # Days to return to full sizing after recovery
DRAWDOWN_SIZE_MULT_WARNING = 0.50    # Size multiplier during WARNING tier
DRAWDOWN_SIZE_MULT_CAUTION = 0.25    # Size multiplier during CAUTION tier

# ── Stop Loss ───────────────────────────────────────────────────────
HARD_STOP_PCT = -0.08               # -8% hard stop loss
ATR_STOP_MULTIPLIER = 2.0           # Initial stop at 2x ATR
TRAILING_ATR_MULTIPLIER = 1.5       # Trailing stop at 1.5x ATR
TRAILING_ACTIVATION_PCT = 0.02      # Activate trailing stop after +2% gain
MAX_HOLDING_DAYS = 30               # Time-based stop at 30 days

# ── Almgren-Chriss ──────────────────────────────────────────────────
ALMGREN_CHRISS_FALLBACK_VOL = 0.20  # Fallback annualized vol when realized unavailable

# ── Model Governance ────────────────────────────────────────────────
GOVERNANCE_SCORE_WEIGHTS = {
    "oos_spearman": 1.5,      # Weight for OOS Spearman correlation
    "holdout_spearman": 1.0,  # Weight for holdout Spearman correlation
    "cv_gap_penalty": -0.5,   # Penalty weight for IS-OOS gap
}

# ── Validation ──────────────────────────────────────────────────────
IC_ROLLING_WINDOW = 60  # Rolling window for Information Coefficient calculation

# ── Alert System ──────────────────────────────────────────────────────
ALERT_WEBHOOK_URL = ""  # Empty = disabled; set to Slack/Discord/custom webhook URL
ALERT_HISTORY_FILE = RESULTS_DIR / "alerts_history.json"
