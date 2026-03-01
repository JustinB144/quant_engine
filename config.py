"""
Central configuration for the quant engine.

Backward-compatible flat-constant interface.  All values that overlap
with ``config_structured.py`` are derived from the structured config
singleton so there is a single source of truth.  Constants that exist
only in this module (e.g. KALSHI_*, STRUCTURAL_*, universes, paths)
are defined here and will be migrated to structured config over time.

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

try:
    from .config_structured import get_config as _get_config
except ImportError:
    from config_structured import get_config as _get_config

_cfg = _get_config()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent                  # STATUS: ACTIVE — base path for all relative references
FRAMEWORK_DIR = ROOT_DIR.parent                   # STATUS: ACTIVE — parent directory of quant_engine
MODEL_DIR = ROOT_DIR / "trained_models"           # STATUS: ACTIVE — models/versioning.py, models/trainer.py
RESULTS_DIR = ROOT_DIR / "results"                # STATUS: ACTIVE — backtest output, autopilot reports, alerts

# ── Data Sources ──────────────────────────────────────────────────────
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"      # STATUS: ACTIVE — data/loader.py, data/local_cache.py
DATA_CACHE_ALPACA_DIR = ROOT_DIR / "data" / "cache_alpaca"  # STATUS: ACTIVE — archived Alpaca intraday data; scripts/alpaca_intraday_download.py
WRDS_ENABLED = _cfg.data.wrds_enabled             # STATUS: ACTIVE — data/loader.py; try WRDS first, fall back to local cache / IBKR

# PLACEHOLDER — OptionMetrics IV surface data integration.
# Set to True once api/routers/iv_surface.py has a working /iv-surface/heston endpoint
# and data loader merges OptionMetrics surface into OHLCV panels.
# Note: data/loader.py does gate on this flag (lines 296, 625), but the upstream
# OptionMetrics data source is not configured in most environments, so the try/except
# blocks silently fall through.  Disabled until the full pipeline is verified.
OPTIONMETRICS_ENABLED = _cfg.data.optionmetrics_enabled  # STATUS: PLACEHOLDER — data/loader.py gates on this but pipeline incomplete

# ── Execution Contract (Truth Layer) ─────────────────────────────────
RET_TYPE = _cfg.preconditions.ret_type.value      # STATUS: ACTIVE — "log" (log returns) or "simple" (pct returns)
LABEL_H = _cfg.preconditions.label_h              # STATUS: ACTIVE — label horizon in trading days
PX_TYPE = _cfg.preconditions.px_type.value        # STATUS: ACTIVE — "close" or "open" for price baseline
ENTRY_PRICE_TYPE = _cfg.preconditions.entry_price_type.value  # STATUS: ACTIVE — "next_bar_open" (no look-ahead)

# ── Truth Layer Feature Flags ────────────────────────────────────────
TRUTH_LAYER_STRICT_PRECONDITIONS = True           # STATUS: ACTIVE — raise on invalid execution contract
TRUTH_LAYER_FAIL_ON_CORRUPT = True                # STATUS: ACTIVE — block corrupt OHLCV from pipeline
TRUTH_LAYER_ENFORCE_CAUSALITY = True              # STATUS: ACTIVE — enforce feature causality at runtime
TRUTH_LAYER_COMPUTE_NULL_BASELINES = False        # STATUS: ACTIVE — compute null baselines per backtest (adds ~4x time)
TRUTH_LAYER_COST_STRESS_ENABLED = _cfg.cost_stress.enabled  # STATUS: ACTIVE — run cost stress sweep per backtest (adds ~4x time)

# ── Cost Stress Testing ──────────────────────────────────────────────
COST_STRESS_MULTIPLIERS = list(_cfg.cost_stress.multipliers)  # STATUS: ACTIVE — cost sweep factors

KALSHI_ENABLED = _cfg.data.kalshi_enabled          # STATUS: ACTIVE — kalshi/provider.py, run_kalshi_event_pipeline.py; disabled by design
KALSHI_ENV = "demo"                               # STATUS: ACTIVE — selects demo vs prod API URL; "demo" (safety) or "prod"
KALSHI_DEMO_API_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"   # STATUS: ACTIVE — used to compute KALSHI_API_BASE_URL
KALSHI_PROD_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"  # STATUS: ACTIVE — used when KALSHI_ENV="prod"
KALSHI_API_BASE_URL = KALSHI_DEMO_API_BASE_URL    # STATUS: ACTIVE — kalshi/provider.py base URL
KALSHI_HISTORICAL_API_BASE_URL = KALSHI_API_BASE_URL  # STATUS: ACTIVE — kalshi historical data endpoint
KALSHI_HISTORICAL_CUTOFF_TS = None                # STATUS: ACTIVE — optional cutoff for historical fetches
KALSHI_RATE_LIMIT_RPS = 6.0                       # STATUS: ACTIVE — kalshi/provider.py rate limiter
KALSHI_RATE_LIMIT_BURST = 2                       # STATUS: ACTIVE — kalshi/provider.py burst allowance
KALSHI_DB_PATH = ROOT_DIR / "data" / "kalshi.duckdb"  # STATUS: ACTIVE — DuckDB path for Kalshi snapshots
KALSHI_SNAPSHOT_HORIZONS = ["7D", "1D", "4h", "1h", "15min", "5min"]  # STATUS: ACTIVE — kalshi/provider.py, run_kalshi_event_pipeline.py
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
KALSHI_DISTANCE_LAGS = ["1h", "1D"]               # STATUS: ACTIVE — kalshi/provider.py distance lag features
KALSHI_TAIL_THRESHOLDS = {                         # STATUS: ACTIVE — kalshi/provider.py tail-risk thresholds
    "CPI": [3.0, 3.5, 4.0],
    "UNEMPLOYMENT": [4.0, 4.2, 4.5],
    "FOMC": [0.0, 25.0, 50.0],
    "_default": [0.0, 0.5, 1.0],
}
DEFAULT_UNIVERSE_SOURCE = _cfg.data.default_universe_source  # STATUS: PLACEHOLDER — defined but never imported; "wrds", "static", or "ibkr"
CACHE_TRUSTED_SOURCES = list(_cfg.data.cache_trusted_sources)  # STATUS: ACTIVE — data/local_cache.py source ranking
CACHE_MAX_STALENESS_DAYS = _cfg.data.cache_max_staleness_days  # STATUS: ACTIVE — data/local_cache.py max cache age
CACHE_WRDS_SPAN_ADVANTAGE_DAYS = 180              # STATUS: ACTIVE — data/local_cache.py WRDS preference window
REQUIRE_PERMNO = True                             # STATUS: ACTIVE — data/loader.py, backtest/engine.py PERMNO validation

# ── Survivorship ──────────────────────────────────────────────────────
SURVIVORSHIP_DB = ROOT_DIR / "data" / "universe_history.db"  # STATUS: ACTIVE — data/survivorship.py, data/loader.py
SURVIVORSHIP_UNIVERSE_NAME = "SP500"              # STATUS: ACTIVE — data/loader.py, autopilot/engine.py
SURVIVORSHIP_SNAPSHOT_FREQ = "quarterly"           # STATUS: ACTIVE — data/loader.py; "annual" or "quarterly"

# ── Model Versioning ─────────────────────────────────────────────────
MODEL_REGISTRY = MODEL_DIR / "registry.json"      # STATUS: PLACEHOLDER — defined but never imported; CHAMPION_REGISTRY is used instead
MAX_MODEL_VERSIONS = _cfg.model.max_model_versions  # STATUS: ACTIVE — models/versioning.py; keep last 5 versions for rollback
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
LOOKBACK_YEARS = _cfg.data.lookback_years         # STATUS: ACTIVE — data/loader.py; years of historical data to load
MIN_BARS = _cfg.data.min_bars                     # STATUS: ACTIVE — data/loader.py; minimum bars needed for feature warm-up

# ── Intraday Data ─────────────────────────────────────────────────
INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]  # STATUS: PLACEHOLDER — defined but never imported
INTRADAY_CACHE_SOURCE = "ibkr"                    # STATUS: PLACEHOLDER — defined but never imported
INTRADAY_MIN_BARS = 100                           # STATUS: ACTIVE — features/pipeline.py; minimum intraday bars for feature computation
MARKET_OPEN = "09:30"                             # STATUS: ACTIVE — features/intraday.py; US equity regular-session open (ET)
MARKET_CLOSE = "16:00"                            # STATUS: ACTIVE — features/intraday.py; US equity regular-session close (ET)

# ── Intraday Data Integrity (SPEC_11) ──────────────────────────────
INTRADAY_VALIDATION_ENABLED = True                # STATUS: ACTIVE — scripts/alpaca_intraday_download.py; enable IBKR cross-validation
INTRADAY_CLOSE_TOLERANCE_PCT = 0.15               # STATUS: ACTIVE — data/cross_source_validator.py; max close price diff vs IBKR (%)
INTRADAY_OPEN_TOLERANCE_PCT = 0.20                # STATUS: ACTIVE — data/cross_source_validator.py; max open price diff vs IBKR (%)
INTRADAY_HIGHLOW_TOLERANCE_PCT = 0.25             # STATUS: ACTIVE — data/cross_source_validator.py; max H/L price diff vs IBKR (%)
INTRADAY_VOLUME_TOLERANCE_PCT = 5.0               # STATUS: ACTIVE — data/cross_source_validator.py; max volume diff vs IBKR (%)
INTRADAY_MAX_REJECTED_BAR_PCT = 5.0               # STATUS: ACTIVE — data/intraday_quality.py; quarantine if >N% bars rejected
INTRADAY_MAX_MISMATCH_RATE_PCT = 5.0              # STATUS: ACTIVE — data/cross_source_validator.py; quarantine if >N% bars mismatch
INTRADAY_VALIDATION_SAMPLE_WINDOWS = 10           # STATUS: ACTIVE — data/cross_source_validator.py; stratified date sampling windows
INTRADAY_VALIDATION_DAYS_PER_WINDOW = 2           # STATUS: ACTIVE — data/cross_source_validator.py; days sampled per window
INTRADAY_QUARANTINE_DIR = DATA_CACHE_DIR / "quarantine"  # STATUS: ACTIVE — data/intraday_quality.py; quarantined data location

# ── Targets ────────────────────────────────────────────────────────────
FORWARD_HORIZONS = list(_cfg.model.forward_horizons)  # STATUS: ACTIVE — models/trainer.py, run_*.py; days ahead to predict

# ── Feature Deduplication ──────────────────────────────────────────────
STRICT_FEATURE_DEDUP = False                       # STATUS: ACTIVE — features/pipeline.py; raise error on duplicate features instead of warning

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
REGIME_SUPPRESS_ID = 3                             # STATUS: ACTIVE — models/predictor.py; regime code to suppress (3 = high_volatility per REGIME_NAMES)
MIN_REGIME_SAMPLES = 50                           # STATUS: ACTIVE — models/trainer.py; minimum training samples per regime model (SPEC_10 T7: reduced from 100 to 50 for short regimes like crashes)
REGIME_MODEL_TYPE = _cfg.regime.model_type         # STATUS: ACTIVE — regime/detector.py; "jump", "hmm", or "rule"
REGIME_HMM_STATES = _cfg.regime.n_states           # STATUS: ACTIVE — regime/hmm.py; number of hidden states
REGIME_HMM_MAX_ITER = _cfg.regime.hmm_max_iter     # STATUS: ACTIVE — regime/hmm.py; EM iteration limit
REGIME_HMM_STICKINESS = _cfg.regime.hmm_stickiness  # STATUS: ACTIVE — regime/hmm.py; diagonal prior bias for sticky transitions
REGIME_MIN_DURATION = _cfg.regime.min_duration     # STATUS: ACTIVE — regime/detector.py; minimum regime duration in days
REGIME_SOFT_ASSIGNMENT_THRESHOLD = 0.35           # STATUS: ACTIVE — models/trainer.py; probability threshold for soft regime labels
REGIME_HMM_PRIOR_WEIGHT = _cfg.regime.hmm_prior_weight  # STATUS: ACTIVE — regime/hmm.py; shrinkage weight toward sticky prior in M-step
REGIME_HMM_COVARIANCE_TYPE = _cfg.regime.hmm_covariance_type  # STATUS: ACTIVE — regime/hmm.py; "full" (captures return-vol correlation) or "diag"
REGIME_HMM_AUTO_SELECT_STATES = _cfg.regime.hmm_auto_select_states  # STATUS: ACTIVE — regime/hmm.py; use BIC to select optimal number of states
REGIME_HMM_MIN_STATES = _cfg.regime.hmm_min_states  # STATUS: ACTIVE — regime/hmm.py; BIC state search lower bound
REGIME_HMM_MAX_STATES = _cfg.regime.hmm_max_states  # STATUS: ACTIVE — regime/hmm.py; BIC state search upper bound
REGIME_JUMP_MODEL_ENABLED = _cfg.regime.jump_model_enabled  # STATUS: ACTIVE — regime/detector.py; statistical jump model alongside HMM
REGIME_JUMP_PENALTY = _cfg.regime.jump_penalty     # STATUS: ACTIVE — regime/jump_model.py; jump penalty lambda (higher = fewer transitions)
REGIME_EXPECTED_CHANGES_PER_YEAR = _cfg.regime.expected_changes_per_year  # STATUS: ACTIVE — regime/jump_model.py; calibrate jump penalty from expected regime changes/yr
REGIME_ENSEMBLE_ENABLED = _cfg.regime.ensemble_enabled  # STATUS: ACTIVE — regime/detector.py; combine HMM + JM + rule-based via majority vote
REGIME_ENSEMBLE_CONSENSUS_THRESHOLD = _cfg.regime.ensemble_consensus_threshold  # STATUS: ACTIVE — regime/detector.py; require N of 3 methods to agree for transition

# PyPI jumpmodels package configuration
REGIME_JUMP_USE_PYPI_PACKAGE = True               # STATUS: ACTIVE — regime/jump_model_pypi.py; True=PyPI jumpmodels, False=legacy custom
REGIME_JUMP_CV_FOLDS = 5                          # STATUS: ACTIVE — regime/jump_model_pypi.py; time-series CV folds for lambda selection
REGIME_JUMP_LAMBDA_RANGE = (0.005, 0.15)          # STATUS: ACTIVE — regime/jump_model_pypi.py; search range for jump penalty
REGIME_JUMP_LAMBDA_STEPS = 20                     # STATUS: ACTIVE — regime/jump_model_pypi.py; grid points for lambda search
REGIME_JUMP_MAX_ITER = 50                         # STATUS: ACTIVE — regime/jump_model_pypi.py; coordinate descent iterations
REGIME_JUMP_TOL = 1e-6                            # STATUS: ACTIVE — regime/jump_model_pypi.py; convergence tolerance
REGIME_JUMP_USE_CONTINUOUS = True                  # STATUS: ACTIVE — regime/jump_model_pypi.py; continuous JM for soft probabilities
REGIME_JUMP_MODE_LOSS_WEIGHT = 0.1                # STATUS: ACTIVE — regime/jump_model_pypi.py; mode loss penalty (continuous JM)

# ── Regime Detection Upgrade (SPEC_10) ────────────────────────────────
# Confidence-weighted ensemble voting
REGIME_ENSEMBLE_DEFAULT_WEIGHTS = {               # STATUS: ACTIVE — regime/detector.py; default component weights before calibration
    "hmm": 0.5,
    "rule": 0.3,
    "jump": 0.2,
}
REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD = 0.40     # STATUS: ACTIVE — regime/detector.py; if max weighted vote < this, regime is uncertain
REGIME_ENSEMBLE_UNCERTAIN_FALLBACK = 3            # STATUS: ACTIVE — regime/detector.py; regime to assume when uncertain (3=high_volatility)

# Regime uncertainty gating
REGIME_UNCERTAINTY_ENTROPY_THRESHOLD = 0.50       # STATUS: ACTIVE — regime/uncertainty_gate.py; flag if normalized entropy > this
REGIME_UNCERTAINTY_STRESS_THRESHOLD = 0.80        # STATUS: ACTIVE — regime/uncertainty_gate.py; assume stress if entropy > this
REGIME_UNCERTAINTY_SIZING_MAP = {                 # STATUS: ACTIVE — regime/uncertainty_gate.py; entropy->sizing multiplier
    0.0: 1.0,
    0.5: 0.95,
    1.0: 0.85,
}
REGIME_UNCERTAINTY_MIN_MULTIPLIER = 0.80          # STATUS: ACTIVE — regime/uncertainty_gate.py; floor for sizing multiplier

# Cross-sectional regime consensus
REGIME_CONSENSUS_THRESHOLD = 0.80                 # STATUS: ACTIVE — regime/consensus.py; high confidence consensus threshold
REGIME_CONSENSUS_EARLY_WARNING = 0.60             # STATUS: ACTIVE — regime/consensus.py; early warning if consensus drops below
REGIME_CONSENSUS_DIVERGENCE_WINDOW = 20           # STATUS: ACTIVE — regime/consensus.py; window for trend detection
REGIME_CONSENSUS_DIVERGENCE_SLOPE = -0.01         # STATUS: ACTIVE — regime/consensus.py; slope threshold for divergence

# Online regime updating
REGIME_ONLINE_UPDATE_ENABLED = True               # STATUS: ACTIVE — regime/online_update.py; enable incremental HMM updates
REGIME_ONLINE_REFIT_DAYS = 30                     # STATUS: ACTIVE — regime/online_update.py; full refit every N days

# Expanded observation matrix
REGIME_EXPANDED_FEATURES_ENABLED = True           # STATUS: ACTIVE — regime/hmm.py; add spectral/SSA/BOCPD to observation matrix

# Regime training thresholds
MIN_REGIME_DAYS = 10                              # STATUS: ACTIVE — models/trainer.py; minimum days in regime before training

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

# ── Structural Sample Weighting (SPEC_03 T4) ──────────────────────────
STRUCTURAL_WEIGHT_ENABLED = _cfg.model.structural_weight_enabled            # STATUS: ACTIVE — models/trainer.py; enable structural-state-aware sample weighting
STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY = _cfg.model.structural_weight_changepoint_penalty  # STATUS: ACTIVE — models/trainer.py; max weight reduction near changepoints (0.0=none, 1.0=zero weight)
STRUCTURAL_WEIGHT_JUMP_PENALTY = _cfg.model.structural_weight_jump_penalty  # STATUS: ACTIVE — models/trainer.py; weight multiplier for jump events (0.0=exclude, 1.0=no penalty)
STRUCTURAL_WEIGHT_STRESS_PENALTY = _cfg.model.structural_weight_stress_penalty  # STATUS: ACTIVE — models/trainer.py; max weight reduction for high systemic stress

# ── Kalshi Purge/Embargo by Event Type (E3) ─────────────────────────────
KALSHI_PURGE_WINDOW_BY_EVENT = {"CPI": 3, "FOMC": 1, "UNEMPLOYMENT": 2, "GDP": 3}  # STATUS: ACTIVE — kalshi/events.py, kalshi/walkforward.py; event-type purge windows (days)
KALSHI_DEFAULT_PURGE_WINDOW = 2                    # STATUS: ACTIVE — kalshi/events.py, kalshi/walkforward.py; default purge window (days)

# ── Model ──────────────────────────────────────────────────────────────
MODEL_PARAMS = dict(_cfg.model.params)             # STATUS: ACTIVE — models/trainer.py; GBR hyperparameters
MAX_FEATURES_SELECTED = _cfg.model.max_features_selected  # STATUS: ACTIVE — models/trainer.py; after permutation importance
MAX_IS_OOS_GAP = _cfg.model.max_is_oos_gap         # STATUS: ACTIVE — models/trainer.py; max allowed IS-OOS degradation (R^2 or correlation)
CV_FOLDS = _cfg.model.cv_folds                     # STATUS: ACTIVE — models/trainer.py; cross-validation folds
HOLDOUT_FRACTION = _cfg.model.holdout_fraction     # STATUS: ACTIVE — models/trainer.py; holdout set fraction
ENSEMBLE_DIVERSIFY = _cfg.model.ensemble_diversify  # STATUS: ACTIVE — models/trainer.py; train GBR + ElasticNet + RandomForest and average

# ── Walk-Forward Rolling Window ───────────────────────────────────────
# When set, training windows are capped at this many unique dates so old
# data rolls off.  ``None`` means expanding windows (all history).
# A typical value of 1260 ~ 5 years of trading days.
WF_MAX_TRAIN_DATES = _cfg.backtest.wf_max_train_dates  # STATUS: ACTIVE — backtest/engine.py, models/trainer.py; rolling walk-forward window

# ── Backtest ───────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = _cfg.backtest.transaction_cost_bps  # STATUS: ACTIVE — backtest/engine.py; 20 bps round-trip
ENTRY_THRESHOLD = _cfg.backtest.entry_threshold    # STATUS: ACTIVE — backtest/engine.py; minimum predicted return to enter (0.5%)
CONFIDENCE_THRESHOLD = _cfg.backtest.confidence_threshold  # STATUS: ACTIVE — backtest/engine.py; minimum model confidence
MAX_POSITIONS = _cfg.backtest.max_positions         # STATUS: ACTIVE — backtest/engine.py; max simultaneous positions
POSITION_SIZE_PCT = _cfg.backtest.position_size_pct  # STATUS: ACTIVE — backtest/engine.py; 5% of capital per position
BACKTEST_ASSUMED_CAPITAL_USD = _cfg.backtest.assumed_capital_usd  # STATUS: ACTIVE — backtest/engine.py; initial capital
EXEC_SPREAD_BPS = _cfg.execution.spread_bps        # STATUS: ACTIVE — backtest/engine.py; base spread cost
EXEC_MAX_PARTICIPATION = _cfg.execution.max_participation  # STATUS: ACTIVE — backtest/engine.py; max 2% of daily volume
EXEC_IMPACT_COEFF_BPS = _cfg.execution.impact_coeff_bps  # STATUS: ACTIVE — backtest/engine.py; market impact coefficient
EXEC_MIN_FILL_RATIO = _cfg.execution.min_fill_ratio  # STATUS: ACTIVE — backtest/engine.py; minimum fill ratio
EXEC_DYNAMIC_COSTS = _cfg.execution.dynamic_costs   # STATUS: ACTIVE — backtest/engine.py; condition costs on market state
EXEC_DOLLAR_VOLUME_REF_USD = 25_000_000.0        # STATUS: ACTIVE — backtest/engine.py; dollar volume reference
EXEC_VOL_REF = 0.20                              # STATUS: ACTIVE — backtest/engine.py; reference volatility
EXEC_VOL_SPREAD_BETA = 1.0                       # STATUS: ACTIVE — backtest/engine.py; vol-spread sensitivity
EXEC_GAP_SPREAD_BETA = 4.0                       # STATUS: ACTIVE — backtest/engine.py; gap-spread sensitivity
EXEC_RANGE_SPREAD_BETA = 2.0                     # STATUS: ACTIVE — backtest/engine.py; range-spread sensitivity
EXEC_VOL_IMPACT_BETA = 1.0                       # STATUS: ACTIVE — backtest/engine.py; vol-impact sensitivity

# ── Structural State-Aware Costs (Spec 06) ────────────────────────────
EXEC_STRUCTURAL_STRESS_ENABLED = True             # STATUS: ACTIVE — backtest/execution.py; enable structural state cost multipliers
EXEC_BREAK_PROB_COST_MULT = {                     # STATUS: ACTIVE — backtest/execution.py; break probability → cost multiplier tiers
    "low": 1.0,                                   #   break_prob < 0.05
    "medium": 1.3,                                #   0.05 <= break_prob < 0.15
    "high": 2.0,                                  #   break_prob >= 0.15
}
EXEC_STRUCTURE_UNCERTAINTY_COST_MULT = 0.50       # STATUS: ACTIVE — backtest/execution.py; +50% per 1.0 increase in uncertainty
EXEC_DRIFT_SCORE_COST_REDUCTION = 0.20            # STATUS: ACTIVE — backtest/execution.py; -20% per 1.0 increase in drift (strong trend = cheaper)
EXEC_SYSTEMIC_STRESS_COST_MULT = 0.30             # STATUS: ACTIVE — backtest/execution.py; +30% per 1.0 increase in systemic stress

# ── ADV Computation (Spec 06) ─────────────────────────────────────────
ADV_LOOKBACK_DAYS = 20                            # STATUS: ACTIVE — backtest/adv_tracker.py; window for ADV calculation
ADV_EMA_SPAN = 20                                 # STATUS: ACTIVE — backtest/adv_tracker.py; EMA smoothing parameter
EXEC_VOLUME_TREND_ENABLED = True                  # STATUS: ACTIVE — backtest/execution.py; adjust costs based on volume trend
EXEC_LOW_VOLUME_COST_MULT = 1.5                   # STATUS: ACTIVE — backtest/adv_tracker.py; +50% on below-average volume days

# ── Entry/Exit Urgency (Spec 06) ─────────────────────────────────────
EXEC_EXIT_URGENCY_COST_LIMIT_MULT = 1.5           # STATUS: ACTIVE — backtest/execution.py; exits tolerate 1.5x higher costs
EXEC_ENTRY_URGENCY_COST_LIMIT_MULT = 1.0          # STATUS: ACTIVE — backtest/execution.py; entries use standard cost limits
EXEC_STRESS_PULLBACK_MIN_SIZE = 0.10              # STATUS: ACTIVE — backtest/execution.py; reduce order size by 10% per urgency level

# ── Cost Calibration (Spec 06) ────────────────────────────────────────
EXEC_CALIBRATION_ENABLED = True                   # STATUS: ACTIVE — backtest/cost_calibrator.py; enable per-segment calibration
EXEC_CALIBRATION_MIN_TRADES = 100                 # STATUS: ACTIVE — backtest/cost_calibrator.py; min trades before calibration
EXEC_CALIBRATION_MIN_SEGMENT_TRADES = 20          # STATUS: ACTIVE — backtest/cost_calibrator.py; min trades per segment
EXEC_CALIBRATION_SMOOTHING = 0.30                 # STATUS: ACTIVE — backtest/cost_calibrator.py; new-coefficient weight in EMA update
EXEC_COST_IMPACT_COEFF_BY_MARKETCAP = {           # STATUS: ACTIVE — backtest/cost_calibrator.py; default coefficients by segment
    "micro": 40.0,                                #   market_cap < 300M
    "small": 30.0,                                #   300M - 2B
    "mid": 20.0,                                  #   2B - 10B
    "large": 15.0,                                #   > 10B
}
EXEC_MARKETCAP_THRESHOLDS = {                     # STATUS: ACTIVE — backtest/cost_calibrator.py; segment boundaries in USD
    "micro": 300e6,
    "small": 2e9,
    "mid": 10e9,
}

# ── Calibration Feedback Loop (SPEC-E04) ─────────────────────────────
EXEC_CALIBRATION_FEEDBACK_ENABLED = True           # STATUS: ACTIVE — backtest/cost_calibrator.py; enable predicted-vs-actual feedback loop
EXEC_CALIBRATION_FEEDBACK_INTERVAL_DAYS = 30       # STATUS: ACTIVE — backtest/cost_calibrator.py; recalibration cadence in calendar days
EXEC_CALIBRATION_FEEDBACK_PATH = MODEL_DIR / "cost_calibration_feedback.json"  # STATUS: ACTIVE — backtest/cost_calibrator.py; persisted fill history

# ── No-Trade Gate (Spec 06) ──────────────────────────────────────────
EXEC_NO_TRADE_STRESS_THRESHOLD = 0.95             # STATUS: ACTIVE — backtest/execution.py; VIX percentile above which low-urgency orders blocked

MAX_PORTFOLIO_VOL = _cfg.backtest.max_portfolio_vol  # STATUS: ACTIVE — backtest/engine.py, risk/portfolio_optimizer.py; max annualized vol
REGIME_RISK_MULTIPLIER = dict(_cfg.regime.risk_multiplier)  # STATUS: ACTIVE — backtest/engine.py; regime-conditional position sizing multipliers
REGIME_STOP_MULTIPLIER = dict(_cfg.regime.stop_multiplier)  # STATUS: ACTIVE — backtest/engine.py; regime-conditional stop loss multipliers
MAX_ANNUALIZED_TURNOVER = _cfg.backtest.max_annualized_turnover  # STATUS: ACTIVE — backtest/engine.py; 500% annualized turnover warning threshold

# ── Portfolio Turnover Penalty (SPEC-P04) ────────────────────────────
# Penalty per unit turnover in the mean-variance optimizer (decimal).
# Higher values discourage trading; lower values allow more rebalancing.
# When PORTFOLIO_TURNOVER_DYNAMIC is True, the optimizer uses
# max(PORTFOLIO_TURNOVER_PENALTY, 2× estimated execution cost) so the
# penalty always exceeds real trading friction.
PORTFOLIO_TURNOVER_PENALTY = 0.001                # STATUS: ACTIVE — autopilot/engine.py, risk/portfolio_optimizer.py; base turnover penalty
PORTFOLIO_TURNOVER_DYNAMIC = True                 # STATUS: ACTIVE — autopilot/engine.py; enable cost-aware dynamic penalty
PORTFOLIO_TURNOVER_COST_MULTIPLIER = 2.0          # STATUS: ACTIVE — autopilot/engine.py; multiplier on estimated cost for dynamic penalty floor

# Blend weight between Kelly sizing (0.0) and mean-variance optimizer (1.0).
# When optimizer weights are available, final size = (1-blend)*kelly + blend*optimizer.
OPTIMIZER_BLEND_WEIGHT = 0.4                       # STATUS: ACTIVE — autopilot/paper_trader.py; blends Kelly and optimizer sizing

# Maximum net weight in any single GICS sector (+/-10%).
# Enforced by risk/portfolio_optimizer.py ONLY when GICS_SECTORS is populated.
MAX_SECTOR_EXPOSURE = _cfg.backtest.max_sector_exposure  # STATUS: ACTIVE — risk/portfolio_optimizer.py; but INACTIVE when GICS_SECTORS is empty

# Maps ticker -> GICS sector name for sector-neutrality constraints.
# Loaded automatically from config_data/universe.yaml below.
# Can be refreshed from WRDS Compustat via: run_wrds_daily_refresh.py --gics
GICS_SECTORS: Dict[str, str] = {}                 # STATUS: ACTIVE — populated from config_data/universe.yaml on import

# Attempt to load GICS_SECTORS from universe.yaml at import time.
# This ensures sector-neutrality constraints in portfolio_optimizer.py
# are enforced without requiring a manual WRDS refresh.
import yaml as _yaml

_UNIVERSE_YAML = Path(__file__).parent / "config_data" / "universe.yaml"
if _UNIVERSE_YAML.exists() and not GICS_SECTORS:
    try:
        with open(_UNIVERSE_YAML) as _f:
            _universe = _yaml.safe_load(_f)
        if isinstance(_universe, dict) and "sectors" in _universe:
            for _sector_name, _tickers in _universe["sectors"].items():
                if isinstance(_tickers, list):
                    for _ticker in _tickers:
                        GICS_SECTORS[str(_ticker).upper()] = str(_sector_name)
    except Exception:
        pass  # Will be caught by validate_config()

del _yaml, _UNIVERSE_YAML  # Clean up module namespace

# ── Almgren-Chriss Optimal Execution ─────────────────────────────────
# PLACEHOLDER — Almgren-Chriss optimal execution model.
# The AC model exists in backtest/engine.py and is gated on this flag,
# but it is not wired into the trade simulation loop end-to-end.
# Set to True once backtest/execution.py integrates AC into the fill model.
ALMGREN_CHRISS_ENABLED = _cfg.execution.almgren_chriss_enabled  # STATUS: PLACEHOLDER — backtest/engine.py gates on this but not fully integrated
ALMGREN_CHRISS_ADV_THRESHOLD = _cfg.execution.almgren_chriss_adv_threshold  # STATUS: ACTIVE — backtest/engine.py; positions > 5% of ADV use AC cost model

# ── Validation ─────────────────────────────────────────────────────────
CPCV_PARTITIONS = _cfg.validation.cpcv_partitions  # STATUS: ACTIVE — backtest/validation.py; combinatorial purged CV partitions
CPCV_TEST_PARTITIONS = _cfg.validation.cpcv_test_partitions  # STATUS: ACTIVE — backtest/validation.py; CPCV test partitions
SPA_BOOTSTRAPS = _cfg.validation.spa_bootstraps    # STATUS: ACTIVE — backtest/validation.py; SPA bootstrap trials

# ── Data Quality ───────────────────────────────────────────────────────
DATA_QUALITY_ENABLED = True                       # STATUS: ACTIVE — data/loader.py; OHLCV quality checks
MAX_MISSING_BAR_FRACTION = _cfg.data.max_missing_bar_fraction  # STATUS: ACTIVE — data/quality.py; max fraction of missing bars
MAX_ZERO_VOLUME_FRACTION = _cfg.data.max_zero_volume_fraction  # STATUS: ACTIVE — data/quality.py; max fraction of zero-volume bars
MAX_ABS_DAILY_RETURN = _cfg.data.max_abs_daily_return  # STATUS: ACTIVE — data/quality.py; max absolute single-day return

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

PROMOTION_MIN_TRADES = _cfg.promotion.min_trades   # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WIN_RATE = _cfg.promotion.min_win_rate  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_SHARPE = _cfg.promotion.min_sharpe   # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_PROFIT_FACTOR = _cfg.promotion.min_profit_factor  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_DRAWDOWN = _cfg.promotion.max_drawdown  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_ANNUAL_RETURN = _cfg.promotion.min_annual_return  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_ACTIVE_STRATEGIES = _cfg.promotion.max_active_strategies  # STATUS: ACTIVE — autopilot/registry.py
PROMOTION_GRACE_CYCLES = 3  # STATUS: ACTIVE — autopilot/registry.py; cycles before removing a non-evaluated incumbent
PROMOTION_REQUIRE_ADVANCED_CONTRACT = _cfg.promotion.require_advanced_contract  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_DSR_PVALUE = _cfg.promotion.max_dsr_pvalue  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_PBO = _cfg.promotion.max_pbo         # STATUS: ACTIVE — autopilot/promotion_gate.py; tightened from 0.50 (Bailey et al. 2017)
PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED = True    # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_CAPACITY_UTILIZATION = 1.0           # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WF_OOS_CORR = _cfg.promotion.min_wf_oos_corr  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION = _cfg.promotion.min_wf_positive_fold_fraction  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MAX_WF_IS_OOS_GAP = _cfg.promotion.max_wf_is_oos_gap  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_MIN_REGIME_POSITIVE_FRACTION = _cfg.promotion.min_regime_positive_fraction  # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MAX_WORST_EVENT_LOSS = -0.08      # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE = 0.50      # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_EVENT_MIN_REGIME_STABILITY = 0.40       # STATUS: ACTIVE — autopilot/promotion_gate.py
PROMOTION_REQUIRE_STATISTICAL_TESTS = _cfg.promotion.require_statistical_tests  # STATUS: ACTIVE — autopilot/promotion_gate.py; require IC/FDR tests to pass
PROMOTION_REQUIRE_CPCV = _cfg.promotion.require_cpcv  # STATUS: ACTIVE — autopilot/promotion_gate.py; require CPCV to pass
PROMOTION_REQUIRE_SPA = _cfg.promotion.require_spa  # STATUS: ACTIVE — autopilot/promotion_gate.py; SPA is informational by default

# ── Stress-Regime Promotion Gates (SPEC-V02) ─────────────────────
PROMOTION_MAX_STRESS_DRAWDOWN = _cfg.promotion.max_stress_drawdown  # STATUS: ACTIVE — autopilot/promotion_gate.py; max drawdown in stress regimes (2, 3)
PROMOTION_MIN_STRESS_SHARPE = _cfg.promotion.min_stress_sharpe  # STATUS: ACTIVE — autopilot/promotion_gate.py; min Sharpe in stress regimes (not deeply negative)
PROMOTION_MAX_TRANSITION_DRAWDOWN = _cfg.promotion.max_transition_drawdown  # STATUS: ACTIVE — autopilot/promotion_gate.py; max drawdown near regime transitions
PROMOTION_STRESS_REGIMES = list(_cfg.promotion.stress_regimes)  # STATUS: ACTIVE — autopilot/promotion_gate.py; regime codes considered stress buckets

# ── Stress-Regime Capacity Gates (SPEC-V03) ──────────────────────
PROMOTION_MIN_STRESS_CAPACITY_USD = 500_000       # STATUS: ACTIVE — autopilot/promotion_gate.py; min capacity during stress regimes
PROMOTION_MIN_STRESS_CAPACITY_RATIO = 0.20        # STATUS: ACTIVE — autopilot/promotion_gate.py; min stress/overall capacity ratio

# ── Signal Selection — Spec 04 ─────────────────────────────────────
SIGNAL_TOPK_QUANTILE = 0.70                      # STATUS: ACTIVE — autopilot/engine.py; select top 70% by cross-sectional z-score
SIGNAL_Z_THRESHOLD = 1.5                         # STATUS: DEPRECATED — superseded by SIGNAL_TOPK_QUANTILE; kept for backward compat

# ── Meta-Labeling — Spec 04 ───────────────────────────────────────
META_LABELING_ENABLED = True                     # STATUS: ACTIVE — autopilot/engine.py, autopilot/meta_labeler.py
META_LABELING_RETRAIN_FREQ_DAYS = 7              # STATUS: ACTIVE — autopilot/engine.py; weekly retraining schedule
META_LABELING_FOLD_COUNT = 5                     # STATUS: ACTIVE — autopilot/meta_labeler.py; CV folds for training
META_LABELING_MIN_SAMPLES = 500                  # STATUS: ACTIVE — autopilot/meta_labeler.py; minimum samples to train
META_LABELING_CONFIDENCE_THRESHOLD = 0.55        # STATUS: ACTIVE — autopilot/engine.py; min confidence to pass filter
META_LABELING_XGB_PARAMS = {                     # STATUS: ACTIVE — autopilot/meta_labeler.py; XGBoost hyperparameters
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# ── Fold-Level Metrics — Spec 04 ──────────────────────────────────
FOLD_CONSISTENCY_PENALTY_WEIGHT = 0.15           # STATUS: ACTIVE — autopilot/promotion_gate.py; weight in composite score

# ── Kelly Sizing ──────────────────────────────────────────────────────
KELLY_FRACTION = _cfg.kelly.fraction               # STATUS: ACTIVE — risk/position_sizer.py; half-Kelly (conservative default)
MAX_PORTFOLIO_DD = _cfg.kelly.max_portfolio_dd     # STATUS: ACTIVE — risk/position_sizer.py; max portfolio drawdown for governor
KELLY_PORTFOLIO_BLEND = _cfg.kelly.portfolio_blend  # STATUS: PLACEHOLDER — defined but never imported; Kelly weight in composite blend
KELLY_BAYESIAN_ALPHA = _cfg.kelly.bayesian_alpha   # STATUS: ACTIVE — risk/position_sizer.py; Beta prior alpha for win rate
KELLY_BAYESIAN_BETA = _cfg.kelly.bayesian_beta     # STATUS: ACTIVE — risk/position_sizer.py; Beta prior beta for win rate
KELLY_REGIME_CONDITIONAL = _cfg.kelly.regime_conditional  # STATUS: PLACEHOLDER — defined but never imported; use regime-specific parameters
KELLY_MIN_SAMPLES_FOR_UPDATE = 10                 # STATUS: ACTIVE — risk/position_sizer.py; min trades before Bayesian posterior overrides prior

# ── Regime Trade Statistics (SPEC-P01) ─────────────────────────────
MIN_REGIME_TRADES_FOR_STATS = 30                  # STATUS: ACTIVE — risk/position_sizer.py; min trades per regime before learned stats override Bayesian prior
REGIME_STATS_PERSIST_PATH = MODEL_DIR / "regime_trade_stats.json"  # STATUS: ACTIVE — risk/position_sizer.py; persisted learned regime statistics

# ── Correlation Stress Tightening (SPEC-P03) ─────────────────────────
# When average pairwise correlation exceeds a threshold, constraint limits
# (sector cap, single-name cap, gross exposure) are scaled by the
# corresponding multiplier.  Applied ON TOP of regime multipliers so that
# correlation spikes are caught even before the regime detector reacts.
CORRELATION_STRESS_THRESHOLDS: Dict[float, float] = {  # STATUS: ACTIVE — risk/portfolio_risk.py; avg_corr → constraint multiplier
    0.6: 0.85,  # 15% tighter when avg pairwise |corr| > 0.6
    0.7: 0.70,  # 30% tighter when avg pairwise |corr| > 0.7
    0.8: 0.50,  # 50% tighter when avg pairwise |corr| > 0.8
}

# ── Risk-Free Rate ──────────────────────────────────────────────────
RISK_FREE_RATE = 0.04  # STATUS: ACTIVE — backtest/sharpe_utils.py; annualized risk-free rate for Sharpe/Sortino

# ── Risk Governor — Spec 05 ─────────────────────────────────────────
# Shock budget: reserve fraction of capital for tail events
SHOCK_BUDGET_PCT = 0.05                           # STATUS: ACTIVE — risk/position_sizer.py; reserve 5% of capital (positions capped at 95%)

# Concentration limit: max single-position notional as % of portfolio
CONCENTRATION_LIMIT_PCT = 0.20                    # STATUS: ACTIVE — risk/position_sizer.py; max 20% in any single position

# Turnover budget enforcement
TURNOVER_BUDGET_ENFORCEMENT = True                # STATUS: ACTIVE — risk/position_sizer.py; enable turnover budget constraint
TURNOVER_BUDGET_LOOKBACK_DAYS = 252               # STATUS: ACTIVE — risk/position_sizer.py; annualized turnover lookback

# Blend weights — static (regime-insensitive fallback)
BLEND_WEIGHTS_STATIC = {                          # STATUS: ACTIVE — risk/position_sizer.py; default blend weights
    'kelly': 0.30,
    'vol_scaled': 0.40,
    'atr_based': 0.30,
}

# Blend weights — regime-conditional (takes precedence over static)
BLEND_WEIGHTS_BY_REGIME = {                       # STATUS: ACTIVE — risk/position_sizer.py; regime-conditional blend weights
    'NORMAL': {'kelly': 0.35, 'vol_scaled': 0.35, 'atr_based': 0.30},
    'WARNING': {'kelly': 0.25, 'vol_scaled': 0.45, 'atr_based': 0.30},
    'CAUTION': {'kelly': 0.15, 'vol_scaled': 0.50, 'atr_based': 0.35},
    'CRITICAL': {'kelly': 0.05, 'vol_scaled': 0.50, 'atr_based': 0.45},
    'RECOVERY': {'kelly': 0.20, 'vol_scaled': 0.40, 'atr_based': 0.40},
}

# Uncertainty-aware sizing
UNCERTAINTY_SCALING_ENABLED = True                # STATUS: ACTIVE — risk/position_sizer.py; enable uncertainty-based size reduction
UNCERTAINTY_SIGNAL_WEIGHT = 0.40                  # STATUS: ACTIVE — risk/position_sizer.py; weight of signal_uncertainty in composite
UNCERTAINTY_REGIME_WEIGHT = 0.30                  # STATUS: ACTIVE — risk/position_sizer.py; weight of regime_entropy
UNCERTAINTY_DRIFT_WEIGHT = 0.30                   # STATUS: ACTIVE — risk/position_sizer.py; weight of drift_score (inverted)
UNCERTAINTY_REDUCTION_FACTOR = 0.30               # STATUS: ACTIVE — risk/position_sizer.py; max reduction from base size (30%)

PAPER_INITIAL_CAPITAL = _cfg.paper_trading.initial_capital  # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_MAX_TOTAL_POSITIONS = _cfg.paper_trading.max_total_positions  # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_USE_KELLY_SIZING = _cfg.paper_trading.use_kelly_sizing  # STATUS: ACTIVE — autopilot/paper_trader.py, api/services/backtest_service.py
PAPER_KELLY_FRACTION = _cfg.paper_trading.kelly_fraction  # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_LOOKBACK_TRADES = _cfg.paper_trading.kelly_lookback_trades  # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_MIN_SIZE_MULTIPLIER = _cfg.paper_trading.kelly_min_size_multiplier  # STATUS: ACTIVE — autopilot/paper_trader.py
PAPER_KELLY_MAX_SIZE_MULTIPLIER = _cfg.paper_trading.kelly_max_size_multiplier  # STATUS: ACTIVE — autopilot/paper_trader.py

# ── Feature Profiles ────────────────────────────────────────────────────
FEATURE_MODE_DEFAULT = _cfg.model.feature_mode     # STATUS: ACTIVE — run_backtest.py, run_train.py, run_predict.py; "full" or "core"

# ── Regime Trade Gating (SPEC-E02) ───────────────────────────────────
# Per-regime trade policy: each regime ID maps to an enabled flag and a
# minimum confidence threshold.  When ``enabled`` is False, signals in
# that regime are suppressed *unless* confidence >= ``min_confidence``.
REGIME_TRADE_POLICY: Dict[int, Dict] = {         # STATUS: ACTIVE — backtest/engine.py, api/routers/signals.py
    k: dict(v) for k, v in _cfg.regime.trade_policy.items()
}

# Deprecated aliases — kept for backward-compatibility with API layer
# and any external scripts that reference the old names.  These are
# derived from REGIME_TRADE_POLICY[2] and stay in sync.
REGIME_2_TRADE_ENABLED = REGIME_TRADE_POLICY[2]["enabled"]          # STATUS: DEPRECATED — use REGIME_TRADE_POLICY
REGIME_2_SUPPRESSION_MIN_CONFIDENCE = REGIME_TRADE_POLICY[2]["min_confidence"]  # STATUS: DEPRECATED — use REGIME_TRADE_POLICY

# ── Edge-After-Costs Trade Gating (SPEC-E01) ────────────────────────
EDGE_COST_GATE_ENABLED = True                    # STATUS: ACTIVE — backtest/engine.py; skip trades where predicted edge <= expected cost + buffer
EDGE_COST_BUFFER_BASE_BPS = 5.0                  # STATUS: ACTIVE — backtest/engine.py; additional buffer beyond expected cost (scales with uncertainty)

# ── Shock-Mode Execution Policy (SPEC-E03) ──────────────────────────
SHOCK_MODE_ENABLED = True                        # STATUS: ACTIVE — backtest/engine.py; enable unified shock-mode execution policy
SHOCK_MODE_SHOCK_MAX_PARTICIPATION = 0.005       # STATUS: ACTIVE — backtest/execution.py; max participation during full shock (0.5% of ADV)
SHOCK_MODE_SHOCK_SPREAD_MULT = 2.0               # STATUS: ACTIVE — backtest/execution.py; spread multiplier during full shock events
SHOCK_MODE_SHOCK_MIN_CONFIDENCE = 0.80           # STATUS: ACTIVE — backtest/engine.py; min confidence to enter during full shock events
SHOCK_MODE_ELEVATED_MAX_PARTICIPATION = 0.01     # STATUS: ACTIVE — backtest/execution.py; max participation during elevated uncertainty (1%)
SHOCK_MODE_ELEVATED_SPREAD_MULT = 1.5            # STATUS: ACTIVE — backtest/execution.py; spread multiplier during elevated uncertainty
SHOCK_MODE_ELEVATED_MIN_CONFIDENCE = 0.65        # STATUS: ACTIVE — backtest/engine.py; min confidence during elevated uncertainty
SHOCK_MODE_UNCERTAINTY_THRESHOLD = 0.7           # STATUS: ACTIVE — backtest/execution.py; HMM uncertainty above which elevated mode activates

# ── Regime Strategy Allocation ──────────────────────────────────────
REGIME_STRATEGY_ALLOCATION_ENABLED = True            # STATUS: ACTIVE — autopilot/strategy_allocator.py; adapt parameters by regime

# ── Drawdown Tiers ──────────────────────────────────────────────────
DRAWDOWN_WARNING_THRESHOLD = _cfg.drawdown.warning_threshold  # STATUS: ACTIVE — risk/stop_loss.py; -5% drawdown: reduce sizing 50%
DRAWDOWN_CAUTION_THRESHOLD = _cfg.drawdown.caution_threshold  # STATUS: ACTIVE — risk/stop_loss.py; -10% drawdown: no new entries, 25% sizing
DRAWDOWN_CRITICAL_THRESHOLD = _cfg.drawdown.critical_threshold  # STATUS: ACTIVE — risk/stop_loss.py; -15% drawdown: force liquidate
DRAWDOWN_DAILY_LOSS_LIMIT = _cfg.drawdown.daily_loss_limit  # STATUS: ACTIVE — risk/stop_loss.py; -3% daily loss limit
DRAWDOWN_WEEKLY_LOSS_LIMIT = _cfg.drawdown.weekly_loss_limit  # STATUS: ACTIVE — risk/stop_loss.py; -5% weekly loss limit
DRAWDOWN_RECOVERY_DAYS = _cfg.drawdown.recovery_days  # STATUS: ACTIVE — risk/stop_loss.py; days to return to full sizing after recovery
DRAWDOWN_SIZE_MULT_WARNING = _cfg.drawdown.size_mult_warning  # STATUS: ACTIVE — risk/stop_loss.py; size multiplier during WARNING tier
DRAWDOWN_SIZE_MULT_CAUTION = _cfg.drawdown.size_mult_caution  # STATUS: ACTIVE — risk/stop_loss.py; size multiplier during CAUTION tier

# ── Covariance ────────────────────────────────────────────────────
COVARIANCE_HALF_LIFE = 60                         # STATUS: ACTIVE — risk/covariance.py; EWMA half-life in trading days (60 ≈ 3 months)

# ── Stop Loss ───────────────────────────────────────────────────────
STOP_LOSS_SPREAD_BUFFER_BPS = 3.0                 # STATUS: ACTIVE — risk/stop_loss.py; bid-ask spread buffer for stop prices (bps)
HARD_STOP_PCT = _cfg.stop_loss.hard_stop_pct       # STATUS: ACTIVE — risk/stop_loss.py; -8% hard stop loss
ATR_STOP_MULTIPLIER = _cfg.stop_loss.atr_stop_multiplier  # STATUS: ACTIVE — risk/stop_loss.py; initial stop at 2x ATR
TRAILING_ATR_MULTIPLIER = _cfg.stop_loss.trailing_atr_multiplier  # STATUS: ACTIVE — risk/stop_loss.py; trailing stop at 1.5x ATR
TRAILING_ACTIVATION_PCT = _cfg.stop_loss.trailing_activation_pct  # STATUS: ACTIVE — risk/stop_loss.py; activate trailing stop after +2% gain
MAX_HOLDING_DAYS = _cfg.stop_loss.max_holding_days  # STATUS: ACTIVE — risk/stop_loss.py, backtest/engine.py; time-based stop at 30 days

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
IC_ROLLING_WINDOW = _cfg.validation.ic_rolling_window  # STATUS: ACTIVE — backtest/validation.py; rolling window for Information Coefficient

# ── Evaluation Layer (Spec 08) ──────────────────────────────────────
# Walk-forward with embargo
EVAL_WF_TRAIN_WINDOW = 250                        # STATUS: ACTIVE — evaluation/engine.py; training window in trading days
EVAL_WF_EMBARGO_DAYS = 5                          # STATUS: ACTIVE — evaluation/engine.py; embargo gap to prevent data leakage
EVAL_WF_TEST_WINDOW = 60                          # STATUS: ACTIVE — evaluation/engine.py; test window in trading days
EVAL_WF_SLIDE_FREQ = "weekly"                     # STATUS: ACTIVE — evaluation/engine.py; "weekly" or "daily"

# Rolling IC and decay detection
EVAL_IC_ROLLING_WINDOW = 60                       # STATUS: ACTIVE — evaluation/engine.py; rolling IC window
EVAL_IC_DECAY_THRESHOLD = 0.02                    # STATUS: ACTIVE — evaluation/engine.py; warn if IC falls below this
EVAL_IC_DECAY_LOOKBACK = 20                       # STATUS: ACTIVE — evaluation/engine.py; days to check for sustained low IC

# IC tracking for health system (SPEC-H01)
IC_TRACKING_LOOKBACK = 20                         # STATUS: ACTIVE — health_service.py; number of recent cycles to consider
IC_TRACKING_WARN_THRESHOLD = 0.01                 # STATUS: ACTIVE — health_service.py; WARNING if rolling IC mean < this
IC_TRACKING_CRITICAL_THRESHOLD = 0.0              # STATUS: ACTIVE — health_service.py; CRITICAL if rolling IC mean < this

# Execution quality monitoring (SPEC-H03)
EXEC_QUALITY_LOOKBACK = 50                        # STATUS: ACTIVE — health_service.py; number of recent fills to analyze
EXEC_QUALITY_WARN_SURPRISE_BPS = 2.0              # STATUS: ACTIVE — health_service.py; WARN if mean cost surprise > this (model underestimates)
EXEC_QUALITY_CRITICAL_SURPRISE_BPS = 5.0          # STATUS: ACTIVE — health_service.py; FAIL if mean cost surprise > this

# Ensemble disagreement tracking for health system (SPEC-H02)
ENSEMBLE_DISAGREEMENT_LOOKBACK = 20                # STATUS: ACTIVE — health_service.py; number of recent snapshots to consider
ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD = 0.015       # STATUS: ACTIVE — health_service.py; WARNING if mean disagreement > this
ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD = 0.03    # STATUS: ACTIVE — health_service.py; CRITICAL if mean disagreement > this

# Decile spread
EVAL_DECILE_SPREAD_MIN = 0.005                    # STATUS: ACTIVE — evaluation/metrics.py; minimum expected spread for a good predictor

# Calibration analysis
EVAL_CALIBRATION_BINS = 10                        # STATUS: ACTIVE — evaluation/calibration_analysis.py; number of bins for calibration curve
EVAL_OVERCONFIDENCE_THRESHOLD = 0.2               # STATUS: ACTIVE — evaluation/calibration_analysis.py; max gap before flagging

# Fragility metrics
EVAL_TOP_N_TRADES = [5, 10, 20]                   # STATUS: ACTIVE — evaluation/fragility.py; top N for PnL concentration
EVAL_RECOVERY_WINDOW = 60                         # STATUS: ACTIVE — evaluation/fragility.py; rolling window for recovery time
EVAL_CRITICAL_SLOWING_WINDOW = 60                 # STATUS: ACTIVE — evaluation/fragility.py; window for trend detection
EVAL_CRITICAL_SLOWING_SLOPE_THRESHOLD = 0.05      # STATUS: ACTIVE — evaluation/fragility.py; slope above this = danger

# ML diagnostics
EVAL_FEATURE_DRIFT_THRESHOLD = 0.7                # STATUS: ACTIVE — evaluation/ml_diagnostics.py; correlation below this = drift
EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD = 0.5        # STATUS: ACTIVE — evaluation/ml_diagnostics.py; correlation below this = disagreement

# Slice minimum sample size
EVAL_MIN_SLICE_SAMPLES = 20                       # STATUS: ACTIVE — evaluation/slicing.py; flag slices with fewer observations

# Red flag thresholds
EVAL_REGIME_SHARPE_DIVERGENCE = 0.5               # STATUS: ACTIVE — evaluation/engine.py; regime Sharpes differ > this = red flag
EVAL_OVERFIT_GAP_THRESHOLD = 0.10                 # STATUS: ACTIVE — evaluation/engine.py; IS-OOS gap > this = overfitting
EVAL_PNL_CONCENTRATION_THRESHOLD = 0.70           # STATUS: ACTIVE — evaluation/engine.py; top-20 PnL > this = fragile
EVAL_CALIBRATION_ERROR_THRESHOLD = 0.15           # STATUS: ACTIVE — evaluation/engine.py; calibration error > this = red flag

# ── Alert System ──────────────────────────────────────────────────────
ALERT_WEBHOOK_URL = ""                            # STATUS: ACTIVE — utils/logging.py; empty = disabled; set to Slack/Discord webhook URL
ALERT_HISTORY_FILE = RESULTS_DIR / "alerts_history.json"  # STATUS: ACTIVE — utils/logging.py; alert history persistence

# ── Log Configuration ──────────────────────────────────────────────
LOG_LEVEL = "INFO"                                # STATUS: ACTIVE — api/main.py; "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FORMAT = "structured"                         # STATUS: ACTIVE — api/main.py; "structured" or "json"

# ── Structural Features (Spec 02) ──────────────────────────────────
# Master toggle — set False to disable all structural features and revert
# to classical-only pipeline.  Individual families can also be disabled
# independently via the per-family flags below.
STRUCTURAL_FEATURES_ENABLED = True                # STATUS: ACTIVE — features/pipeline.py; master toggle for spectral/SSA/tail/eigen/OT

# Spectral Features (FFT-based frequency decomposition)
SPECTRAL_FFT_WINDOW = 252                         # STATUS: ACTIVE — indicators/spectral.py; rolling lookback for FFT (~1 year)
SPECTRAL_CUTOFF_PERIOD = 20                       # STATUS: ACTIVE — indicators/spectral.py; HF/LF boundary in trading days

# SSA Features (Singular Spectrum Analysis)
SSA_WINDOW = 60                                   # STATUS: ACTIVE — indicators/ssa.py; rolling SSA lookback
SSA_EMBED_DIM = 12                                # STATUS: ACTIVE — indicators/ssa.py; embedding dimension (< SSA_WINDOW)
SSA_N_SINGULAR = 5                                # STATUS: ACTIVE — indicators/ssa.py; signal components (rest = noise)

# Jump / Tail Risk Features
JUMP_INTENSITY_WINDOW = 20                        # STATUS: ACTIVE — indicators/tail_risk.py; lookback for jump detection
JUMP_INTENSITY_THRESHOLD = 2.5                    # STATUS: ACTIVE — indicators/tail_risk.py; sigma threshold for jumps

# Eigenvalue Spectrum Features (cross-asset, computed in compute_universe)
EIGEN_CONCENTRATION_WINDOW = 60                   # STATUS: ACTIVE — indicators/eigenvalue.py; rolling correlation window
EIGEN_MIN_ASSETS = 5                              # STATUS: ACTIVE — indicators/eigenvalue.py; min assets for eigenvalue features
EIGEN_REGULARIZATION = 0.01                       # STATUS: ACTIVE — indicators/eigenvalue.py; Tikhonov regularization

# Optimal Transport Features (distribution drift)
WASSERSTEIN_WINDOW = 30                           # STATUS: ACTIVE — indicators/ot_divergence.py; current distribution window
WASSERSTEIN_REF_WINDOW = 60                       # STATUS: ACTIVE — indicators/ot_divergence.py; reference distribution window
SINKHORN_EPSILON = 0.01                           # STATUS: ACTIVE — indicators/ot_divergence.py; entropic regularization
SINKHORN_MAX_ITER = 100                           # STATUS: ACTIVE — indicators/ot_divergence.py; max Sinkhorn iterations


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
                "Populate via run_wrds_daily_refresh.py --gics, or ensure "
                "config_data/universe.yaml has a 'sectors' key."
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
                    "The system will operate in degraded mode: "
                    "OHLCV data falls back to local cache (may have survivorship bias), "
                    "and alternative data (earnings, options, short interest, insider) will be unavailable. "
                    "Set via: export WRDS_USERNAME=<your_username>"
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

    # ── SPEC-A02: Config validation completeness ────────────────────

    # 7. Ensemble weights must sum to 1.0
    weights = REGIME_ENSEMBLE_DEFAULT_WEIGHTS
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) >= 1e-6:
        issues.append({
            "level": "ERROR",
            "message": (
                f"REGIME_ENSEMBLE_DEFAULT_WEIGHTS sum to {weight_sum:.6f}, "
                "expected 1.0. Adjust weights so they sum to exactly 1.0."
            ),
        })

    # 8. No negative cost multipliers
    if TRANSACTION_COST_BPS < 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"TRANSACTION_COST_BPS={TRANSACTION_COST_BPS} is negative. "
                "Transaction costs must be >= 0."
            ),
        })
    if EXEC_SPREAD_BPS < 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"EXEC_SPREAD_BPS={EXEC_SPREAD_BPS} is negative. "
                "Spread costs must be >= 0."
            ),
        })
    if EXEC_IMPACT_COEFF_BPS < 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"EXEC_IMPACT_COEFF_BPS={EXEC_IMPACT_COEFF_BPS} is negative. "
                "Market impact coefficient must be >= 0."
            ),
        })

    # 9. Regime uncertainty entropy threshold in valid range
    if not (0 < REGIME_UNCERTAINTY_ENTROPY_THRESHOLD < 2.0):
        issues.append({
            "level": "ERROR",
            "message": (
                f"REGIME_UNCERTAINTY_ENTROPY_THRESHOLD={REGIME_UNCERTAINTY_ENTROPY_THRESHOLD} "
                "is out of range. Must be in (0, 2.0) for normalized entropy."
            ),
        })

    # 10. Label horizon must be a positive integer
    if not isinstance(LABEL_H, int) or LABEL_H <= 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"LABEL_H={LABEL_H} is invalid. "
                "Label horizon must be a positive integer."
            ),
        })

    # 11. Forward horizons must all be positive
    if not FORWARD_HORIZONS or not all(
        isinstance(h, int) and h > 0 for h in FORWARD_HORIZONS
    ):
        issues.append({
            "level": "ERROR",
            "message": (
                f"FORWARD_HORIZONS={FORWARD_HORIZONS} is invalid. "
                "Must be a non-empty list of positive integers."
            ),
        })

    # 12. Blend weights (static) must sum to 1.0
    blend_sum = sum(BLEND_WEIGHTS_STATIC.values())
    if abs(blend_sum - 1.0) >= 1e-6:
        issues.append({
            "level": "ERROR",
            "message": (
                f"BLEND_WEIGHTS_STATIC values sum to {blend_sum:.6f}, "
                "expected 1.0."
            ),
        })

    # 13. Blend weights (per-regime) must each sum to 1.0
    for regime_label, regime_weights in BLEND_WEIGHTS_BY_REGIME.items():
        rw_sum = sum(regime_weights.values())
        if abs(rw_sum - 1.0) >= 1e-6:
            issues.append({
                "level": "ERROR",
                "message": (
                    f"BLEND_WEIGHTS_BY_REGIME['{regime_label}'] values sum to "
                    f"{rw_sum:.6f}, expected 1.0."
                ),
            })

    # 14. Kelly fraction must be in (0, 1]
    if not (0 < KELLY_FRACTION <= 1.0):
        issues.append({
            "level": "ERROR",
            "message": (
                f"KELLY_FRACTION={KELLY_FRACTION} is out of range. "
                "Must be in (0, 1.0]."
            ),
        })

    # 15. Drawdown thresholds must be negative and ordered correctly
    dd_thresholds = [
        DRAWDOWN_WARNING_THRESHOLD,
        DRAWDOWN_CAUTION_THRESHOLD,
        DRAWDOWN_CRITICAL_THRESHOLD,
    ]
    if not all(t < 0 for t in dd_thresholds):
        issues.append({
            "level": "ERROR",
            "message": (
                "Drawdown thresholds must all be negative. Got: "
                f"WARNING={DRAWDOWN_WARNING_THRESHOLD}, "
                f"CAUTION={DRAWDOWN_CAUTION_THRESHOLD}, "
                f"CRITICAL={DRAWDOWN_CRITICAL_THRESHOLD}."
            ),
        })
    elif not (
        DRAWDOWN_WARNING_THRESHOLD
        > DRAWDOWN_CAUTION_THRESHOLD
        > DRAWDOWN_CRITICAL_THRESHOLD
    ):
        issues.append({
            "level": "ERROR",
            "message": (
                "Drawdown thresholds must be ordered: WARNING > CAUTION > CRITICAL. Got: "
                f"WARNING={DRAWDOWN_WARNING_THRESHOLD}, "
                f"CAUTION={DRAWDOWN_CAUTION_THRESHOLD}, "
                f"CRITICAL={DRAWDOWN_CRITICAL_THRESHOLD}."
            ),
        })

    # 16. Position sizing percentage must be positive and <= 1.0
    if not (0 < POSITION_SIZE_PCT <= 1.0):
        issues.append({
            "level": "ERROR",
            "message": (
                f"POSITION_SIZE_PCT={POSITION_SIZE_PCT} is out of range. "
                "Must be in (0, 1.0]."
            ),
        })

    # 17. MAX_POSITIONS must be a positive integer
    if not isinstance(MAX_POSITIONS, int) or MAX_POSITIONS <= 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"MAX_POSITIONS={MAX_POSITIONS} is invalid. "
                "Must be a positive integer."
            ),
        })

    # 18. Confidence threshold must be in [0, 1]
    if not (0 <= CONFIDENCE_THRESHOLD <= 1.0):
        issues.append({
            "level": "ERROR",
            "message": (
                f"CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD} is out of range. "
                "Must be in [0, 1.0]."
            ),
        })

    # 19. MAX_PORTFOLIO_VOL must be positive
    if MAX_PORTFOLIO_VOL <= 0:
        issues.append({
            "level": "ERROR",
            "message": (
                f"MAX_PORTFOLIO_VOL={MAX_PORTFOLIO_VOL} is invalid. "
                "Must be positive."
            ),
        })

    # 20. Regime risk multipliers must all be positive
    for regime_id, mult in REGIME_RISK_MULTIPLIER.items():
        if mult < 0:
            issues.append({
                "level": "ERROR",
                "message": (
                    f"REGIME_RISK_MULTIPLIER[{regime_id}]={mult} is negative. "
                    "Risk multipliers must be >= 0."
                ),
            })

    return issues
