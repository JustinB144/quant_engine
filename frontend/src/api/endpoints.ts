/** All 27 API endpoint URL constants */

// Health
export const HEALTH_QUICK = '/health'
export const HEALTH_DETAILED = '/health/detailed'

// Dashboard
export const DASHBOARD_SUMMARY = '/dashboard/summary'
export const DASHBOARD_REGIME = '/dashboard/regime'
export const DASHBOARD_RETURNS_DISTRIBUTION = '/dashboard/returns-distribution'
export const DASHBOARD_ROLLING_RISK = '/dashboard/rolling-risk'
export const DASHBOARD_EQUITY = '/dashboard/equity'
export const DASHBOARD_ATTRIBUTION = '/dashboard/attribution'

// Signals
export const SIGNALS_LATEST = '/signals/latest'

// Backtests
export const BACKTESTS_LATEST = '/backtests/latest'
export const BACKTESTS_TRADES = '/backtests/latest/trades'
export const BACKTESTS_EQUITY_CURVE = '/backtests/latest/equity-curve'
export const BACKTESTS_RUN = '/backtests/run'

// Models
export const MODELS_VERSIONS = '/models/versions'
export const MODELS_HEALTH = '/models/health'
export const MODELS_FEATURES = '/models/features/importance'
export const MODELS_TRAIN = '/models/train'
export const MODELS_PREDICT = '/models/predict'
export const MODELS_FEATURE_CORRELATIONS = '/models/features/correlations'

// Data
export const DATA_UNIVERSE = '/data/universe'
export const DATA_STATUS = '/data/status'
export const DATA_TICKER = (ticker: string) => `/data/ticker/${ticker}`

// Benchmark
export const BENCHMARK_COMPARISON = '/benchmark/comparison'
export const BENCHMARK_EQUITY_CURVES = '/benchmark/equity-curves'
export const BENCHMARK_ROLLING_METRICS = '/benchmark/rolling-metrics'

// Logs
export const LOGS = '/logs'

// Autopilot
export const AUTOPILOT_LATEST_CYCLE = '/autopilot/latest-cycle'
export const AUTOPILOT_STRATEGIES = '/autopilot/strategies'
export const AUTOPILOT_PAPER_STATE = '/autopilot/paper-state'
export const AUTOPILOT_RUN_CYCLE = '/autopilot/run-cycle'

// Jobs
export const JOBS_LIST = '/jobs'
export const JOBS_GET = (id: string) => `/jobs/${id}`
export const JOBS_EVENTS = (id: string) => `/jobs/${id}/events`
export const JOBS_CANCEL = (id: string) => `/jobs/${id}/cancel`

// Config
export const CONFIG = '/config'
export const CONFIG_STATUS = '/config/status'

// Regime
export const REGIME_METADATA = '/regime/metadata'

// IV Surface
export const IV_SURFACE_ARB_FREE = '/iv-surface/arb-free-svi'

// Data: bars & indicators (timeframe-aware)
export const DATA_TICKER_BARS = (ticker: string) => `/data/ticker/${ticker}/bars`
export const DATA_TICKER_INDICATORS = (ticker: string) => `/data/ticker/${ticker}/indicators`
export const DATA_TICKER_INDICATORS_BATCH = (ticker: string) => `/data/ticker/${ticker}/indicators/batch`

// Health history
export const HEALTH_HISTORY = '/health/history'
