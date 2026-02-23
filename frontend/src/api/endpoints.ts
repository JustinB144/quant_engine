/** All 27 API endpoint URL constants */

// Health
export const HEALTH_QUICK = '/health'
export const HEALTH_DETAILED = '/health/detailed'

// Dashboard
export const DASHBOARD_SUMMARY = '/dashboard/summary'
export const DASHBOARD_REGIME = '/dashboard/regime'

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

// Data
export const DATA_UNIVERSE = '/data/universe'
export const DATA_TICKER = (ticker: string) => `/data/ticker/${ticker}`

// Benchmark
export const BENCHMARK_COMPARISON = '/benchmark/comparison'

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
