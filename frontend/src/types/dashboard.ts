/** Matches actual /api/dashboard/summary response.data */
export interface DashboardSummary {
  horizon: number
  total_trades: number
  win_rate: number
  avg_return: number
  sharpe: number
  sortino: number
  max_drawdown: number
  profit_factor: number
  annualized_return: number
  trades_per_year: number
  regime_breakdown: Record<string, {
    n_trades: number
    avg_return: number
    win_rate: number
    sharpe: number
  }>
  available: boolean
  model_staleness_days: number | null
  retrain_overdue: boolean
  model_version: string | null
  sizing_method: string
  walk_forward_mode: string
}

/** Matches actual /api/dashboard/regime response.data */
export interface RegimeInfo {
  current_label: string
  as_of: string
  current_probs: Record<string, number>
  transition_matrix: number[][]
  prob_history: Array<{
    date: string
    regime_prob_0: number
    regime_prob_1: number
    regime_prob_2: number
    regime_prob_3: number
  }>
  regime_changes?: Array<{
    from_regime: string
    to_regime: string
    date: string
    duration_days: number
  }>
  current_regime_duration_days?: number
}

/** Matches /api/regime/metadata response.data */
export interface RegimeMetadata {
  regimes: Record<string, {
    name: string
    definition: string
    detection: string
    portfolio_impact: {
      position_size_multiplier: number
      stop_loss_multiplier: number
      description: string
    }
    color: string
  }>
  detection_method: string
  ensemble_enabled: boolean
  transition_matrix_explanation: string
}

export interface TimeSeriesPoint {
  date: string
  value: number
}

/** Matches /api/dashboard/returns-distribution response.data */
export interface ReturnsDistribution {
  bins: Array<{ x: number; count: number }>
  var95: number
  var99: number
  cvar95: number
  cvar99: number
  count: number
  mean: number
  std: number
  skew: number
  kurtosis: number
}

/** Matches /api/dashboard/rolling-risk response.data */
export interface RollingRisk {
  rolling_vol: TimeSeriesPoint[]
  rolling_sharpe: TimeSeriesPoint[]
  drawdown: TimeSeriesPoint[]
  points: number
  vol_window: number
  sharpe_window: number
}

/** Matches /api/dashboard/equity with benchmark overlay response.data */
export interface EquityWithBenchmark {
  strategy: TimeSeriesPoint[]
  benchmark: TimeSeriesPoint[]
  points: number
}

/** Matches /api/dashboard/attribution response.data */
export interface Attribution {
  factors: Array<{
    name: string
    coefficient: number
    annualized_contribution: number
  }>
  residual_alpha: number
  r_squared: number
  points: number
}
