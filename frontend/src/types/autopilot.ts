/** Matches actual /api/autopilot/latest-cycle response.data */
export interface CycleReport {
  horizon: number
  years: number
  feature_mode: string
  strict_oos: boolean
  survivorship_mode: boolean
  n_candidates: number
  n_passed: number
  n_promoted: number
  n_active: number
  top_decisions: TopDecision[]
  paper_report: PaperReport
  available: boolean
}

export interface TopDecision {
  candidate: {
    strategy_id: string
    horizon: number
    entry_threshold: number
    confidence_threshold: number
    use_risk_management: boolean
    max_positions: number
    position_size_pct: number
  }
  passed: boolean
  score: number
  reasons: string[]
  metrics: Record<string, number | boolean | string>
}

export interface PaperReport {
  as_of: string
  entries: number
  exits: number
  cash: number
  equity: number
  realized_pnl: number
  open_positions: number
  active_strategies: number
}

/** Matches actual /api/autopilot/strategies response.data */
export interface StrategiesResponse {
  active: StrategyInfo[]
  history_count: number
}

export interface StrategyInfo {
  strategy_id: string
  horizon?: number
  entry_threshold?: number
  confidence_threshold?: number
  use_risk_management?: boolean
  max_positions?: number
  position_size_pct?: number
  sharpe?: number
  return_pct?: number
}

/** Matches actual /api/autopilot/paper-state response.data */
export interface PaperState {
  cash: number
  realized_pnl: number
  positions: PaperPosition[]
  trades: PaperTrade[]
  last_update: string
  available: boolean
  trade_history?: PaperTrade[]
  initial_capital?: number
}

export interface PaperPosition {
  ticker: string
  shares: number
  entry_price: number
  current_price: number
  pnl: number
}

export interface PaperTrade {
  ticker: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  pnl: number
  cumulative_pnl?: number
}
