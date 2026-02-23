/** Matches actual /api/backtests/latest response.data */
export interface BacktestResult {
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
}

/** Matches actual /api/backtests/latest/trades response.data */
export interface TradesResponse {
  available: boolean
  trades: TradeRecord[]
  total: number
  offset: number
  limit: number
}

export interface TradeRecord {
  ticker: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  predicted_return: number
  actual_return: number
  net_return: number
  regime: string
  confidence: number
  holding_days: number
  position_size: number
  exit_reason: string
}

/** Matches actual /api/backtests/latest/equity-curve response.data */
export interface EquityCurveResponse {
  available: boolean
  points: EquityCurvePoint[]
}

export interface EquityCurvePoint {
  date: string
  value: number
}
