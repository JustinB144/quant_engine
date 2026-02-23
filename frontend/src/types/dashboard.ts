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
}
