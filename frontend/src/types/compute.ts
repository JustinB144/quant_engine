export interface TrainRequest {
  horizons?: number[]
  feature_mode?: string
  survivorship?: boolean
  full_universe?: boolean
}

export interface BacktestRequest {
  horizon?: number
  tickers?: string[]
  years?: number
  feature_mode?: string
  risk_management?: boolean
  version?: string
  full_universe?: boolean
  holding_period?: number
  max_positions?: number
  entry_threshold?: number
  position_size?: number
}

export interface PredictRequest {
  horizon?: number
  top_n?: number
}

export interface AutopilotRequest {
  dry_run?: boolean
}
