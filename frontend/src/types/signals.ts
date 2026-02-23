/** Matches actual /api/signals/latest response.data */
export interface SignalsResponse {
  available: boolean
  horizon: number
  signals: SignalRow[]
  total: number
}

export interface SignalRow {
  ticker: string
  global_prediction: number
  regime_prediction: number
  blend_alpha: number
  regime: number
  predicted_return: number
  confidence: number
  date: string
  cs_zscore?: number
  regime_suppressed?: boolean
}
