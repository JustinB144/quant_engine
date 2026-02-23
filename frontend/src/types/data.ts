/** Matches actual /api/data/universe response.data */
export interface UniverseInfo {
  full_size: number
  quick_size: number
  full_tickers: string[]
  quick_tickers: string[]
  benchmark: string
}

export interface OHLCVBar {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

/** Matches actual /api/data/ticker/{ticker} response.data */
export interface TickerDetail {
  ticker: string
  bars: OHLCVBar[]
  start_date: string
  end_date: string
  total_bars: number
  data_quality?: string
}
