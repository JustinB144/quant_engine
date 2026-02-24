import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { DATA_UNIVERSE, DATA_STATUS, DATA_TICKER, DATA_TICKER_BARS, DATA_TICKER_INDICATORS } from '@/api/endpoints'
import type { UniverseInfo, TickerDetail } from '@/types/data'

export interface TickerCacheEntry {
  ticker: string
  permno: string
  source: string
  last_bar_date: string
  start_date: string
  total_bars: number
  timeframes_available: string[]
  freshness: 'FRESH' | 'STALE' | 'VERY_STALE' | 'UNKNOWN'
  days_stale: number | null
}

export interface DataStatusSummary {
  total_cached: number
  fresh: number
  stale: number
  very_stale: number
  cache_dir: string
  cache_exists: boolean
}

export interface DataStatus {
  tickers: TickerCacheEntry[]
  summary: DataStatusSummary
}

export function useUniverse() {
  return useQuery({
    queryKey: ['data', 'universe'],
    queryFn: () => get<UniverseInfo>(DATA_UNIVERSE),
    staleTime: 300_000,
  })
}

export function useDataStatus() {
  return useQuery({
    queryKey: ['data', 'status'],
    queryFn: () => get<DataStatus>(DATA_STATUS),
    staleTime: 60_000,
  })
}

export function useTickerDetail(ticker: string, years: number = 2) {
  return useQuery({
    queryKey: ['data', 'ticker', ticker, years],
    queryFn: () => get<TickerDetail>(`${DATA_TICKER(ticker)}?years=${years}`),
    enabled: !!ticker,
  })
}

/** Bars with timeframe support */
export interface BarsResponse {
  ticker: string
  timeframe: string
  bars: Array<{
    time: string
    open: number
    high: number
    low: number
    close: number
    volume: number
  }>
  total_bars: number
  available_timeframes: string[]
}

export function useTickerBars(ticker: string, timeframe: string = '1d', bars: number = 500) {
  return useQuery({
    queryKey: ['data', 'bars', ticker, timeframe, bars],
    queryFn: () => get<BarsResponse>(`${DATA_TICKER_BARS(ticker)}?timeframe=${timeframe}&bars=${bars}`),
    enabled: !!ticker,
    staleTime: 60_000,
  })
}

/** Indicator computation results */
export interface IndicatorsResponse {
  ticker: string
  timeframe: string
  indicators: Record<string, Record<string, unknown>>
  available_indicators: string[]
}

export function useTickerIndicators(
  ticker: string,
  timeframe: string = '1d',
  indicators: string[] = [],
) {
  const indicatorStr = indicators.join(',')
  return useQuery({
    queryKey: ['data', 'indicators', ticker, timeframe, indicatorStr],
    queryFn: () =>
      get<IndicatorsResponse>(
        `${DATA_TICKER_INDICATORS(ticker)}?timeframe=${timeframe}&indicators=${indicatorStr}`,
      ),
    enabled: !!ticker && indicators.length > 0,
    staleTime: 300_000,
  })
}
