import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { get } from '@/api/client'
import { BACKTESTS_LATEST, BACKTESTS_TRADES, BACKTESTS_EQUITY_CURVE } from '@/api/endpoints'
import type { BacktestResult, TradesResponse, EquityCurveResponse } from '@/types/backtests'

export function useLatestBacktest(horizon: number = 10) {
  return useQuery({
    queryKey: ['backtests', 'latest', horizon],
    queryFn: () => get<BacktestResult>(`${BACKTESTS_LATEST}?horizon=${encodeURIComponent(String(horizon))}`),
    staleTime: 120_000,
  })
}

export function useTrades(horizon: number = 10, limit: number = 200, offset: number = 0) {
  return useQuery({
    queryKey: ['backtests', 'trades', horizon, limit, offset],
    queryFn: () =>
      get<TradesResponse>(`${BACKTESTS_TRADES}?horizon=${encodeURIComponent(String(horizon))}&limit=${encodeURIComponent(String(limit))}&offset=${encodeURIComponent(String(offset))}`),
    placeholderData: keepPreviousData,
  })
}

export function useEquityCurve(horizon: number = 10) {
  return useQuery({
    queryKey: ['backtests', 'equity-curve', horizon],
    queryFn: () => get<EquityCurveResponse>(`${BACKTESTS_EQUITY_CURVE}?horizon=${encodeURIComponent(String(horizon))}`),
    staleTime: 120_000,
  })
}
