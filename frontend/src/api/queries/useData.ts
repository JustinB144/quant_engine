import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { DATA_UNIVERSE, DATA_TICKER } from '@/api/endpoints'
import type { UniverseInfo, TickerDetail } from '@/types/data'

export function useUniverse() {
  return useQuery({
    queryKey: ['data', 'universe'],
    queryFn: () => get<UniverseInfo>(DATA_UNIVERSE),
    staleTime: 300_000,
  })
}

export function useTickerDetail(ticker: string, years: number = 2) {
  return useQuery({
    queryKey: ['data', 'ticker', ticker, years],
    queryFn: () => get<TickerDetail>(`${DATA_TICKER(ticker)}?years=${years}`),
    enabled: !!ticker,
  })
}
