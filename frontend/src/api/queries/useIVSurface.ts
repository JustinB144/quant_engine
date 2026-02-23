import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { IV_SURFACE_ARB_FREE } from '@/api/endpoints'

export interface ArbFreeSVIResult {
  moneyness: number[]
  expiries: number[]
  raw_iv: number[][]
  adj_iv: number[][]
  market_iv: number[][]
  objectives: (number | null)[]
  n_expiries: number
  n_strikes: number
  raw_calendar_violation: number
  adj_calendar_violation: number
}

export function useArbFreeSVI() {
  return useQuery({
    queryKey: ['iv-surface', 'arb-free-svi'],
    queryFn: () => get<ArbFreeSVIResult>(IV_SURFACE_ARB_FREE),
    staleTime: 300_000,
  })
}
