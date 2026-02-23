import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { SIGNALS_LATEST } from '@/api/endpoints'
import type { SignalsResponse } from '@/types/signals'

export function useLatestSignals(horizon: number = 10) {
  return useQuery({
    queryKey: ['signals', 'latest', horizon],
    queryFn: () => get<SignalsResponse>(`${SIGNALS_LATEST}?horizon=${horizon}`),
    staleTime: 120_000,
  })
}
