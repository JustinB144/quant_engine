import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { AUTOPILOT_LATEST_CYCLE, AUTOPILOT_STRATEGIES, AUTOPILOT_PAPER_STATE } from '@/api/endpoints'
import type { CycleReport, StrategiesResponse, PaperState } from '@/types/autopilot'

export function useLatestCycle() {
  return useQuery({
    queryKey: ['autopilot', 'latest-cycle'],
    queryFn: () => get<CycleReport>(AUTOPILOT_LATEST_CYCLE),
    staleTime: 60_000,
  })
}

export function useStrategies() {
  return useQuery({
    queryKey: ['autopilot', 'strategies'],
    queryFn: () => get<StrategiesResponse>(AUTOPILOT_STRATEGIES),
    staleTime: 60_000,
  })
}

export function usePaperState() {
  return useQuery({
    queryKey: ['autopilot', 'paper-state'],
    queryFn: () => get<PaperState>(AUTOPILOT_PAPER_STATE),
    staleTime: 60_000,
  })
}
