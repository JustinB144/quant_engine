import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { DASHBOARD_SUMMARY, DASHBOARD_REGIME } from '@/api/endpoints'
import type { DashboardSummary, RegimeInfo } from '@/types/dashboard'

export function useDashboardSummary() {
  return useQuery({
    queryKey: ['dashboard', 'summary'],
    queryFn: () => get<DashboardSummary>(DASHBOARD_SUMMARY),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

export function useDashboardRegime() {
  return useQuery({
    queryKey: ['dashboard', 'regime'],
    queryFn: () => get<RegimeInfo>(DASHBOARD_REGIME),
    staleTime: 60_000,
  })
}
