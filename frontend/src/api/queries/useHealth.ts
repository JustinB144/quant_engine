import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { HEALTH_QUICK, HEALTH_DETAILED } from '@/api/endpoints'
import type { QuickStatus, SystemHealthDetail } from '@/types/health'

export function useQuickHealth() {
  return useQuery({
    queryKey: ['health', 'quick'],
    queryFn: () => get<QuickStatus>(HEALTH_QUICK),
    staleTime: 30_000,
  })
}

export function useDetailedHealth() {
  return useQuery({
    queryKey: ['health', 'detailed'],
    queryFn: () => get<SystemHealthDetail>(HEALTH_DETAILED),
    staleTime: 60_000,
  })
}
