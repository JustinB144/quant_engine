import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { CONFIG } from '@/api/endpoints'
import type { RuntimeConfigValues } from '@/types/config'

export function useConfig() {
  return useQuery({
    queryKey: ['config'],
    queryFn: () => get<RuntimeConfigValues>(CONFIG),
    staleTime: 300_000,
  })
}
