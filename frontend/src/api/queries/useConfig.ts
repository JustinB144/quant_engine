import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { CONFIG, CONFIG_STATUS } from '@/api/endpoints'
import type { RuntimeConfigValues } from '@/types/config'

export function useConfig() {
  return useQuery({
    queryKey: ['config'],
    queryFn: () => get<RuntimeConfigValues>(CONFIG),
    staleTime: 300_000,
  })
}

export function useConfigStatus() {
  return useQuery({
    queryKey: ['config-status'],
    queryFn: () => get<Record<string, Record<string, { value: unknown; status: string }>>>(CONFIG_STATUS),
    staleTime: 300_000,
  })
}
