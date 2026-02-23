import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { LOGS } from '@/api/endpoints'

/** Matches actual /api/logs response.data entries */
export interface LogEntry {
  ts: number
  level: string
  logger: string
  message: string
}

export function useLogs(lastN: number = 200) {
  return useQuery({
    queryKey: ['logs', lastN],
    queryFn: () => get<LogEntry[]>(`${LOGS}?last_n=${lastN}`),
    refetchInterval: 3_000,
  })
}
