import { useLogs, type LogEntry } from '@/api/queries/useLogs'
import type { ResponseMeta } from '@/types/api'

export function useLogStream(lastN: number = 200): {
  logs: LogEntry[]
  isLoading: boolean
  meta?: ResponseMeta
} {
  const query = useLogs(lastN)
  return {
    logs: query.data?.data ?? [],
    isLoading: query.isLoading,
    meta: query.data?.meta,
  }
}
