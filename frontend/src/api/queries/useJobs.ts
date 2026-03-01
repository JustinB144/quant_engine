import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { JOBS_LIST, JOBS_GET } from '@/api/endpoints'
import type { JobRecord } from '@/types/jobs'

export function useJobsList() {
  return useQuery({
    queryKey: ['jobs', 'list'],
    queryFn: () => get<JobRecord[]>(JOBS_LIST),
    refetchInterval: 5_000,
  })
}

export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ['jobs', jobId],
    queryFn: () => get<JobRecord>(JOBS_GET(jobId!)),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.data?.status
      return status === 'running' || status === 'queued' ? 2_000 : false
    },
  })
}
