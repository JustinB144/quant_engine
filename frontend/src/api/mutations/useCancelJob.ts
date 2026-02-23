import { useMutation, useQueryClient } from '@tanstack/react-query'
import { post } from '@/api/client'
import { JOBS_CANCEL } from '@/api/endpoints'

export function useCancelJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => post<{ cancelled: boolean }>(JOBS_CANCEL(jobId)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
