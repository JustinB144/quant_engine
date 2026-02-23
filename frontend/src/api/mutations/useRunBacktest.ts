import { useMutation, useQueryClient } from '@tanstack/react-query'
import { post } from '@/api/client'
import { BACKTESTS_RUN } from '@/api/endpoints'
import type { BacktestRequest } from '@/types/compute'
import type { JobRecord } from '@/types/jobs'

export function useRunBacktest() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: BacktestRequest) => post<JobRecord>(BACKTESTS_RUN, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
