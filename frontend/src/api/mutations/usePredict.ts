import { useMutation, useQueryClient } from '@tanstack/react-query'
import { post } from '@/api/client'
import { MODELS_PREDICT } from '@/api/endpoints'
import type { PredictRequest } from '@/types/compute'
import type { JobRecord } from '@/types/jobs'

export function usePredict() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: PredictRequest) => post<JobRecord>(MODELS_PREDICT, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
      qc.invalidateQueries({ queryKey: ['signals'] })
    },
  })
}
