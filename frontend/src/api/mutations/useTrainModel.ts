import { useMutation, useQueryClient } from '@tanstack/react-query'
import { post } from '@/api/client'
import { MODELS_TRAIN } from '@/api/endpoints'
import type { TrainRequest } from '@/types/compute'
import type { JobRecord } from '@/types/jobs'

export function useTrainModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: TrainRequest) => post<JobRecord>(MODELS_TRAIN, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
