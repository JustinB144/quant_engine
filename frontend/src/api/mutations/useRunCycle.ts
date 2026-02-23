import { useMutation, useQueryClient } from '@tanstack/react-query'
import { post } from '@/api/client'
import { AUTOPILOT_RUN_CYCLE } from '@/api/endpoints'
import type { AutopilotRequest } from '@/types/compute'
import type { JobRecord } from '@/types/jobs'

export function useRunCycle() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req?: AutopilotRequest) => post<JobRecord>(AUTOPILOT_RUN_CYCLE, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] })
      qc.invalidateQueries({ queryKey: ['autopilot'] })
    },
  })
}
