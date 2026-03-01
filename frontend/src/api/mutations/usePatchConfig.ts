import { useMutation, useQueryClient } from '@tanstack/react-query'
import { patch } from '@/api/client'
import { CONFIG } from '@/api/endpoints'
import type { RuntimeConfigValues } from '@/types/config'

export function usePatchConfig() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (values: Partial<RuntimeConfigValues>) => patch<RuntimeConfigValues>(CONFIG, values),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['config'] })
      qc.invalidateQueries({ queryKey: ['config-status'] })
    },
  })
}
