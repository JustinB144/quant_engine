import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import { MODELS_VERSIONS, MODELS_HEALTH, MODELS_FEATURES } from '@/api/endpoints'
import type { ModelVersionInfo, ModelHealth, FeatureImportance } from '@/types/models'

export function useModelVersions() {
  return useQuery({
    queryKey: ['models', 'versions'],
    queryFn: () => get<ModelVersionInfo[]>(MODELS_VERSIONS),
    staleTime: 120_000,
  })
}

export function useModelHealth() {
  return useQuery({
    queryKey: ['models', 'health'],
    queryFn: () => get<ModelHealth>(MODELS_HEALTH),
    staleTime: 120_000,
  })
}

export function useFeatureImportance() {
  return useQuery({
    queryKey: ['models', 'features'],
    queryFn: () => get<FeatureImportance>(MODELS_FEATURES),
    staleTime: 120_000,
  })
}
