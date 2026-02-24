import { useQuery } from '@tanstack/react-query'
import { get } from '@/api/client'
import {
  DASHBOARD_SUMMARY,
  DASHBOARD_REGIME,
  DASHBOARD_RETURNS_DISTRIBUTION,
  DASHBOARD_ROLLING_RISK,
  DASHBOARD_EQUITY,
  DASHBOARD_ATTRIBUTION,
  REGIME_METADATA,
} from '@/api/endpoints'
import type {
  DashboardSummary,
  RegimeInfo,
  RegimeMetadata,
  ReturnsDistribution,
  RollingRisk,
  EquityWithBenchmark,
  Attribution,
} from '@/types/dashboard'

export function useDashboardSummary() {
  return useQuery({
    queryKey: ['dashboard', 'summary'],
    queryFn: () => get<DashboardSummary>(DASHBOARD_SUMMARY),
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

export function useDashboardRegime() {
  return useQuery({
    queryKey: ['dashboard', 'regime'],
    queryFn: () => get<RegimeInfo>(DASHBOARD_REGIME),
    staleTime: 60_000,
  })
}

export function useReturnsDistribution() {
  return useQuery({
    queryKey: ['dashboard', 'returns-distribution'],
    queryFn: () => get<ReturnsDistribution>(DASHBOARD_RETURNS_DISTRIBUTION),
    staleTime: 60_000,
  })
}

export function useRollingRisk() {
  return useQuery({
    queryKey: ['dashboard', 'rolling-risk'],
    queryFn: () => get<RollingRisk>(DASHBOARD_ROLLING_RISK),
    staleTime: 60_000,
  })
}

export function useEquityWithBenchmark() {
  return useQuery({
    queryKey: ['dashboard', 'equity-benchmark'],
    queryFn: () => get<EquityWithBenchmark>(DASHBOARD_EQUITY),
    staleTime: 60_000,
  })
}

export function useAttribution() {
  return useQuery({
    queryKey: ['dashboard', 'attribution'],
    queryFn: () => get<Attribution>(DASHBOARD_ATTRIBUTION),
    staleTime: 60_000,
  })
}

export function useRegimeMetadata() {
  return useQuery({
    queryKey: ['regime', 'metadata'],
    queryFn: () => get<RegimeMetadata>(REGIME_METADATA),
    staleTime: 600_000, // Metadata changes rarely
  })
}
