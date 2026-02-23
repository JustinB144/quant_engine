import {
  useDashboardSummary,
  useDashboardRegime,
  useReturnsDistribution,
  useRollingRisk,
  useEquityWithBenchmark,
  useAttribution,
} from '@/api/queries/useDashboard'
import { useModelHealth, useFeatureImportance } from '@/api/queries/useModels'
import { useEquityCurve, useTrades } from '@/api/queries/useBacktests'
import type { ResponseMeta } from '@/types/api'

export function useDashboardData() {
  const summary = useDashboardSummary()
  const regime = useDashboardRegime()
  const modelHealth = useModelHealth()
  const featureImportance = useFeatureImportance()
  const equityCurve = useEquityCurve()
  const trades = useTrades(10, 200, 0)

  // New chart-parity hooks: fetched lazily by React Query,
  // do NOT block overall dashboard loading state.
  const returnsDistribution = useReturnsDistribution()
  const rollingRisk = useRollingRisk()
  const equityWithBenchmark = useEquityWithBenchmark()
  const attribution = useAttribution()

  const isLoading = summary.isLoading || regime.isLoading
  const error = summary.error?.message || regime.error?.message

  const meta: ResponseMeta | undefined = summary.data?.meta

  return {
    summary: summary.data?.data,
    regime: regime.data?.data,
    modelHealth: modelHealth.data?.data,
    featureImportance: featureImportance.data?.data,
    equityCurve: equityCurve.data?.data,
    trades: trades.data?.data,
    isLoading,
    error,
    meta,
    refetch: () => {
      summary.refetch()
      regime.refetch()
      modelHealth.refetch()
      featureImportance.refetch()
      equityCurve.refetch()
      trades.refetch()
      returnsDistribution.refetch()
      rollingRisk.refetch()
      equityWithBenchmark.refetch()
      attribution.refetch()
    },
  }
}
