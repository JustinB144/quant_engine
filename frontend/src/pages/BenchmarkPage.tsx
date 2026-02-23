import React from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import Spinner from '@/components/ui/Spinner'
import ErrorPanel from '@/components/ui/ErrorPanel'
import BenchmarkMetricCards from './BenchmarkPage/BenchmarkMetricCards'
import BenchmarkChartGrid from './BenchmarkPage/BenchmarkChartGrid'
import { useBenchmarkComparison, useBenchmarkEquityCurves, useBenchmarkRollingMetrics } from '@/api/queries/useBenchmark'

export default function BenchmarkPage() {
  const comparison = useBenchmarkComparison()
  const equityCurves = useBenchmarkEquityCurves()
  const rollingMetrics = useBenchmarkRollingMetrics()

  const benchData = comparison.data?.data
  const equityData = equityCurves.data?.data
  const rollingData = rollingMetrics.data?.data
  const meta = comparison.data?.meta

  const isLoading = comparison.isLoading
  const error = comparison.error

  return (
    <PageContainer>
      <PageHeader title="S&P Comparison" subtitle="Strategy vs benchmark performance analysis" />
      {isLoading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : error ? (
        <ErrorPanel message={error.message} onRetry={() => comparison.refetch()} />
      ) : benchData ? (
        <>
          <BenchmarkMetricCards data={benchData} />
          <BenchmarkChartGrid
            comparison={benchData}
            equityCurves={equityData}
            rollingMetrics={rollingData}
            meta={meta}
          />
        </>
      ) : null}
    </PageContainer>
  )
}
