import React from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import Spinner from '@/components/ui/Spinner'
import ErrorPanel from '@/components/ui/ErrorPanel'
import BenchmarkMetricCards from './BenchmarkPage/BenchmarkMetricCards'
import BenchmarkChartGrid from './BenchmarkPage/BenchmarkChartGrid'
import { useBenchmarkComparison } from '@/api/queries/useBenchmark'

export default function BenchmarkPage() {
  const { data, isLoading, error, refetch } = useBenchmarkComparison()
  const benchData = data?.data

  return (
    <PageContainer>
      <PageHeader title="S&P Comparison" subtitle="Strategy vs benchmark performance analysis" />

      {isLoading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : error ? (
        <ErrorPanel message={error.message} onRetry={refetch} />
      ) : benchData ? (
        <>
          <BenchmarkMetricCards data={benchData} />
          <BenchmarkChartGrid data={benchData} />
        </>
      ) : null}
    </PageContainer>
  )
}
