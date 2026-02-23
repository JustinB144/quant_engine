import React from 'react'
import MetricCard from '@/components/ui/MetricCard'
import type { BenchmarkData } from '@/api/queries/useBenchmark'

export default function BenchmarkMetricCards({ data }: { data: BenchmarkData }) {
  const s = data.strategy
  const b = data.benchmark

  return (
    <div className="grid grid-cols-4 gap-3 mb-4">
      <MetricCard
        label="Strategy Return"
        value={`${(s.annual_return * 100).toFixed(1)}%`}
        color={s.annual_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        subtitle={`Sharpe: ${s.sharpe.toFixed(2)}`}
      />
      <MetricCard
        label="Benchmark Return"
        value={`${(b.annual_return * 100).toFixed(1)}%`}
        color={b.annual_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        subtitle={`Sharpe: ${b.sharpe.toFixed(2)}`}
      />
      <MetricCard
        label="Strategy Max DD"
        value={`${(s.max_drawdown * 100).toFixed(1)}%`}
        color="var(--accent-red)"
        subtitle={`VaR95: ${(s.var95 * 100).toFixed(2)}%`}
      />
      <MetricCard
        label="Benchmark Max DD"
        value={`${(b.max_drawdown * 100).toFixed(1)}%`}
        color="var(--accent-amber)"
        subtitle={`VaR95: ${(b.var95 * 100).toFixed(2)}%`}
      />
    </div>
  )
}
