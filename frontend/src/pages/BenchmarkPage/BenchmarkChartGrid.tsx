import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import type { BenchmarkData } from '@/api/queries/useBenchmark'

export default function BenchmarkChartGrid({ data }: { data: BenchmarkData }) {
  const s = data.strategy
  const b = data.benchmark

  return (
    <div>
      <div className="card-panel mb-3">
        <div className="card-panel-header">Detailed Comparison</div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="font-mono mb-2" style={{ fontSize: 11, color: 'var(--accent-blue)', fontWeight: 600 }}>STRATEGY</div>
            <div className="grid grid-cols-3 gap-2">
              <MetricCard label="Annual Vol" value={`${(s.annual_vol * 100).toFixed(1)}%`} />
              <MetricCard label="Sortino" value={s.sortino.toFixed(2)} />
              <MetricCard label="CVaR 95" value={`${(s.cvar95 * 100).toFixed(2)}%`} color="var(--accent-amber)" />
            </div>
          </div>
          <div>
            <div className="font-mono mb-2" style={{ fontSize: 11, color: 'var(--text-tertiary)', fontWeight: 600 }}>BENCHMARK (SPY)</div>
            <div className="grid grid-cols-3 gap-2">
              <MetricCard label="Annual Vol" value={`${(b.annual_vol * 100).toFixed(1)}%`} />
              <MetricCard label="Sortino" value={b.sortino.toFixed(2)} />
              <MetricCard label="CVaR 95" value={`${(b.cvar95 * 100).toFixed(2)}%`} color="var(--accent-amber)" />
            </div>
          </div>
        </div>
      </div>

      <div className="card-panel">
        <div className="card-panel-header">Data Points</div>
        <pre className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', margin: 0 }}>
          Strategy: {data.strategy_points} observations | Benchmark: {data.benchmark_points} observations
        </pre>
      </div>
    </div>
  )
}
