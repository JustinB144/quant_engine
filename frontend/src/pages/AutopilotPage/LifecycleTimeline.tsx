import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import { useLatestCycle } from '@/api/queries/useAutopilot'

export default function LifecycleTimeline() {
  const { data, isLoading } = useLatestCycle()
  const cycle = data?.data

  return (
    <ChartContainer
      title="Latest Cycle Summary"
      meta={data?.meta}
      isLoading={isLoading}
      isEmpty={!cycle}
      emptyMessage="No autopilot cycles have been run"
    >
      {cycle && (
        <div>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <MetricCard label="Horizon" value={String(cycle.horizon)} subtitle={`${cycle.years} years`} />
            <MetricCard label="Feature Mode" value={cycle.feature_mode} />
            <MetricCard label="Candidates" value={String(cycle.n_candidates)} subtitle={`${cycle.n_passed} passed`} />
            <MetricCard label="Promoted" value={String(cycle.n_promoted)} color={cycle.n_promoted > 0 ? 'var(--accent-green)' : 'var(--text-tertiary)'} subtitle={`${cycle.n_active} active`} />
          </div>

          {cycle.paper_report && (
            <div className="card-panel">
              <div className="card-panel-header">Paper Trading Report</div>
              <div className="grid grid-cols-4 gap-3">
                <MetricCard label="Cash" value={`$${cycle.paper_report.cash.toLocaleString()}`} />
                <MetricCard label="Equity" value={`$${cycle.paper_report.equity.toLocaleString()}`} />
                <MetricCard label="Realized P&L" value={`$${cycle.paper_report.realized_pnl.toFixed(2)}`} color={cycle.paper_report.realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} />
                <MetricCard label="Open Positions" value={String(cycle.paper_report.open_positions)} />
              </div>
            </div>
          )}

          {cycle.top_decisions.length > 0 && (
            <div className="card-panel mt-3">
              <div className="card-panel-header">Top Decisions ({cycle.top_decisions.length})</div>
              {cycle.top_decisions.slice(0, 5).map((d, i) => (
                <div key={i} className="flex items-center gap-3 mb-1.5">
                  <span
                    className="inline-block rounded-full"
                    style={{
                      width: 8,
                      height: 8,
                      backgroundColor: d.passed ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}
                  />
                  <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1 }}>
                    {d.candidate.strategy_id}
                  </span>
                  <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                    Score: {d.score.toFixed(1)}
                  </span>
                  <span className="font-mono" style={{ fontSize: 10, color: d.passed ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {d.passed ? 'PASSED' : 'FAILED'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </ChartContainer>
  )
}
