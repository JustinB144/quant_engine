import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import LineChart from '@/components/charts/LineChart'
import { usePaperState } from '@/api/queries/useAutopilot'

export default function PaperPnLTracker() {
  const { data, isLoading } = usePaperState()
  const paper = data?.data

  if (!paper?.available) return null

  const positions = paper.positions ?? []
  const totalUnrealized = positions.reduce((sum: number, p: any) => sum + (p.pnl ?? 0), 0)
  const totalValue = paper.cash + positions.reduce((sum: number, p: any) => sum + (p.shares ?? 0) * (p.current_price ?? 0), 0)
  const totalReturn = ((totalValue - (paper.initial_capital ?? 1_000_000)) / (paper.initial_capital ?? 1_000_000) * 100)

  // Build equity curve from trade history if available
  const tradeHistory = paper.trade_history ?? []
  const equityPoints = tradeHistory.map((t: any, i: number) => ({
    date: t.exit_date ?? t.date ?? `Trade ${i + 1}`,
    value: t.cumulative_pnl ?? paper.realized_pnl,
  }))

  return (
    <div>
      <div className="grid grid-cols-5 gap-3 mb-4">
        <MetricCard
          label="Total Value"
          value={`$${totalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          color="var(--accent-blue)"
        />
        <MetricCard
          label="Cash"
          value={`$${paper.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
        />
        <MetricCard
          label="Realized P&L"
          value={`$${paper.realized_pnl.toFixed(2)}`}
          color={paper.realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
        <MetricCard
          label="Unrealized P&L"
          value={`$${totalUnrealized.toFixed(2)}`}
          color={totalUnrealized >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
        <MetricCard
          label="Total Return"
          value={`${totalReturn.toFixed(2)}%`}
          color={totalReturn >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
      </div>

      {equityPoints.length > 2 && (
        <ChartContainer title="Cumulative P&L" isLoading={isLoading}>
          <LineChart
            categories={equityPoints.map((p: any) => p.date)}
            series={[
              {
                name: 'Cumulative P&L',
                data: equityPoints.map((p: any) => p.value),
                color: '#58a6ff',
              },
            ]}
            height={280}
            yAxisName="P&L ($)"
          />
        </ChartContainer>
      )}

      {/* Per-position P&L breakdown */}
      {positions.length > 0 && (
        <div className="mt-3">
          <div
            className="font-mono uppercase mb-2"
            style={{ fontSize: 11, color: 'var(--text-tertiary)', letterSpacing: '0.5px' }}
          >
            Position P&L Breakdown
          </div>
          <div className="grid gap-1">
            {positions
              .sort((a: any, b: any) => (b.pnl ?? 0) - (a.pnl ?? 0))
              .map((pos: any) => (
                <div
                  key={pos.ticker}
                  className="flex items-center justify-between px-3 py-2 rounded"
                  style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
                >
                  <span className="font-mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>
                    {pos.ticker}
                  </span>
                  <div className="flex items-center gap-4">
                    <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                      {pos.shares} shares
                    </span>
                    <span
                      className="font-mono"
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        color: (pos.pnl ?? 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                      }}
                    >
                      {(pos.pnl ?? 0) >= 0 ? '+' : ''}${(pos.pnl ?? 0).toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
