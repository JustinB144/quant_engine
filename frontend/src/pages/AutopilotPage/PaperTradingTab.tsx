import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import MetricCard from '@/components/ui/MetricCard'
import DataTable from '@/components/tables/DataTable'
import EmptyState from '@/components/ui/EmptyState'
import { usePaperState } from '@/api/queries/useAutopilot'
import { createColumnHelper } from '@tanstack/react-table'
import type { PaperPosition } from '@/types/autopilot'

const col = createColumnHelper<PaperPosition>()
const columns = [
  col.accessor('ticker', { header: 'Ticker' }),
  col.accessor('shares', { header: 'Shares' }),
  col.accessor('entry_price', { header: 'Entry', cell: (info) => `$${info.getValue()?.toFixed(2) ?? 'N/A'}` }),
  col.accessor('current_price', { header: 'Current', cell: (info) => `$${info.getValue()?.toFixed(2) ?? 'N/A'}` }),
  col.accessor('pnl', { header: 'P&L', cell: (info) => {
    const v = info.getValue()
    if (v == null) return 'N/A'
    return <span style={{ color: v >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>${v.toFixed(2)}</span>
  }}),
]

export default function PaperTradingTab() {
  const { data, isLoading } = usePaperState()
  const paper = data?.data

  if (!paper?.available) {
    return <EmptyState message="Paper trading data is not currently available" />
  }

  const positions = paper.positions ?? []

  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-4">
        <MetricCard label="Cash" value={`$${paper.cash.toLocaleString()}`} color="var(--accent-blue)" />
        <MetricCard label="Realized P&L" value={`$${paper.realized_pnl.toFixed(2)}`} color={paper.realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'} />
        <MetricCard label="Open Positions" value={String(positions.length)} />
        <MetricCard label="Last Update" value={new Date(paper.last_update).toLocaleDateString()} subtitle={new Date(paper.last_update).toLocaleTimeString()} />
      </div>

      <ChartContainer title={`Open Positions (${positions.length})`} meta={data?.meta} isLoading={isLoading} isEmpty={positions.length === 0} emptyMessage="No open positions">
        <DataTable data={positions} columns={columns} pageSize={20} />
      </ChartContainer>
    </div>
  )
}
