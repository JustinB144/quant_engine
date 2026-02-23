import React from 'react'
import { createColumnHelper } from '@tanstack/react-table'
import DataTable from './DataTable'
import RegimeBadge from '@/components/ui/RegimeBadge'
import { formatSignedPercent, colorForReturn, formatPercent } from './tableUtils'
import { useCSVExport } from '@/hooks/useCSVExport'
import { Download } from 'lucide-react'
import type { TradeRecord } from '@/types/backtests'

const col = createColumnHelper<TradeRecord>()

const columns = [
  col.accessor('ticker', { header: 'Ticker', size: 80 }),
  col.accessor('entry_date', { header: 'Entry', size: 100 }),
  col.accessor('exit_date', { header: 'Exit', size: 100 }),
  col.accessor('predicted_return', {
    header: 'Pred',
    cell: (info) => (
      <span style={{ color: colorForReturn(info.getValue()) }}>
        {formatSignedPercent(info.getValue(), 4)}
      </span>
    ),
  }),
  col.accessor('actual_return', {
    header: 'Actual',
    cell: (info) => (
      <span style={{ color: colorForReturn(info.getValue()) }}>
        {formatSignedPercent(info.getValue(), 4)}
      </span>
    ),
  }),
  col.accessor('net_return', {
    header: 'Net',
    cell: (info) => (
      <span style={{ color: colorForReturn(info.getValue()), fontWeight: 600 }}>
        {formatSignedPercent(info.getValue(), 4)}
      </span>
    ),
  }),
  col.accessor('regime', {
    header: 'Regime',
    cell: (info) => <RegimeBadge regime={info.getValue()} />,
  }),
  col.accessor('confidence', {
    header: 'Confidence',
    cell: (info) => formatPercent(info.getValue(), 2),
  }),
  col.accessor('exit_reason', { header: 'Reason', size: 120 }),
]

interface TradeTableProps {
  data: TradeRecord[]
  pageSize?: number
}

export default function TradeTable({ data, pageSize = 25 }: TradeTableProps) {
  const exportCSV = useCSVExport()

  return (
    <div>
      <div className="flex justify-end mb-2">
        <button
          onClick={() => exportCSV(data as unknown as Record<string, unknown>[], 'trades.csv')}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md"
          style={{
            backgroundColor: 'var(--bg-tertiary)',
            border: '1px solid var(--border-light)',
            color: 'var(--text-secondary)',
            fontSize: 11,
            cursor: 'pointer',
          }}
        >
          <Download size={11} />
          Export CSV
        </button>
      </div>
      <DataTable data={data} columns={columns} pageSize={pageSize} />
    </div>
  )
}
