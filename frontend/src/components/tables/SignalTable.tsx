import React from 'react'
import { createColumnHelper } from '@tanstack/react-table'
import DataTable from './DataTable'
import RegimeBadge from '@/components/ui/RegimeBadge'
import { formatSignedPercent, formatPercent, colorForReturn } from './tableUtils'
import type { SignalRow } from '@/types/signals'

const col = createColumnHelper<SignalRow>()

const columns = [
  col.accessor('ticker', { header: 'Ticker', size: 80 }),
  col.accessor('predicted_return', {
    header: 'Predicted Return',
    cell: (info) => (
      <span style={{ color: colorForReturn(info.getValue()), fontWeight: 600 }}>
        {formatSignedPercent(info.getValue(), 4)}
      </span>
    ),
  }),
  col.accessor('confidence', {
    header: 'Confidence',
    cell: (info) => {
      const val = info.getValue()
      const pct = Math.min(100, Math.max(0, val * 100))
      return (
        <div className="flex items-center gap-2">
          <div
            className="rounded-full overflow-hidden"
            style={{ width: 60, height: 4, backgroundColor: 'var(--bg-primary)' }}
          >
            <div
              style={{
                width: `${pct}%`,
                height: '100%',
                backgroundColor: 'var(--accent-blue)',
              }}
            />
          </div>
          <span>{formatPercent(val, 1)}</span>
        </div>
      )
    },
  }),
  col.accessor('regime', {
    header: 'Regime',
    cell: (info) => <RegimeBadge regime={info.getValue()} />,
  }),
]

export default function SignalTable({ data }: { data: SignalRow[] }) {
  return <DataTable data={data} columns={columns} pageSize={50} />
}
