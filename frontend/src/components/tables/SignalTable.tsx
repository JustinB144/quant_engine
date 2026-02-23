import React from 'react'
import { createColumnHelper } from '@tanstack/react-table'
import DataTable from './DataTable'
import RegimeBadge from '@/components/ui/RegimeBadge'
import { formatSignedPercent, formatPercent, formatNumber, colorForReturn } from './tableUtils'
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
  col.accessor('cs_zscore', {
    header: 'CS Z-Score',
    size: 90,
    cell: (info) => {
      const val = info.getValue()
      if (val == null) return <span style={{ color: 'var(--text-tertiary)' }}>—</span>
      const absVal = Math.abs(val)
      const color = absVal >= 2.0
        ? 'var(--accent-green)'
        : absVal >= 1.0
          ? 'var(--accent-blue)'
          : 'var(--text-secondary)'
      return (
        <span className="font-mono" style={{ color, fontWeight: absVal >= 1.5 ? 600 : 400 }}>
          {val >= 0 ? '+' : ''}{formatNumber(val, 2)}
        </span>
      )
    },
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
  col.accessor('regime_suppressed', {
    header: 'Status',
    size: 90,
    cell: (info) => {
      const suppressed = info.getValue()
      if (suppressed) {
        return (
          <span
            className="font-mono uppercase"
            style={{ fontSize: 10, color: 'var(--accent-amber)', fontWeight: 600 }}
          >
            SUPPRESSED
          </span>
        )
      }
      return (
        <span
          className="font-mono uppercase"
          style={{ fontSize: 10, color: 'var(--accent-green)' }}
        >
          ACTIVE
        </span>
      )
    },
  }),
]

export default function SignalTable({ data }: { data: SignalRow[] }) {
  return <DataTable data={data} columns={columns} pageSize={50} />
}
