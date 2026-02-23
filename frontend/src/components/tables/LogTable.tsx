import React from 'react'
import { createColumnHelper } from '@tanstack/react-table'
import DataTable from './DataTable'
import type { LogEntry } from '@/api/queries/useLogs'

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'var(--text-tertiary)',
  INFO: 'var(--accent-blue)',
  WARNING: 'var(--accent-amber)',
  ERROR: 'var(--accent-red)',
  CRITICAL: 'var(--accent-red)',
}

const col = createColumnHelper<LogEntry>()

const columns = [
  col.accessor('ts', {
    header: 'Timestamp',
    size: 180,
    cell: (info) => (
      <span className="font-mono" style={{ fontSize: 10 }}>
        {new Date(info.getValue() * 1000).toISOString().replace('T', ' ').slice(0, 19)}
      </span>
    ),
  }),
  col.accessor('level', {
    header: 'Level',
    size: 80,
    cell: (info) => (
      <span
        style={{
          color: LEVEL_COLORS[info.getValue()] || 'var(--text-secondary)',
          fontWeight: 600,
        }}
      >
        {info.getValue()}
      </span>
    ),
  }),
  col.accessor('logger', { header: 'Logger', size: 160 }),
  col.accessor('message', { header: 'Message' }),
]

interface LogTableProps {
  data: LogEntry[]
  globalFilter?: string
}

export default function LogTable({ data, globalFilter }: LogTableProps) {
  return (
    <DataTable
      data={data}
      columns={columns}
      enableVirtualization={true}
      globalFilter={globalFilter}
    />
  )
}
