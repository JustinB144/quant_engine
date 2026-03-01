import React from 'react'
import type { JobStatus } from '@/types/jobs'

const STATUS_CONFIG: Record<JobStatus, { color: string; label: string }> = {
  queued: { color: 'var(--text-tertiary)', label: 'Queued' },
  running: { color: 'var(--accent-blue)', label: 'Running' },
  succeeded: { color: 'var(--accent-green)', label: 'Succeeded' },
  failed: { color: 'var(--accent-red)', label: 'Failed' },
  cancelled: { color: 'var(--accent-amber)', label: 'Cancelled' },
}

export default function JobStatusBadge({ status }: { status: JobStatus }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.queued

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-mono"
      style={{
        fontSize: 10,
        color: config.color,
        backgroundColor: 'var(--bg-tertiary)',
        border: `1px solid ${config.color}`,
      }}
    >
      <span
        className="inline-block rounded-full"
        style={{
          width: 5,
          height: 5,
          backgroundColor: config.color,
          animation: status === 'running' ? 'pulse 2s ease-in-out infinite' : undefined,
        }}
      />
      {config.label}
    </span>
  )
}
