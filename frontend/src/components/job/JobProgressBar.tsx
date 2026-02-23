import React from 'react'

interface JobProgressBarProps {
  progress: number
  message?: string
}

export default function JobProgressBar({ progress, message }: JobProgressBarProps) {
  const pct = Math.min(100, Math.max(0, progress))

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          {message || 'Processing...'}
        </span>
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div
        className="w-full rounded-full overflow-hidden"
        style={{ height: 6, backgroundColor: 'var(--bg-primary)' }}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            backgroundColor: pct >= 100 ? 'var(--accent-green)' : 'var(--accent-blue)',
          }}
        />
      </div>
    </div>
  )
}
