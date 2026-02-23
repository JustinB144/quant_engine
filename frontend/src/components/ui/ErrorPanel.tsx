import React from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface ErrorPanelProps {
  message: string
  suggestion?: string
  onRetry?: () => void
}

export default function ErrorPanel({ message, suggestion, onRetry }: ErrorPanelProps) {
  return (
    <div
      className="rounded-lg p-5 flex flex-col items-center text-center"
      style={{
        backgroundColor: 'rgba(248, 81, 73, 0.08)',
        border: '1px solid rgba(248, 81, 73, 0.3)',
      }}
    >
      <AlertCircle size={28} style={{ color: 'var(--accent-red)', marginBottom: 12 }} />
      <div style={{ color: 'var(--accent-red)', fontSize: 14, fontWeight: 500, marginBottom: 4 }}>
        {message}
      </div>
      {suggestion && (
        <div style={{ color: 'var(--text-tertiary)', fontSize: 12, marginBottom: 12 }}>
          {suggestion}
        </div>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 rounded-md transition-colors"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
            fontSize: 12,
            cursor: 'pointer',
          }}
        >
          <RefreshCw size={12} />
          Retry
        </button>
      )}
    </div>
  )
}
