import React, { useRef } from 'react'
import { useJobProgress } from '@/hooks/useJobProgress'
import JobProgressBar from './JobProgressBar'
import JobStatusBadge from './JobStatusBadge'

interface JobMonitorProps {
  jobId: string | null
  onComplete?: (result: unknown) => void
}

export default function JobMonitor({ jobId, onComplete }: JobMonitorProps) {
  const { progress, progress_message, status, result, error, isActive } = useJobProgress(jobId)
  const completedRef = useRef(false)

  // Reset when a new job starts
  React.useEffect(() => {
    completedRef.current = false
  }, [jobId])

  React.useEffect(() => {
    if (status === 'succeeded' && result && onComplete && !completedRef.current) {
      completedRef.current = true
      onComplete(result)
    }
  }, [status, result, onComplete])

  if (!jobId) return null

  return (
    <div
      className="rounded-lg p-3 mt-3"
      style={{
        backgroundColor: 'var(--bg-tertiary)',
        border: '1px solid var(--border-light)',
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          Job: {jobId.slice(0, 8)}...
        </span>
        <JobStatusBadge status={status} />
      </div>

      {isActive && <JobProgressBar progress={progress} message={progress_message} />}

      {error && (
        <div className="mt-2 font-mono" style={{ fontSize: 11, color: 'var(--accent-red)' }}>
          Error: {error}
        </div>
      )}

      {status === 'succeeded' && (
        <div className="mt-2 font-mono" style={{ fontSize: 11, color: 'var(--accent-green)' }}>
          Completed successfully
        </div>
      )}
    </div>
  )
}
