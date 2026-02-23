import React from 'react'
import { useRunBacktest } from '@/api/mutations/useRunBacktest'
import { Play } from 'lucide-react'
import type { BacktestRequest } from '@/types/compute'

interface Props {
  config: BacktestRequest
  onJobCreated: (jobId: string) => void
}

export default function RunBacktestButton({ config, onJobCreated }: Props) {
  const runBacktest = useRunBacktest()

  const handleRun = () => {
    runBacktest.mutate(config, {
      onSuccess: (data) => {
        const jobId = (data.data as { job_id?: string })?.job_id
        if (jobId) onJobCreated(jobId)
      },
    })
  }

  return (
    <button
      onClick={handleRun}
      disabled={runBacktest.isPending}
      className="flex items-center gap-1.5 px-4 py-2 rounded-md w-full justify-center mt-3"
      style={{
        backgroundColor: 'var(--accent-blue)',
        color: 'var(--text-primary)',
        fontSize: 12,
        fontWeight: 500,
        cursor: runBacktest.isPending ? 'wait' : 'pointer',
        opacity: runBacktest.isPending ? 0.6 : 1,
        border: 'none',
      }}
    >
      <Play size={14} />
      {runBacktest.isPending ? 'Submitting...' : 'Run Backtest'}
    </button>
  )
}
