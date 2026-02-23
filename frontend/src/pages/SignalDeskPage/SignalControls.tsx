import React from 'react'
import { useFilterStore } from '@/store/filterStore'
import { usePredict } from '@/api/mutations/usePredict'
import { Play } from 'lucide-react'

interface Props {
  onJobCreated: (jobId: string) => void
}

export default function SignalControls({ onJobCreated }: Props) {
  const horizon = useFilterStore((s) => s.horizon)
  const setHorizon = useFilterStore((s) => s.setHorizon)
  const predict = usePredict()

  const handleGenerate = () => {
    predict.mutate({ horizon, top_n: 50 }, {
      onSuccess: (data) => {
        const jobId = (data.data as { job_id?: string })?.job_id
        if (jobId) onJobCreated(jobId)
      },
    })
  }

  return (
    <div className="flex items-center gap-3 mb-4">
      <label className="font-mono" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
        Horizon:
      </label>
      <select
        value={horizon}
        onChange={(e) => setHorizon(Number(e.target.value))}
        className="rounded px-2 py-1 font-mono"
        style={{
          fontSize: 11,
          backgroundColor: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
        }}
      >
        <option value={5}>5 days</option>
        <option value={10}>10 days</option>
        <option value={20}>20 days</option>
      </select>
      <button
        onClick={handleGenerate}
        disabled={predict.isPending}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md"
        style={{
          backgroundColor: 'var(--accent-blue)',
          color: 'var(--text-primary)',
          fontSize: 12,
          cursor: predict.isPending ? 'wait' : 'pointer',
          opacity: predict.isPending ? 0.6 : 1,
          border: 'none',
        }}
      >
        <Play size={12} />
        Generate Signals
      </button>
    </div>
  )
}
