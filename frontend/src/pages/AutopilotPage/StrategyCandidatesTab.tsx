import React, { useState } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import BarChart from '@/components/charts/BarChart'
import DataTable from '@/components/tables/DataTable'
import JobMonitor from '@/components/job/JobMonitor'
import { useLatestCycle } from '@/api/queries/useAutopilot'
import { useRunCycle } from '@/api/mutations/useRunCycle'
import { createColumnHelper } from '@tanstack/react-table'
import { Play } from 'lucide-react'
import type { TopDecision } from '@/types/autopilot'

const col = createColumnHelper<TopDecision>()
const columns = [
  col.accessor('candidate.strategy_id', { header: 'Strategy' }),
  col.accessor('score', { header: 'Score', cell: (info) => info.getValue().toFixed(1) }),
  col.accessor('passed', { header: 'Passed', cell: (info) => {
    const v = info.getValue()
    return <span style={{ color: v ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>{v ? 'Yes' : 'No'}</span>
  }}),
  col.accessor('reasons', { header: 'Reasons', cell: (info) => {
    const reasons = info.getValue()
    return <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{reasons.slice(0, 3).join(', ')}</span>
  }}),
]

export default function StrategyCandidatesTab() {
  const { data, isLoading } = useLatestCycle()
  const cycle = data?.data
  const runCycle = useRunCycle()
  const [jobId, setJobId] = useState<string | null>(null)

  const handleRunCycle = () => {
    runCycle.mutate(undefined, {
      onSuccess: (data) => {
        const id = (data.data as { job_id?: string })?.job_id
        if (id) setJobId(id)
      },
    })
  }

  // Promotion funnel from cycle data
  const funnel = cycle ? {
    'Candidates': cycle.n_candidates,
    'Passed': cycle.n_passed,
    'Promoted': cycle.n_promoted,
    'Active': cycle.n_active,
  } : {}
  const funnelLabels = Object.keys(funnel)
  const funnelValues = Object.values(funnel)

  const decisions = cycle?.top_decisions ?? []

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={handleRunCycle}
          disabled={runCycle.isPending}
          className="flex items-center gap-1.5 px-4 py-2 rounded-md"
          style={{ backgroundColor: 'var(--accent-blue)', color: 'var(--text-primary)', fontSize: 12, cursor: runCycle.isPending ? 'wait' : 'pointer', border: 'none' }}
        >
          <Play size={12} /> Run Cycle
        </button>
        {jobId && <JobMonitor jobId={jobId} />}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <ChartContainer title="Promotion Funnel">
          {funnelLabels.length > 0 ? (
            <BarChart categories={funnelLabels} series={[{ name: 'Strategies', data: funnelValues, color: ['#58a6ff', '#d29922', '#3fb950', '#bc8cff'] }]} height={250} showValues />
          ) : (
            <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>No cycle data</div>
          )}
        </ChartContainer>
        <ChartContainer title="Top Strategy Decisions" meta={data?.meta} isLoading={isLoading} isEmpty={decisions.length === 0}>
          <DataTable data={decisions} columns={columns} pageSize={15} />
        </ChartContainer>
      </div>
    </div>
  )
}
