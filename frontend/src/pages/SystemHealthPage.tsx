import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import Spinner from '@/components/ui/Spinner'
import ErrorPanel from '@/components/ui/ErrorPanel'
import HealthRadar from './SystemHealthPage/HealthRadar'
import HealthScoreCards from './SystemHealthPage/HealthScoreCards'
import HealthCheckPanel from './SystemHealthPage/HealthCheckPanel'
import LineChart from '@/components/charts/LineChart'
import ChartContainer from '@/components/charts/ChartContainer'
import { useDetailedHealth } from '@/api/queries/useHealth'
import { ChevronDown, ChevronRight } from 'lucide-react'

function MethodologyPanel({ methodology }: { methodology?: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="card-panel mb-3">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between"
        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown size={14} style={{ color: 'var(--text-tertiary)' }} /> : <ChevronRight size={14} style={{ color: 'var(--text-tertiary)' }} />}
          <span style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>How Scores Are Calculated</span>
        </div>
      </button>
      {open && (
        <div className="mt-3">
          {methodology ? (
            <p className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {methodology}
            </p>
          ) : (
            <p className="font-mono" style={{ fontSize: 11, color: 'var(--text-tertiary)', lineHeight: 1.6 }}>
              Weighted average of 5 domains: Data Integrity (25%), Signal Quality (25%), Risk Management (20%),
              Execution Quality (15%), Model Governance (15%). Only checks with available data are scored.
              UNAVAILABLE checks are excluded from domain averages. Critical checks are weighted 3x,
              standard 1x, informational 0.5x within each domain.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function HealthTrendChart({ history }: { history?: Array<{ timestamp: string; overall_score: number }> }) {
  if (!history || history.length < 2) return null

  const categories = history.map((h) => {
    const d = new Date(h.timestamp)
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`
  })
  const series = [
    {
      name: 'Overall Score',
      data: history.map((h) => h.overall_score),
      color: '#58a6ff',
    },
  ]

  return (
    <ChartContainer title="Health Score Trend">
      <LineChart
        categories={categories}
        series={series}
        height={180}
        yAxisName="Score"
        yAxisRange={[0, 100]}
      />
    </ChartContainer>
  )
}

export default function SystemHealthPage() {
  const { data, isLoading, error, refetch } = useDetailedHealth()
  const health = data?.data

  return (
    <PageContainer>
      <PageHeader title="System Health" subtitle="Comprehensive system diagnostics and scoring" />

      {isLoading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : error ? (
        <ErrorPanel message={error.message} onRetry={refetch} />
      ) : health ? (
        <>
          <MethodologyPanel methodology={health.overall_methodology} />
          <HealthScoreCards health={health} />
          <HealthTrendChart history={health.history} />
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-5">
              <HealthRadar health={health} />
            </div>
            <div className="col-span-7">
              <HealthCheckPanel title="Data Integrity" checks={[...health.survivorship_checks, ...health.data_quality_checks]} />
              <HealthCheckPanel title="Promotion Gate" checks={health.promotion_checks} />
              <HealthCheckPanel title="Walk-Forward" checks={health.wf_checks} />
              <HealthCheckPanel title="Execution" checks={health.execution_checks} />
              <HealthCheckPanel title="Complexity" checks={health.complexity_checks} />
            </div>
          </div>
          {health.strengths.length > 0 && (
            <div className="card-panel mt-3">
              <div className="card-panel-header">Strengths</div>
              {health.strengths.map((s, i) => (
                <div key={i} className="flex items-start gap-2 mb-1.5">
                  <span
                    className="inline-block rounded-full mt-1 flex-shrink-0"
                    style={{ width: 6, height: 6, backgroundColor: 'var(--accent-green)' }}
                  />
                  <div>
                    <span className="font-mono" style={{ fontSize: 11, color: 'var(--accent-green)' }}>
                      {s.name}
                    </span>
                    {s.detail && (
                      <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 8 }}>
                        — {s.detail}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : null}
    </PageContainer>
  )
}
