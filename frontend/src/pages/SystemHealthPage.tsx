import React from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import Spinner from '@/components/ui/Spinner'
import ErrorPanel from '@/components/ui/ErrorPanel'
import HealthRadar from './SystemHealthPage/HealthRadar'
import HealthScoreCards from './SystemHealthPage/HealthScoreCards'
import HealthCheckPanel from './SystemHealthPage/HealthCheckPanel'
import { useDetailedHealth } from '@/api/queries/useHealth'

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
          <HealthScoreCards health={health} />
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
