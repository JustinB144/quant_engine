import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import Spinner from '@/components/ui/Spinner'
import AlertBanner from '@/components/ui/AlertBanner'
import JobMonitor from '@/components/job/JobMonitor'
import SignalControls from './SignalDeskPage/SignalControls'
import SignalRankingsPanel from './SignalDeskPage/SignalRankingsPanel'
import SignalDistribution from './SignalDeskPage/SignalDistribution'
import ConfidenceScatter from './SignalDeskPage/ConfidenceScatter'
import { useLatestSignals } from '@/api/queries/useSignals'
import { useFilterStore } from '@/store/filterStore'

export default function SignalDeskPage() {
  const horizon = useFilterStore((s) => s.horizon)
  const { data, isLoading, error } = useLatestSignals(horizon)
  const [jobId, setJobId] = useState<string | null>(null)

  const signals = data?.data?.signals ?? []
  const meta = data?.meta
  const regimeSuppressed = meta?.regime_suppressed === true
  const suppressedCount = signals.filter((s) => s.regime_suppressed).length

  return (
    <PageContainer>
      <PageHeader title="Signal Desk" subtitle="Trading signal generation and analysis" />
      <SignalControls onJobCreated={setJobId} />
      {jobId && <JobMonitor jobId={jobId} />}

      {regimeSuppressed && (
        <AlertBanner
          severity="warning"
          message="Regime suppression is active — some signals are suppressed"
          detail={`${suppressedCount} of ${signals.length} signals marked as suppressed under the current regime configuration`}
        />
      )}

      {isLoading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : error ? (
        <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
          {error.message}
        </div>
      ) : (
        <>
          <SignalRankingsPanel signals={signals} meta={meta} />
          <div className="grid grid-cols-2 gap-3">
            <SignalDistribution signals={signals} />
            <ConfidenceScatter signals={signals} />
          </div>
        </>
      )}
    </PageContainer>
  )
}
