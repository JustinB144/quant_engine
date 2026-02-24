import React, { useState, useEffect } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import JobMonitor from '@/components/job/JobMonitor'
import BacktestConfigPanel from './BacktestPage/BacktestConfigPanel'
import RunBacktestButton from './BacktestPage/RunBacktestButton'
import BacktestResults from './BacktestPage/BacktestResults'
import { useConfig } from '@/api/queries/useConfig'
import type { BacktestRequest } from '@/types/compute'

const FALLBACK_DEFAULTS: BacktestRequest = {
  horizon: 10,
  holding_period: 10,
  max_positions: 10,
  entry_threshold: 0.01,
  position_size: 0.1,
}

export default function BacktestPage() {
  const { data: configData } = useConfig()
  const [config, setConfig] = useState<BacktestRequest>(FALLBACK_DEFAULTS)
  const [usingServerDefaults, setUsingServerDefaults] = useState(false)

  useEffect(() => {
    if (!configData?.data) return
    const srv = configData.data
    setConfig({
      horizon: 10,
      holding_period: typeof srv.MAX_HOLDING_DAYS === 'number' ? srv.MAX_HOLDING_DAYS : FALLBACK_DEFAULTS.holding_period,
      max_positions: typeof srv.MAX_POSITIONS === 'number' ? srv.MAX_POSITIONS : FALLBACK_DEFAULTS.max_positions,
      entry_threshold: typeof srv.ENTRY_THRESHOLD === 'number' ? srv.ENTRY_THRESHOLD : FALLBACK_DEFAULTS.entry_threshold,
      position_size: typeof srv.POSITION_SIZE_PCT === 'number' ? srv.POSITION_SIZE_PCT : FALLBACK_DEFAULTS.position_size,
    })
    setUsingServerDefaults(true)
  }, [configData])

  const [jobId, setJobId] = useState<string | null>(null)

  return (
    <PageContainer>
      <PageHeader title="Backtest & Risk" subtitle="Strategy backtesting and risk analysis" />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-3">
          {usingServerDefaults && (
            <div className="mb-2 px-2 py-1 rounded font-mono"
              style={{ fontSize: 10, color: 'var(--accent-green)', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--accent-green)', opacity: 0.7 }}>
              Using server defaults
            </div>
          )}
          <BacktestConfigPanel config={config} onChange={setConfig} />
          <RunBacktestButton config={config} onJobCreated={setJobId} />
          {jobId && <JobMonitor jobId={jobId} />}
        </div>
        <div className="col-span-9">
          <BacktestResults />
        </div>
      </div>
    </PageContainer>
  )
}
