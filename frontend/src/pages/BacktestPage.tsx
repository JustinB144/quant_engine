import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import JobMonitor from '@/components/job/JobMonitor'
import BacktestConfigPanel from './BacktestPage/BacktestConfigPanel'
import RunBacktestButton from './BacktestPage/RunBacktestButton'
import BacktestResults from './BacktestPage/BacktestResults'
import type { BacktestRequest } from '@/types/compute'

export default function BacktestPage() {
  const [config, setConfig] = useState<BacktestRequest>({
    horizon: 10,
    holding_period: 10,
    max_positions: 10,
    entry_threshold: 0.01,
    position_size: 0.1,
  })
  const [jobId, setJobId] = useState<string | null>(null)

  return (
    <PageContainer>
      <PageHeader title="Backtest & Risk" subtitle="Strategy backtesting and risk analysis" />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-3">
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
