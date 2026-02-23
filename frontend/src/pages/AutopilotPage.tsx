import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import TabGroup from '@/components/ui/TabGroup'
import StrategyCandidatesTab from './AutopilotPage/StrategyCandidatesTab'
import PaperTradingTab from './AutopilotPage/PaperTradingTab'
import KalshiEventsTab from './AutopilotPage/KalshiEventsTab'
import LifecycleTimeline from './AutopilotPage/LifecycleTimeline'

const TABS = [
  { key: 'strategies', label: 'Strategy Candidates' },
  { key: 'paper', label: 'Paper Trading' },
  { key: 'kalshi', label: 'Kalshi Events' },
  { key: 'lifecycle', label: 'Lifecycle' },
]

export default function AutopilotPage() {
  const [tab, setTab] = useState('strategies')

  return (
    <PageContainer>
      <PageHeader title="Autopilot & Events" subtitle="Automated strategy lifecycle management" />
      <TabGroup tabs={TABS} activeKey={tab} onChange={setTab} />
      {tab === 'strategies' && <StrategyCandidatesTab />}
      {tab === 'paper' && <PaperTradingTab />}
      {tab === 'kalshi' && <KalshiEventsTab />}
      {tab === 'lifecycle' && <LifecycleTimeline />}
    </PageContainer>
  )
}
