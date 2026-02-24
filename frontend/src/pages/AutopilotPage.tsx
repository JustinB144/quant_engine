import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import TabGroup from '@/components/ui/TabGroup'
import AlertBanner from '@/components/ui/AlertBanner'
import StrategyCandidatesTab from './AutopilotPage/StrategyCandidatesTab'
import PaperTradingTab from './AutopilotPage/PaperTradingTab'
import PaperPnLTracker from './AutopilotPage/PaperPnLTracker'
import KalshiEventsTab from './AutopilotPage/KalshiEventsTab'
import LifecycleTimeline from './AutopilotPage/LifecycleTimeline'
import { useLatestCycle } from '@/api/queries/useAutopilot'

const TABS = [
  { key: 'strategies', label: 'Strategy Candidates' },
  { key: 'paper', label: 'Paper Trading' },
  { key: 'pnl', label: 'Live P&L' },
  { key: 'kalshi', label: 'Kalshi Events' },
  { key: 'lifecycle', label: 'Lifecycle' },
]

export default function AutopilotPage() {
  const [tab, setTab] = useState('strategies')
  const { data } = useLatestCycle()
  const meta = data?.meta

  const predictorType = meta?.predictor_type
  const isHeuristic = predictorType === 'heuristic'
  const walkForwardMode = meta?.walk_forward_mode

  return (
    <PageContainer>
      <PageHeader
        title="Autopilot & Events"
        subtitle="Automated strategy lifecycle management"
        actions={
          <div className="flex items-center gap-3">
            {predictorType && (
              <span
                className="font-mono uppercase px-2.5 py-1 rounded"
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  backgroundColor: isHeuristic ? 'rgba(248,81,73,0.15)' : 'rgba(63,185,80,0.15)',
                  color: isHeuristic ? 'var(--accent-red)' : 'var(--accent-green)',
                }}
              >
                {isHeuristic ? 'Heuristic Predictor' : 'Ensemble Model'}
              </span>
            )}
            {walkForwardMode && (
              <span
                className="font-mono px-2.5 py-1 rounded"
                style={{
                  fontSize: 10,
                  backgroundColor: walkForwardMode === 'full' ? 'rgba(63,185,80,0.1)' : 'rgba(210,153,34,0.1)',
                  color: walkForwardMode === 'full' ? 'var(--accent-green)' : 'var(--accent-amber)',
                }}
              >
                {walkForwardMode === 'full' ? 'Full Walk-Forward' : 'Single Split'}
              </span>
            )}
            {meta?.model_version && (
              <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                Model: {meta.model_version.slice(0, 8)}
              </span>
            )}
          </div>
        }
      />

      {isHeuristic && (
        <AlertBanner
          severity="error"
          message="Running in heuristic mode — no trained ensemble model found"
          detail="Autopilot decisions are based on rule-based heuristics. Train a model in Model Lab for ML-driven predictions."
        />
      )}

      <TabGroup tabs={TABS} activeKey={tab} onChange={setTab} />
      {tab === 'strategies' && <StrategyCandidatesTab />}
      {tab === 'paper' && <PaperTradingTab />}
      {tab === 'pnl' && <PaperPnLTracker />}
      {tab === 'kalshi' && <KalshiEventsTab />}
      {tab === 'lifecycle' && <LifecycleTimeline />}
    </PageContainer>
  )
}
