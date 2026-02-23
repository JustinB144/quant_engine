import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import TabGroup from '@/components/ui/TabGroup'
import Spinner from '@/components/ui/Spinner'
import ErrorPanel from '@/components/ui/ErrorPanel'
import KPIGrid from './DashboardPage/KPIGrid'
import EquityCurveTab from './DashboardPage/EquityCurveTab'
import RegimeTab from './DashboardPage/RegimeTab'
import ModelHealthTab from './DashboardPage/ModelHealthTab'
import FeatureImportanceTab from './DashboardPage/FeatureImportanceTab'
import TradeLogTab from './DashboardPage/TradeLogTab'
import RiskTab from './DashboardPage/RiskTab'
import { useDashboardData } from './DashboardPage/useDashboardData'
import { RefreshCw } from 'lucide-react'

const TABS = [
  { key: 'portfolio', label: 'Portfolio Overview' },
  { key: 'regime', label: 'Regime State' },
  { key: 'model', label: 'Model Performance' },
  { key: 'features', label: 'Feature Importance' },
  { key: 'trades', label: 'Trade Log' },
  { key: 'risk', label: 'Risk Metrics' },
]

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState('portfolio')
  const { summary, regime, modelHealth, featureImportance, equityCurve, trades, isLoading, error, meta, refetch } = useDashboardData()

  return (
    <PageContainer>
      <PageHeader
        title="Portfolio Intelligence Dashboard"
        actions={
          <div className="flex items-center gap-3">
            {meta?.generated_at && (
              <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                Updated {new Date(meta.generated_at).toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={refetch}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-md"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              <RefreshCw size={12} />
              Refresh
            </button>
          </div>
        }
      />

      {isLoading && !summary ? (
        <div className="flex items-center justify-center py-20">
          <Spinner size={32} />
        </div>
      ) : error && !summary ? (
        <ErrorPanel message={error} suggestion="Check that the backend is running on port 8000" onRetry={refetch} />
      ) : summary ? (
        <>
          <KPIGrid summary={summary} regime={regime} modelHealth={modelHealth} />
          <TabGroup tabs={TABS} activeKey={activeTab} onChange={setActiveTab} />

          {activeTab === 'portfolio' && <EquityCurveTab equityCurve={equityCurve} meta={meta} />}
          {activeTab === 'regime' && <RegimeTab regime={regime} meta={meta} />}
          {activeTab === 'model' && <ModelHealthTab modelHealth={modelHealth} />}
          {activeTab === 'features' && <FeatureImportanceTab featureImportance={featureImportance} />}
          {activeTab === 'trades' && <TradeLogTab trades={trades} meta={meta} />}
          {activeTab === 'risk' && <RiskTab summary={summary} meta={meta} />}
        </>
      ) : null}
    </PageContainer>
  )
}
