import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import TabGroup from '@/components/ui/TabGroup'
import FeaturesTab from './ModelLabPage/FeaturesTab'
import FeatureDiffViewer from './ModelLabPage/FeatureDiffViewer'
import RegimeTab from './ModelLabPage/RegimeTab'
import TrainingTab from './ModelLabPage/TrainingTab'

const TABS = [
  { key: 'features', label: 'Features' },
  { key: 'diff', label: 'Feature Diff' },
  { key: 'regime', label: 'Regime' },
  { key: 'training', label: 'Training' },
]

export default function ModelLabPage() {
  const [tab, setTab] = useState('features')

  return (
    <PageContainer>
      <PageHeader title="Model Lab" subtitle="Feature analysis, regime detection, and model training" />
      <TabGroup tabs={TABS} activeKey={tab} onChange={setTab} />
      {tab === 'features' && <FeaturesTab />}
      {tab === 'diff' && <FeatureDiffViewer />}
      {tab === 'regime' && <RegimeTab />}
      {tab === 'training' && <TrainingTab />}
    </PageContainer>
  )
}
