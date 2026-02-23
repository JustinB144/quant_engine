import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import TabGroup from '@/components/ui/TabGroup'
import SVISurfaceTab from './IVSurfacePage/SVISurfaceTab'
import HestonSurfaceTab from './IVSurfacePage/HestonSurfaceTab'
import ArbAwareSVITab from './IVSurfacePage/ArbAwareSVITab'

const TABS = [
  { key: 'svi', label: 'SVI Surface' },
  { key: 'heston', label: 'Heston Surface' },
  { key: 'arb-svi', label: 'Arb-Aware SVI' },
]

export default function IVSurfacePage() {
  const [tab, setTab] = useState('svi')

  return (
    <PageContainer>
      <PageHeader title="IV Surface" subtitle="Implied volatility modeling and arbitrage" />
      <TabGroup tabs={TABS} activeKey={tab} onChange={setTab} />
      {tab === 'svi' && <SVISurfaceTab />}
      {tab === 'heston' && <HestonSurfaceTab />}
      {tab === 'arb-svi' && <ArbAwareSVITab />}
    </PageContainer>
  )
}
