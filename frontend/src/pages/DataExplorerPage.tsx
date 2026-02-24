import React from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import UniverseSelector from './DataExplorerPage/UniverseSelector'
import CandlestickPanel from './DataExplorerPage/CandlestickPanel'
import DataQualityReport from './DataExplorerPage/DataQualityReport'
import CacheStatusPanel from './DataExplorerPage/CacheStatusPanel'
import { useFilterStore } from '@/store/filterStore'

export default function DataExplorerPage() {
  const selectedTicker = useFilterStore((s) => s.selectedTicker)

  return (
    <PageContainer>
      <PageHeader title="Data Explorer" subtitle="OHLCV data visualization and quality checks" />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-3">
          <UniverseSelector />
          <div className="mt-3">
            <CacheStatusPanel selectedTicker={selectedTicker} />
          </div>
        </div>
        <div className="col-span-9">
          {selectedTicker && <DataQualityReport ticker={selectedTicker} />}
          <div className="mt-3">
            <CandlestickPanel ticker={selectedTicker} />
          </div>
        </div>
      </div>
    </PageContainer>
  )
}
