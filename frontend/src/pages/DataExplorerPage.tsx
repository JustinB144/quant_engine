import React, { useState } from 'react'
import PageContainer from '@/components/layout/PageContainer'
import PageHeader from '@/components/ui/PageHeader'
import UniverseSelector from './DataExplorerPage/UniverseSelector'
import CandlestickPanel from './DataExplorerPage/CandlestickPanel'
import DataQualityReport from './DataExplorerPage/DataQualityReport'
import CacheStatusPanel from './DataExplorerPage/CacheStatusPanel'
import TradingViewWidget from '@/components/charts/TradingViewWidget'
import ChartContainer from '@/components/charts/ChartContainer'
import { useFilterStore } from '@/store/filterStore'

export default function DataExplorerPage() {
  const selectedTicker = useFilterStore((s) => s.selectedTicker)
  const [showTradingView, setShowTradingView] = useState(false)

  return (
    <PageContainer>
      <PageHeader title="Data Explorer" subtitle="OHLCV data visualization, technical indicators, and quality checks" />
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

          {/* TradingView reference toggle */}
          {selectedTicker && (
            <div className="mt-3">
              <button
                onClick={() => setShowTradingView(!showTradingView)}
                className="font-mono px-3 py-1.5 rounded-md"
                style={{
                  fontSize: 11,
                  backgroundColor: showTradingView ? 'rgba(88, 166, 255, 0.1)' : 'var(--bg-secondary)',
                  color: showTradingView ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  border: `1px solid ${showTradingView ? 'var(--accent-blue)' : 'var(--border)'}`,
                  cursor: 'pointer',
                }}
              >
                {showTradingView ? 'Hide' : 'Show'} TradingView Reference
              </button>

              {showTradingView && (
                <div className="mt-2">
                  <ChartContainer title={`${selectedTicker} — TradingView Market Reference`}>
                    <TradingViewWidget
                      symbol={selectedTicker}
                      theme="dark"
                      interval="D"
                      height={500}
                    />
                  </ChartContainer>
                  <p className="font-mono mt-1" style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
                    Market reference data provided by TradingView. The chart above uses your local cached data.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </PageContainer>
  )
}
