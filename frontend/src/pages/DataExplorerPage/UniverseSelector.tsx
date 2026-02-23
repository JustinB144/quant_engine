import React, { useState } from 'react'
import { useUniverse } from '@/api/queries/useData'
import { useFilterStore } from '@/store/filterStore'

export default function UniverseSelector() {
  const { data } = useUniverse()
  const universe = data?.data
  const [universeMode, setUniverseMode] = useState<'full' | 'quick'>('full')
  const tickers = universeMode === 'full' ? (universe?.full_tickers ?? []) : (universe?.quick_tickers ?? [])
  const selectedTicker = useFilterStore((s) => s.selectedTicker)
  const setSelectedTicker = useFilterStore((s) => s.setSelectedTicker)
  const [filterText, setFilterText] = useState('')

  const filtered = filterText
    ? tickers.filter((t) => t.toLowerCase().includes(filterText.toLowerCase()))
    : tickers

  return (
    <div className="card-panel">
      <div className="card-panel-header">Data Selection</div>

      <label className="font-mono block mb-1" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
        Universe ({universeMode === 'full' ? universe?.full_size ?? 0 : universe?.quick_size ?? 0} tickers)
      </label>
      <div className="flex gap-1 mb-3">
        {(['full', 'quick'] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setUniverseMode(mode)}
            className="flex-1 py-1 rounded font-mono"
            style={{
              fontSize: 10,
              backgroundColor: universeMode === mode ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              color: universeMode === mode ? 'var(--text-primary)' : 'var(--text-secondary)',
              border: '1px solid var(--border-light)',
              cursor: 'pointer',
              textTransform: 'capitalize',
            }}
          >
            {mode}
          </button>
        ))}
      </div>

      <label className="font-mono block mb-1" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
        Ticker Search
      </label>
      <input
        type="text"
        value={filterText}
        onChange={(e) => setFilterText(e.target.value)}
        placeholder="Search tickers..."
        className="w-full rounded px-2 py-1.5 mb-3 font-mono"
        style={{
          fontSize: 11,
          backgroundColor: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
        }}
      />

      <div
        className="overflow-y-auto mb-3 rounded"
        style={{
          maxHeight: 300,
          border: '1px solid var(--border-light)',
          backgroundColor: 'var(--bg-primary)',
        }}
      >
        {filtered.map((ticker) => (
          <div
            key={ticker}
            onClick={() => setSelectedTicker(ticker)}
            className="px-3 py-1.5 cursor-pointer font-mono transition-colors"
            style={{
              fontSize: 11,
              backgroundColor: ticker === selectedTicker ? 'var(--bg-active)' : 'transparent',
              color: ticker === selectedTicker ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderLeft: ticker === selectedTicker ? '2px solid var(--accent-blue)' : '2px solid transparent',
            }}
          >
            {ticker}
          </div>
        ))}
      </div>
    </div>
  )
}
