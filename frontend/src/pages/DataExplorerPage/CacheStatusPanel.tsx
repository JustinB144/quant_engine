import React from 'react'
import { useDataStatus } from '@/api/queries/useData'
import type { TickerCacheEntry } from '@/api/queries/useData'

const FRESHNESS_COLORS: Record<string, string> = {
  FRESH: 'var(--accent-green, #22c55e)',
  STALE: 'var(--accent-yellow, #eab308)',
  VERY_STALE: 'var(--accent-red, #ef4444)',
  UNKNOWN: 'var(--text-tertiary)',
}

interface CacheStatusPanelProps {
  selectedTicker?: string
}

export default function CacheStatusPanel({ selectedTicker }: CacheStatusPanelProps) {
  const { data, isLoading, error } = useDataStatus()
  const status = data?.data

  if (isLoading) {
    return (
      <div className="card-panel" style={{ padding: '12px 16px' }}>
        <div className="card-panel-header">Cache Status</div>
        <p className="font-mono" style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
          Scanning cache...
        </p>
      </div>
    )
  }

  if (error || !status) {
    return (
      <div className="card-panel" style={{ padding: '12px 16px' }}>
        <div className="card-panel-header">Cache Status</div>
        <p className="font-mono" style={{ fontSize: 11, color: 'var(--accent-red, #ef4444)' }}>
          Failed to load cache status
        </p>
      </div>
    )
  }

  const { summary, tickers } = status

  // Find the selected ticker's status
  let tickerInfo: TickerCacheEntry | undefined
  if (selectedTicker) {
    tickerInfo = tickers.find(
      (t) => t.ticker.toUpperCase() === selectedTicker.toUpperCase()
    )
  }

  return (
    <div className="card-panel" style={{ padding: '12px 16px' }}>
      <div className="card-panel-header">Cache Status</div>

      {/* Summary bar */}
      <div
        className="flex items-center gap-3 font-mono mb-2"
        style={{ fontSize: 11, color: 'var(--text-secondary)' }}
      >
        <span>{summary.total_cached} tickers cached</span>
        <span style={{ color: 'var(--border-light)' }}>|</span>
        <span style={{ color: FRESHNESS_COLORS.FRESH }}>{summary.fresh} fresh</span>
        <span style={{ color: 'var(--border-light)' }}>|</span>
        <span style={{ color: FRESHNESS_COLORS.STALE }}>{summary.stale} stale</span>
        {summary.very_stale > 0 && (
          <>
            <span style={{ color: 'var(--border-light)' }}>|</span>
            <span style={{ color: FRESHNESS_COLORS.VERY_STALE }}>
              {summary.very_stale} very stale
            </span>
          </>
        )}
      </div>

      {!summary.cache_exists && (
        <div
          className="rounded px-3 py-2 font-mono"
          style={{
            fontSize: 11,
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: 'var(--accent-red, #ef4444)',
          }}
        >
          Cache directory does not exist: {summary.cache_dir}
        </div>
      )}

      {/* Per-ticker detail when selected */}
      {selectedTicker && tickerInfo && (
        <div
          className="rounded px-3 py-2 mt-2"
          style={{
            backgroundColor: 'var(--bg-tertiary)',
            border: '1px solid var(--border-light)',
          }}
        >
          <div className="flex items-center justify-between mb-1">
            <span
              className="font-mono font-semibold"
              style={{ fontSize: 12, color: 'var(--text-primary)' }}
            >
              {tickerInfo.ticker}
            </span>
            <span
              className="font-mono px-2 py-0.5 rounded"
              style={{
                fontSize: 10,
                backgroundColor: `${FRESHNESS_COLORS[tickerInfo.freshness]}20`,
                color: FRESHNESS_COLORS[tickerInfo.freshness],
                border: `1px solid ${FRESHNESS_COLORS[tickerInfo.freshness]}40`,
              }}
            >
              {tickerInfo.freshness}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono" style={{ fontSize: 10 }}>
            <div style={{ color: 'var(--text-tertiary)' }}>Source</div>
            <div style={{ color: 'var(--text-secondary)' }}>{tickerInfo.source}</div>
            <div style={{ color: 'var(--text-tertiary)' }}>Last bar</div>
            <div style={{ color: 'var(--text-secondary)' }}>{tickerInfo.last_bar_date}</div>
            <div style={{ color: 'var(--text-tertiary)' }}>Total bars</div>
            <div style={{ color: 'var(--text-secondary)' }}>{tickerInfo.total_bars.toLocaleString()}</div>
            {tickerInfo.permno && (
              <>
                <div style={{ color: 'var(--text-tertiary)' }}>PERMNO</div>
                <div style={{ color: 'var(--text-secondary)' }}>{tickerInfo.permno}</div>
              </>
            )}
            <div style={{ color: 'var(--text-tertiary)' }}>Timeframes</div>
            <div style={{ color: 'var(--text-secondary)' }}>
              {tickerInfo.timeframes_available.join(', ')}
            </div>
          </div>
          {tickerInfo.freshness === 'STALE' && tickerInfo.days_stale !== null && (
            <div
              className="mt-2 rounded px-2 py-1 font-mono"
              style={{
                fontSize: 10,
                backgroundColor: 'rgba(234, 179, 8, 0.1)',
                border: '1px solid rgba(234, 179, 8, 0.3)',
                color: FRESHNESS_COLORS.STALE,
              }}
            >
              Data may be stale — last updated {tickerInfo.last_bar_date} ({tickerInfo.days_stale}d ago)
            </div>
          )}
          {tickerInfo.freshness === 'VERY_STALE' && tickerInfo.days_stale !== null && (
            <div
              className="mt-2 rounded px-2 py-1 font-mono"
              style={{
                fontSize: 10,
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: FRESHNESS_COLORS.VERY_STALE,
              }}
            >
              Data is very stale — last updated {tickerInfo.last_bar_date} ({tickerInfo.days_stale}d ago)
            </div>
          )}
        </div>
      )}

      {selectedTicker && !tickerInfo && (
        <div
          className="rounded px-3 py-2 mt-2 font-mono"
          style={{
            fontSize: 11,
            backgroundColor: 'rgba(234, 179, 8, 0.1)',
            border: '1px solid rgba(234, 179, 8, 0.3)',
            color: FRESHNESS_COLORS.STALE,
          }}
        >
          {selectedTicker} not found in cache
        </div>
      )}
    </div>
  )
}
