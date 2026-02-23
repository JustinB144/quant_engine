import React, { useMemo } from 'react'
import MetricCard from '@/components/ui/MetricCard'
import { useTickerDetail } from '@/api/queries/useData'
import type { OHLCVBar } from '@/types/data'

interface Props {
  ticker: string
  years?: number
}

/** Compute market stats from OHLCV bars: last price, day change, 52W hi/lo, avg volume. */
function computeStats(bars: OHLCVBar[]) {
  if (!bars.length) return null

  const sorted = [...bars].sort((a, b) => a.date.localeCompare(b.date))
  const lastBar = sorted[sorted.length - 1]
  const prevBar = sorted.length > 1 ? sorted[sorted.length - 2] : null

  const lastPrice = lastBar.close
  const dayChange = prevBar ? (lastBar.close - prevBar.close) / prevBar.close : 0

  // 52-week high/low: roughly 252 trading days
  const recentBars = sorted.slice(-252)
  const high52W = Math.max(...recentBars.map((b) => b.high))
  const low52W = Math.min(...recentBars.map((b) => b.low))

  // Average volume over last 20 days
  const volBars = sorted.slice(-20)
  const avgVol = volBars.reduce((sum, b) => sum + b.volume, 0) / volBars.length

  return { lastPrice, dayChange, high52W, low52W, avgVol }
}

function formatLargeNumber(val: number): string {
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(1)}M`
  if (val >= 1_000) return `${(val / 1_000).toFixed(1)}K`
  return val.toFixed(0)
}

export default function DataQualityReport({ ticker, years = 2 }: Props) {
  const { data } = useTickerDetail(ticker, years)
  const detail = data?.data

  const stats = useMemo(() => computeStats(detail?.bars ?? []), [detail?.bars])

  if (!ticker || !detail) return null

  return (
    <div>
      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-5 gap-3 mb-3">
          <MetricCard
            label="Last Price"
            value={`$${stats.lastPrice.toFixed(2)}`}
            color="var(--accent-blue)"
          />
          <MetricCard
            label="Day Change"
            value={`${stats.dayChange >= 0 ? '+' : ''}${(stats.dayChange * 100).toFixed(2)}%`}
            color={stats.dayChange >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
          />
          <MetricCard
            label="52W High"
            value={`$${stats.high52W.toFixed(2)}`}
            color="var(--accent-green)"
            subtitle={`Low: $${stats.low52W.toFixed(2)}`}
          />
          <MetricCard
            label="Avg Volume (20D)"
            value={formatLargeNumber(stats.avgVol)}
            color="var(--text-secondary)"
          />
          <MetricCard
            label="Data Quality"
            value={detail.data_quality ?? 'N/A'}
            color={detail.data_quality === 'GOOD' ? 'var(--accent-green)' : 'var(--accent-amber)'}
            subtitle={`${detail.total_bars} bars`}
          />
        </div>
      )}
      {/* Date range info */}
      {detail.start_date && detail.end_date && (
        <div
          className="font-mono flex items-center gap-4 mb-2 px-3 py-2 rounded"
          style={{ fontSize: 10, color: 'var(--text-tertiary)', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-light)' }}
        >
          <span>Range: {detail.start_date.slice(0, 10)} → {detail.end_date.slice(0, 10)}</span>
          <span>|</span>
          <span>Total Bars: {detail.total_bars.toLocaleString()}</span>
        </div>
      )}
    </div>
  )
}
