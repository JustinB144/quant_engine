import React from 'react'
import MetricCard from '@/components/ui/MetricCard'
import { useTickerDetail } from '@/api/queries/useData'

interface Props {
  ticker: string
  years?: number
}

export default function DataQualityReport({ ticker, years = 2 }: Props) {
  const { data } = useTickerDetail(ticker, years)
  const detail = data?.data

  if (!ticker || !detail) return null

  return (
    <div className="grid grid-cols-3 gap-3">
      <MetricCard label="Total Bars" value={String(detail.total_bars)} color="var(--accent-blue)" />
      <MetricCard
        label="Date Range"
        value={`${detail.start_date.slice(0, 10)} → ${detail.end_date.slice(0, 10)}`}
        color="var(--text-secondary)"
      />
      <MetricCard
        label="Data Quality"
        value={detail.data_quality ?? 'N/A'}
        color={detail.data_quality === 'GOOD' ? 'var(--accent-green)' : 'var(--accent-amber)'}
      />
    </div>
  )
}
