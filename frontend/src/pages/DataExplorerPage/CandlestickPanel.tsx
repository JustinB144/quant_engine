import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import CandlestickChart from '@/components/charts/CandlestickChart'
import { useTickerDetail } from '@/api/queries/useData'

interface Props {
  ticker: string
  years?: number
}

export default function CandlestickPanel({ ticker, years = 2 }: Props) {
  const { data, isLoading, error } = useTickerDetail(ticker, years)
  const bars = data?.data?.bars ?? []

  return (
    <ChartContainer
      title={ticker ? `${ticker} — OHLCV Chart` : 'Select a ticker'}
      meta={data?.meta}
      isLoading={isLoading}
      error={error?.message}
      isEmpty={!ticker || bars.length === 0}
      emptyMessage={ticker ? 'No data available for this ticker' : 'Select a ticker from the universe'}
    >
      <CandlestickChart bars={bars} height={500} />
    </ChartContainer>
  )
}
