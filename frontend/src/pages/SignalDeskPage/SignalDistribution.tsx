import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import HistogramChart from '@/components/charts/HistogramChart'
import type { SignalRow } from '@/types/signals'

export default function SignalDistribution({ signals }: { signals: SignalRow[] }) {
  const returns = signals.map((s) => s.predicted_return)

  return (
    <ChartContainer title="Predicted Return Distribution" isEmpty={returns.length === 0}>
      <HistogramChart data={returns} bins={30} height={300} xAxisName="Predicted Return" />
    </ChartContainer>
  )
}
