import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import ScatterChart from '@/components/charts/ScatterChart'
import { REGIME_COLORS } from '@/components/charts/theme'
import type { SignalRow } from '@/types/signals'

export default function ConfidenceScatter({ signals }: { signals: SignalRow[] }) {
  const data = signals.map((s) => ({
    x: s.predicted_return,
    y: s.confidence,
    color: REGIME_COLORS[typeof s.regime === 'number' ? s.regime : parseInt(String(s.regime), 10)] ?? '#8b949e',
    size: 8,
  }))

  return (
    <ChartContainer title="Confidence vs Predicted Return" isEmpty={data.length === 0}>
      <ScatterChart data={data} height={300} xAxisName="Predicted Return" yAxisName="Confidence" />
    </ChartContainer>
  )
}
