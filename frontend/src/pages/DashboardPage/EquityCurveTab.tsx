import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import LineChart from '@/components/charts/LineChart'
import type { EquityCurveResponse } from '@/types/backtests'
import type { ResponseMeta } from '@/types/api'

interface Props {
  equityCurve?: EquityCurveResponse
  meta?: ResponseMeta
}

export default function EquityCurveTab({ equityCurve, meta }: Props) {
  const points = equityCurve?.points ?? []

  return (
    <div>
      <ChartContainer title="Equity Curve" meta={meta}>
        {points.length > 0 ? (
          <LineChart
            categories={points.map((p) => p.date)}
            series={[{ name: 'Portfolio', data: points.map((p) => p.value), color: '#58a6ff' }]}
            height={420}
            yAxisName="Portfolio Value"
            showRangeSelector
          />
        ) : (
          <div style={{ color: 'var(--text-tertiary)', textAlign: 'center', padding: 40 }}>
            No equity curve data available
          </div>
        )}
      </ChartContainer>
    </div>
  )
}
