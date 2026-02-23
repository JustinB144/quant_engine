import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, REGIME_COLORS, REGIME_NAMES } from './theme'

interface RegimeTimelineProps {
  dates: string[]
  regimes: number[]
  height?: number
}

export default function RegimeTimeline({ dates, regimes, height = 100 }: RegimeTimelineProps) {
  const option = useMemo(() => {
    const data = regimes.map((r, i) => ({
      value: [i, 0, i + 1, 1],
      itemStyle: { color: REGIME_COLORS[r] || '#8b949e' },
    }))

    return {
      ...bloombergDarkTheme,
      tooltip: {
        ...bloombergDarkTheme.tooltip,
        formatter: (params: { dataIndex: number }) => {
          const idx = params.dataIndex
          return `${dates[idx]}: ${REGIME_NAMES[regimes[idx]] || `Regime ${regimes[idx]}`}`
        },
      },
      grid: { left: 50, right: 20, top: 10, bottom: 30 },
      xAxis: {
        type: 'category' as const,
        data: dates,
        ...bloombergDarkTheme.xAxis,
        axisLabel: { ...bloombergDarkTheme.xAxis.axisLabel, interval: Math.floor(dates.length / 6) },
      },
      yAxis: { type: 'value' as const, show: false, max: 1 },
      series: [
        {
          type: 'bar' as const,
          data: regimes.map((r) => ({
            value: 1,
            itemStyle: { color: REGIME_COLORS[r] || '#8b949e' },
          })),
          barWidth: '100%',
          barCategoryGap: '0%',
        },
      ],
    }
  }, [dates, regimes, height])

  return <ReactECharts option={option} style={{ height }} notMerge />
}
