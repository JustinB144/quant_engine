import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme } from './theme'

interface RadarChartProps {
  indicators: { name: string; max: number }[]
  values: number[]
  height?: number
  color?: string
}

export default function RadarChart({
  indicators,
  values,
  height = 360,
  color = '#58a6ff',
}: RadarChartProps) {
  const option = useMemo(
    () => ({
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip },
      radar: {
        indicator: indicators,
        shape: 'polygon' as const,
        axisName: { color: '#8b949e', fontSize: 10 },
        splitArea: { areaStyle: { color: ['#161b22', '#1c2028'] } },
        splitLine: { lineStyle: { color: '#21262d' } },
        axisLine: { lineStyle: { color: '#30363d' } },
      },
      series: [
        {
          type: 'radar' as const,
          data: [
            {
              value: values,
              areaStyle: { color, opacity: 0.2 },
              lineStyle: { color, width: 2 },
              itemStyle: { color },
            },
          ],
        },
      ],
    }),
    [indicators, values, height, color],
  )

  return <ReactECharts option={option} style={{ height }} notMerge />
}
