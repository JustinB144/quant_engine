import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, CHART_COLORS } from './theme'

interface Series {
  name: string
  data: (number | null)[]
  yAxisIndex?: number
  color?: string
  type?: 'line' | 'bar'
  areaStyle?: boolean
  dash?: boolean
}

interface DualAxisChartProps {
  categories: string[]
  series: Series[]
  height?: number
  yAxisNames?: [string, string]
}

export default function DualAxisChart({
  categories,
  series,
  height = 380,
  yAxisNames = ['', ''],
}: DualAxisChartProps) {
  const option = useMemo(
    () => ({
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip, trigger: 'axis' as const },
      legend: { ...bloombergDarkTheme.legend, data: series.map((s) => s.name), bottom: 0 },
      grid: { left: 60, right: 60, top: 30, bottom: 50 },
      xAxis: { type: 'category' as const, data: categories, ...bloombergDarkTheme.xAxis },
      yAxis: [
        { type: 'value' as const, name: yAxisNames[0], ...bloombergDarkTheme.yAxis },
        { type: 'value' as const, name: yAxisNames[1], ...bloombergDarkTheme.yAxis },
      ],
      dataZoom: [{ type: 'inside' }],
      series: series.map((s, i) => ({
        name: s.name,
        type: s.type || 'line',
        yAxisIndex: s.yAxisIndex || 0,
        data: s.data,
        symbol: 'none',
        lineStyle: {
          color: s.color || CHART_COLORS[i % CHART_COLORS.length],
          width: s.dash ? 1 : 2,
          type: s.dash ? ('dashed' as const) : ('solid' as const),
        },
        itemStyle: { color: s.color || CHART_COLORS[i % CHART_COLORS.length] },
        areaStyle: s.areaStyle
          ? {
              color: s.color || CHART_COLORS[i % CHART_COLORS.length],
              opacity: 0.15,
            }
          : undefined,
      })),
    }),
    [categories, series, height, yAxisNames],
  )

  return <ReactECharts option={option} style={{ height }} notMerge />
}
