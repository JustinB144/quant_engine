import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, CHART_COLORS } from './theme'

interface Series {
  name: string
  data: (number | null)[]
  color?: string
  dash?: boolean
}

interface LineChartProps {
  categories: string[]
  series: Series[]
  height?: number
  yAxisName?: string
  showRangeSelector?: boolean
  yAxisRange?: [number, number]
}

export default function LineChart({
  categories,
  series,
  height = 360,
  yAxisName,
  showRangeSelector = false,
  yAxisRange,
}: LineChartProps) {
  const option = useMemo(
    () => ({
      ...bloombergDarkTheme,
      tooltip: {
        ...bloombergDarkTheme.tooltip,
        trigger: 'axis' as const,
      },
      legend: {
        ...bloombergDarkTheme.legend,
        data: series.map((s) => s.name),
        bottom: 0,
      },
      grid: { left: 50, right: 20, top: 30, bottom: showRangeSelector ? 80 : 40 },
      xAxis: {
        type: 'category' as const,
        data: categories,
        ...bloombergDarkTheme.xAxis,
      },
      yAxis: {
        type: 'value' as const,
        name: yAxisName,
        min: yAxisRange?.[0],
        max: yAxisRange?.[1],
        ...bloombergDarkTheme.yAxis,
      },
      dataZoom: showRangeSelector
        ? [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, bottom: 10, height: 20 },
          ]
        : [{ type: 'inside' }],
      series: series.map((s, i) => ({
        name: s.name,
        type: 'line' as const,
        data: s.data,
        smooth: false,
        symbol: 'none',
        lineStyle: {
          color: s.color || CHART_COLORS[i % CHART_COLORS.length],
          width: 2,
          type: s.dash ? ('dashed' as const) : ('solid' as const),
        },
        itemStyle: { color: s.color || CHART_COLORS[i % CHART_COLORS.length] },
      })),
    }),
    [categories, series, height, yAxisName, showRangeSelector, yAxisRange],
  )

  return <ReactECharts option={option} style={{ height }} notMerge />
}
