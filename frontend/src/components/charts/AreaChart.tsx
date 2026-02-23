import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, CHART_COLORS } from './theme'

interface Series {
  name: string
  data: (number | null)[]
  color?: string
  stack?: string
}

interface AreaChartProps {
  categories: string[]
  series: Series[]
  height?: number
  yAxisName?: string
  yAxisRange?: [number, number]
  showRangeSelector?: boolean
}

export default function AreaChart({
  categories,
  series,
  height = 360,
  yAxisName,
  yAxisRange,
  showRangeSelector = false,
}: AreaChartProps) {
  const option = useMemo(
    () => ({
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip, trigger: 'axis' as const },
      legend: { ...bloombergDarkTheme.legend, data: series.map((s) => s.name), bottom: 0 },
      grid: { left: 50, right: 20, top: 30, bottom: showRangeSelector ? 80 : 40 },
      xAxis: { type: 'category' as const, data: categories, ...bloombergDarkTheme.xAxis },
      yAxis: {
        type: 'value' as const,
        name: yAxisName,
        min: yAxisRange?.[0],
        max: yAxisRange?.[1],
        ...bloombergDarkTheme.yAxis,
      },
      dataZoom: showRangeSelector
        ? [{ type: 'inside' }, { type: 'slider', bottom: 10, height: 20 }]
        : [{ type: 'inside' }],
      series: series.map((s, i) => ({
        name: s.name,
        type: 'line' as const,
        data: s.data,
        stack: s.stack || 'one',
        areaStyle: {
          opacity: 0.6,
          color: s.color || CHART_COLORS[i % CHART_COLORS.length],
        },
        lineStyle: {
          width: 0.5,
          color: s.color || CHART_COLORS[i % CHART_COLORS.length],
        },
        symbol: 'none',
        itemStyle: { color: s.color || CHART_COLORS[i % CHART_COLORS.length] },
      })),
    }),
    [categories, series, height, yAxisName, yAxisRange, showRangeSelector],
  )

  return <ReactECharts option={option} style={{ height }} notMerge />
}
