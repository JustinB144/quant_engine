import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, CHART_COLORS } from './theme'

interface Series {
  name: string
  data: number[]
  color?: string | string[]
  stack?: string
}

interface BarChartProps {
  categories: string[]
  series: Series[]
  height?: number
  horizontal?: boolean
  yAxisName?: string
  xAxisName?: string
  showValues?: boolean
}

export default function BarChart({
  categories,
  series,
  height = 340,
  horizontal = false,
  yAxisName,
  xAxisName,
  showValues = false,
}: BarChartProps) {
  const option = useMemo(() => {
    const catAxis = {
      type: 'category' as const,
      data: categories,
      ...bloombergDarkTheme.xAxis,
    }
    const valAxis = {
      type: 'value' as const,
      name: horizontal ? xAxisName : yAxisName,
      ...bloombergDarkTheme.yAxis,
    }

    return {
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip, trigger: 'axis' as const },
      legend: series.length > 1
        ? { ...bloombergDarkTheme.legend, data: series.map((s) => s.name), bottom: 0 }
        : undefined,
      grid: {
        left: horizontal ? 140 : 50,
        right: 20,
        top: 30,
        bottom: series.length > 1 ? 40 : 20,
      },
      xAxis: horizontal ? valAxis : catAxis,
      yAxis: horizontal ? catAxis : valAxis,
      series: series.map((s, i) => ({
        name: s.name,
        type: 'bar' as const,
        data: s.data,
        stack: s.stack,
        barMaxWidth: 40,
        itemStyle: {
          color: Array.isArray(s.color)
            ? undefined
            : s.color || CHART_COLORS[i % CHART_COLORS.length],
        },
        ...(Array.isArray(s.color) && {
          data: s.data.map((v, j) => ({
            value: v,
            itemStyle: { color: (s.color as string[])[j] || CHART_COLORS[j % CHART_COLORS.length] },
          })),
        }),
        label: showValues
          ? {
              show: true,
              position: horizontal ? 'right' : 'top',
              formatter: '{c}',
              fontSize: 10,
              color: '#c9d1d9',
            }
          : undefined,
      })),
    }
  }, [categories, series, height, horizontal, yAxisName, xAxisName, showValues])

  return <ReactECharts option={option} style={{ height }} notMerge />
}
