import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme } from './theme'

interface HeatmapChartProps {
  xLabels: string[]
  yLabels: string[]
  data: number[][]
  height?: number
  showValues?: boolean
  colorRange?: [string, string, string]
}

export default function HeatmapChart({
  xLabels,
  yLabels,
  data,
  height = 380,
  showValues = true,
  colorRange = ['#161b22', '#58a6ff', '#3fb950'],
}: HeatmapChartProps) {
  const option = useMemo(() => {
    const flatData: [number, number, number][] = []
    let minVal = Infinity
    let maxVal = -Infinity
    data.forEach((row, yi) => {
      row.forEach((val, xi) => {
        flatData.push([xi, yi, val])
        if (val < minVal) minVal = val
        if (val > maxVal) maxVal = val
      })
    })

    return {
      ...bloombergDarkTheme,
      tooltip: {
        ...bloombergDarkTheme.tooltip,
        formatter: (params: { data: number[] }) => {
          const [xi, yi, val] = params.data
          const pct = (val * 100).toFixed(1)
          if (xi === yi) {
            return `<b>${yLabels[yi]}</b> persistence: <b>${pct}%</b><br/>Probability of staying in this regime`
          }
          return `<b>${yLabels[yi]}</b> &rarr; <b>${xLabels[xi]}</b>: <b>${pct}%</b><br/>Transition probability`
        },
      },
      grid: { left: 140, right: 60, top: 20, bottom: 40 },
      xAxis: {
        type: 'category' as const,
        data: xLabels,
        splitArea: { show: false },
        ...bloombergDarkTheme.xAxis,
      },
      yAxis: {
        type: 'category' as const,
        data: yLabels,
        splitArea: { show: false },
        ...bloombergDarkTheme.yAxis,
      },
      visualMap: {
        min: minVal,
        max: maxVal,
        calculable: true,
        orient: 'vertical' as const,
        right: 0,
        top: 'center',
        inRange: { color: colorRange },
        textStyle: { color: '#8b949e', fontSize: 10 },
      },
      series: [
        {
          type: 'heatmap' as const,
          data: flatData,
          label: {
            show: showValues,
            fontSize: 9,
            color: '#c9d1d9',
            formatter: (params: { data: number[] }) => params.data[2].toFixed(2),
          },
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' },
          },
        },
      ],
    }
  }, [xLabels, yLabels, data, height, showValues, colorRange])

  return <ReactECharts option={option} style={{ height }} notMerge />
}
