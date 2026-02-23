import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme } from './theme'

interface VarLine {
  value: number
  label: string
  color: string
}

interface HistogramChartProps {
  data: number[]
  bins?: number
  height?: number
  xAxisName?: string
  varLines?: VarLine[]
}

function computeBins(data: number[], nBins: number): { edges: number[]; counts: number[] } {
  if (!data.length) return { edges: [], counts: [] }
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const binWidth = range / nBins
  const edges: number[] = []
  const counts: number[] = new Array(nBins).fill(0)

  for (let i = 0; i <= nBins; i++) {
    edges.push(min + i * binWidth)
  }
  for (const v of data) {
    let idx = Math.floor((v - min) / binWidth)
    if (idx >= nBins) idx = nBins - 1
    if (idx < 0) idx = 0
    counts[idx]++
  }
  return { edges, counts }
}

export default function HistogramChart({
  data,
  bins = 60,
  height = 380,
  xAxisName,
  varLines = [],
}: HistogramChartProps) {
  const option = useMemo(() => {
    const { edges, counts } = computeBins(data, bins)
    const labels = edges.slice(0, -1).map((e, i) => {
      const mid = (e + edges[i + 1]) / 2
      return (mid * 100).toFixed(2) + '%'
    })

    return {
      ...bloombergDarkTheme,
      tooltip: { ...bloombergDarkTheme.tooltip, trigger: 'axis' as const },
      grid: { left: 50, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: 'category' as const,
        data: labels,
        ...bloombergDarkTheme.xAxis,
        axisLabel: { ...bloombergDarkTheme.xAxis.axisLabel, rotate: 45, interval: Math.floor(bins / 10) },
        name: xAxisName,
      },
      yAxis: {
        type: 'value' as const,
        name: 'Frequency',
        ...bloombergDarkTheme.yAxis,
      },
      series: [
        {
          type: 'bar' as const,
          data: counts,
          itemStyle: { color: '#58a6ff', opacity: 0.75 },
          barCategoryGap: '5%',
          markLine: {
            symbol: 'none',
            data: varLines.map((vl) => {
              // Find closest bin index
              const idx = edges.findIndex((e, i) => i < edges.length - 1 && vl.value >= e && vl.value < edges[i + 1])
              return {
                xAxis: Math.max(0, idx),
                label: {
                  formatter: vl.label,
                  color: vl.color,
                  fontSize: 10,
                },
                lineStyle: {
                  color: vl.color,
                  type: 'dashed' as const,
                  width: 2,
                },
              }
            }),
          },
        },
      ],
    }
  }, [data, bins, height, xAxisName, varLines])

  return <ReactECharts option={option} style={{ height }} notMerge />
}
