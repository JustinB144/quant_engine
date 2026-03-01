import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { bloombergDarkTheme, CHART_COLORS } from './theme'

interface ScatterPoint {
  x: number
  y: number
  label?: string
  color?: string
  size?: number
}

interface ScatterChartProps {
  data: ScatterPoint[]
  height?: number
  xAxisName?: string
  yAxisName?: string
}

export default function ScatterChart({
  data,
  height = 360,
  xAxisName,
  yAxisName,
}: ScatterChartProps) {
  const option = useMemo(
    () => ({
      ...bloombergDarkTheme,
      tooltip: {
        ...bloombergDarkTheme.tooltip,
        formatter: (params: { data: number[] }) => {
          return `${xAxisName || 'X'}: ${params.data[0].toFixed(4)}<br/>${yAxisName || 'Y'}: ${params.data[1].toFixed(4)}`
        },
      },
      grid: { left: 60, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'value' as const,
        name: xAxisName,
        ...bloombergDarkTheme.xAxis,
      },
      yAxis: {
        type: 'value' as const,
        name: yAxisName,
        ...bloombergDarkTheme.yAxis,
      },
      series: [
        {
          type: 'scatter' as const,
          data: data.map((p) => [p.x, p.y]),
          symbolSize: (_value: number[], params: { dataIndex: number }) => {
            const point = data[params.dataIndex]
            return point?.size ?? 8
          },
          itemStyle: {
            color: (params: { dataIndex: number }) =>
              data[params.dataIndex]?.color || CHART_COLORS[0],
          },
        },
      ],
    }),
    [data, height, xAxisName, yAxisName],
  )

  return <ReactECharts option={option} style={{ height }} notMerge />
}
