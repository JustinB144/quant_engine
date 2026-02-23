import React, { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType, LineStyle } from 'lightweight-charts'

interface EquityCurveChartProps {
  dates: string[]
  equity: number[]
  benchmarkDates?: string[]
  benchmarkEquity?: number[]
  height?: number
}

export default function EquityCurveChart({
  dates,
  equity,
  benchmarkDates,
  benchmarkEquity,
  height = 420,
}: EquityCurveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !dates.length) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#8b949e',
        fontFamily: 'Menlo, monospace',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: '#21262d' },
        horzLines: { color: '#21262d' },
      },
      crosshair: {
        vertLine: { color: '#30363d', style: LineStyle.Dotted, width: 1 },
        horzLine: { color: '#30363d', style: LineStyle.Dotted, width: 1 },
      },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: false },
    })
    chartRef.current = chart

    // Portfolio equity line
    const equitySeries = chart.addLineSeries({
      color: '#58a6ff',
      lineWidth: 2,
      title: 'Portfolio',
    })
    equitySeries.setData(
      dates.map((d, i) => ({ time: d, value: equity[i] })),
    )

    // Benchmark overlay
    if (benchmarkDates?.length && benchmarkEquity?.length) {
      const benchSeries = chart.addLineSeries({
        color: '#8b949e',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'Benchmark',
      })
      benchSeries.setData(
        benchmarkDates.map((d, i) => ({ time: d, value: benchmarkEquity[i] })),
      )
    }

    chart.timeScale().fitContent()

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width })
      }
    })
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [dates, equity, benchmarkDates, benchmarkEquity, height])

  return <div ref={containerRef} style={{ width: '100%', height }} />
}
