import React, { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType, LineStyle } from 'lightweight-charts'

interface OHLCVBar {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface CandlestickChartProps {
  bars: OHLCVBar[]
  smaOverlays?: { period: number; color: string }[]
  height?: number
}

function computeSMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = []
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      result.push(null)
    } else {
      let sum = 0
      for (let j = i - period + 1; j <= i; j++) sum += closes[j]
      result.push(sum / period)
    }
  }
  return result
}

export default function CandlestickChart({
  bars,
  smaOverlays = [
    { period: 20, color: '#58a6ff' },
    { period: 50, color: '#d29922' },
  ],
  height = 500,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !bars.length) return

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

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#3fb950',
      downColor: '#f85149',
      borderUpColor: '#3fb950',
      borderDownColor: '#f85149',
      wickUpColor: '#3fb950',
      wickDownColor: '#f85149',
    })
    candleSeries.setData(
      bars.map((b) => ({
        time: b.date,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    )

    // Volume pane
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })
    volumeSeries.setData(
      bars.map((b) => ({
        time: b.date,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
      })),
    )

    // SMA overlays
    const closes = bars.map((b) => b.close)
    for (const { period, color } of smaOverlays) {
      const sma = computeSMA(closes, period)
      const smaSeries = chart.addLineSeries({
        color,
        lineWidth: 1,
        title: `SMA ${period}`,
      })
      smaSeries.setData(
        bars
          .map((b, i) => (sma[i] != null ? { time: b.date, value: sma[i]! } : null))
          .filter(Boolean) as { time: string; value: number }[],
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
  }, [bars, smaOverlays, height])

  return <div ref={containerRef} style={{ width: '100%', height }} />
}
