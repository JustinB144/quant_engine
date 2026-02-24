import React, { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType, LineStyle } from 'lightweight-charts'

interface OHLCVBar {
  time: string
  date?: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface OverlaySeries {
  label: string
  color: string
  data: Array<{ time: string; value: number }>
}

interface BollingerOverlay {
  label: string
  upper: Array<{ time: string; value: number }>
  middle: Array<{ time: string; value: number }>
  lower: Array<{ time: string; value: number }>
  color?: string
}

interface CandlestickChartProps {
  bars: OHLCVBar[]
  smaOverlays?: { period: number; color: string }[]
  /** Overlay line series from API indicators (SMA, EMA, VWAP) */
  overlays?: OverlaySeries[]
  /** Bollinger bands overlay */
  bollingerOverlays?: BollingerOverlay[]
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
  overlays = [],
  bollingerOverlays = [],
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

    // Normalize bar time field (API may use "time" or "date")
    const normalizedBars = bars.map((b) => ({
      ...b,
      time: b.time || b.date || '',
    }))

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
      normalizedBars.map((b) => ({
        time: b.time,
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
      normalizedBars.map((b) => ({
        time: b.time,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
      })),
    )

    // Built-in SMA overlays (fallback when no API indicators loaded)
    if (overlays.length === 0 && bollingerOverlays.length === 0) {
      const closes = normalizedBars.map((b) => b.close)
      for (const { period, color } of smaOverlays) {
        const sma = computeSMA(closes, period)
        const smaSeries = chart.addLineSeries({
          color,
          lineWidth: 1,
          title: `SMA ${period}`,
        })
        smaSeries.setData(
          normalizedBars
            .map((b, i) => (sma[i] != null ? { time: b.time, value: sma[i]! } : null))
            .filter(Boolean) as { time: string; value: number }[],
        )
      }
    }

    // API overlay indicators (SMA, EMA, VWAP from backend)
    const overlayColors = ['#58a6ff', '#d29922', '#bc8cff', '#39d2c0', '#f0883e']
    for (let idx = 0; idx < overlays.length; idx++) {
      const overlay = overlays[idx]
      const lineSeries = chart.addLineSeries({
        color: overlay.color || overlayColors[idx % overlayColors.length],
        lineWidth: 1,
        title: overlay.label,
      })
      lineSeries.setData(overlay.data.map((d) => ({ time: d.time, value: d.value })))
    }

    // Bollinger bands overlay
    for (const bb of bollingerOverlays) {
      const bbColor = bb.color || '#bc8cff'
      const upperSeries = chart.addLineSeries({
        color: bbColor,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: `${bb.label} Upper`,
      })
      upperSeries.setData(bb.upper.map((d) => ({ time: d.time, value: d.value })))

      const middleSeries = chart.addLineSeries({
        color: bbColor,
        lineWidth: 1,
        title: `${bb.label} Mid`,
      })
      middleSeries.setData(bb.middle.map((d) => ({ time: d.time, value: d.value })))

      const lowerSeries = chart.addLineSeries({
        color: bbColor,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: `${bb.label} Lower`,
      })
      lowerSeries.setData(bb.lower.map((d) => ({ time: d.time, value: d.value })))
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
  }, [bars, smaOverlays, overlays, bollingerOverlays, height])

  return <div ref={containerRef} style={{ width: '100%', height }} />
}
