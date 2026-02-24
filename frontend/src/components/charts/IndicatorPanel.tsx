import React, { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType, LineStyle } from 'lightweight-charts'

interface IndicatorPanelProps {
  /** Indicator key, e.g. "rsi_14" */
  label: string
  /** Indicator data from API */
  data: Record<string, unknown>
  height?: number
  /** Callback when crosshair moves for sync */
  onCrosshairMove?: (time: string | null) => void
  /** External crosshair time to sync to */
  syncTime?: string | null
}

const PANEL_COLORS = {
  primary: '#58a6ff',
  secondary: '#d29922',
  histogram_up: 'rgba(63, 185, 80, 0.6)',
  histogram_down: 'rgba(248, 81, 73, 0.6)',
  threshold: '#30363d',
}

export default function IndicatorPanel({
  label,
  data,
  height = 150,
}: IndicatorPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !data) return

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
      timeScale: { borderColor: '#30363d', timeVisible: false, visible: false },
    })
    chartRef.current = chart

    const indicatorType = label.split('_')[0].toLowerCase()

    if (indicatorType === 'macd' && data.macd) {
      // MACD: line + signal + histogram
      const macdData = data.macd as Array<{ time: string; value: number }>
      const signalData = data.signal as Array<{ time: string; value: number }>
      const histData = data.histogram as Array<{ time: string; value: number }>

      const macdSeries = chart.addLineSeries({
        color: PANEL_COLORS.primary,
        lineWidth: 1,
        title: 'MACD',
      })
      macdSeries.setData(macdData.map((d) => ({ time: d.time, value: d.value })))

      if (signalData) {
        const signalSeries = chart.addLineSeries({
          color: PANEL_COLORS.secondary,
          lineWidth: 1,
          title: 'Signal',
        })
        signalSeries.setData(signalData.map((d) => ({ time: d.time, value: d.value })))
      }

      if (histData) {
        const histSeries = chart.addHistogramSeries({
          title: 'Hist',
        })
        histSeries.setData(
          histData.map((d) => ({
            time: d.time,
            value: d.value,
            color: d.value >= 0 ? PANEL_COLORS.histogram_up : PANEL_COLORS.histogram_down,
          })),
        )
      }
    } else if (indicatorType === 'stochastic' && data.k) {
      // Stochastic: %K + %D
      const kData = data.k as Array<{ time: string; value: number }>
      const dData = data.d as Array<{ time: string; value: number }>

      const kSeries = chart.addLineSeries({
        color: PANEL_COLORS.primary,
        lineWidth: 1,
        title: '%K',
      })
      kSeries.setData(kData.map((d) => ({ time: d.time, value: d.value })))

      if (dData) {
        const dSeries = chart.addLineSeries({
          color: PANEL_COLORS.secondary,
          lineWidth: 1,
          title: '%D',
        })
        dSeries.setData(dData.map((d) => ({ time: d.time, value: d.value })))
      }
    } else if (data.values) {
      // Simple single-line indicator (RSI, ATR, ADX, OBV)
      const values = data.values as Array<{ time: string; value: number }>
      const series = chart.addLineSeries({
        color: PANEL_COLORS.primary,
        lineWidth: 1,
        title: label.toUpperCase(),
      })
      series.setData(values.map((d) => ({ time: d.time, value: d.value })))
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
  }, [data, label, height])

  // Threshold labels
  const thresholds = data.thresholds as Record<string, number> | undefined

  return (
    <div style={{ borderTop: '1px solid var(--border-light)' }}>
      <div className="flex items-center justify-between px-2 py-1">
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
          {label}
        </span>
        {thresholds && (
          <div className="flex gap-3">
            {Object.entries(thresholds).map(([key, val]) => (
              <span key={key} className="font-mono" style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
                {key}: {val}
              </span>
            ))}
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ width: '100%', height }} />
    </div>
  )
}
