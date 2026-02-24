import React, { useMemo } from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import CandlestickChart from '@/components/charts/CandlestickChart'
import IndicatorPanel from '@/components/charts/IndicatorPanel'
import { useTickerBars, useTickerIndicators } from '@/api/queries/useData'
import { useFilterStore } from '@/store/filterStore'

interface Props {
  ticker: string
}

const TIMEFRAMES = ['5min', '15min', '30min', '1hr', '1d'] as const
const TIMEFRAME_LABELS: Record<string, string> = {
  '5min': '5m',
  '15min': '15m',
  '30min': '30m',
  '1hr': '1H',
  '1d': '1D',
}

const OVERLAY_INDICATORS = ['sma', 'ema', 'bollinger', 'vwap']
const PANEL_INDICATORS = ['rsi', 'macd', 'atr', 'stochastic', 'obv', 'adx']

export default function CandlestickPanel({ ticker }: Props) {
  const timeframe = useFilterStore((s) => s.selectedTimeframe)
  const setTimeframe = useFilterStore((s) => s.setSelectedTimeframe)
  const enabledIndicators = useFilterStore((s) => s.enabledIndicators)
  const toggleIndicator = useFilterStore((s) => s.toggleIndicator)

  const { data: barsResp, isLoading: barsLoading, error: barsError } = useTickerBars(ticker, timeframe, 500)
  const bars = barsResp?.data?.bars ?? []
  const availableTimeframes = barsResp?.data?.available_timeframes ?? []

  const { data: indicatorResp } = useTickerIndicators(ticker, timeframe, enabledIndicators)
  const indicatorData = indicatorResp?.data?.indicators ?? {}

  // Separate overlay vs panel indicators from response
  const { overlays, bollingerOverlays, panelIndicators } = useMemo(() => {
    const overlays: Array<{ label: string; color: string; data: Array<{ time: string; value: number }> }> = []
    const bollingerOverlays: Array<{
      label: string
      upper: Array<{ time: string; value: number }>
      middle: Array<{ time: string; value: number }>
      lower: Array<{ time: string; value: number }>
    }> = []
    const panelIndicators: Array<{ key: string; data: Record<string, unknown> }> = []

    const overlayColors = ['#58a6ff', '#d29922', '#bc8cff', '#39d2c0', '#f0883e']
    let colorIdx = 0

    for (const [key, data] of Object.entries(indicatorData)) {
      if (!data || (data as Record<string, unknown>).error) continue
      const d = data as Record<string, unknown>
      if (d.type === 'overlay') {
        if (d.upper && d.middle && d.lower) {
          // Bollinger bands
          bollingerOverlays.push({
            label: key,
            upper: d.upper as Array<{ time: string; value: number }>,
            middle: d.middle as Array<{ time: string; value: number }>,
            lower: d.lower as Array<{ time: string; value: number }>,
          })
        } else if (d.values) {
          overlays.push({
            label: key,
            color: overlayColors[colorIdx++ % overlayColors.length],
            data: d.values as Array<{ time: string; value: number }>,
          })
        }
      } else {
        panelIndicators.push({ key, data: d })
      }
    }

    return { overlays, bollingerOverlays, panelIndicators }
  }, [indicatorData])

  const allIndicators = [...OVERLAY_INDICATORS, ...PANEL_INDICATORS]

  return (
    <div>
      {/* Timeframe selector + indicator toggles */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        {/* Timeframe buttons */}
        <div className="flex rounded-md overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          {TIMEFRAMES.map((tf) => {
            const active = tf === timeframe
            const available = availableTimeframes.length === 0 || availableTimeframes.includes(tf)
            return (
              <button
                key={tf}
                onClick={() => available && setTimeframe(tf)}
                disabled={!available}
                className="font-mono px-3 py-1"
                style={{
                  fontSize: 11,
                  fontWeight: active ? 600 : 400,
                  backgroundColor: active ? 'var(--accent-blue)' : 'var(--bg-secondary)',
                  color: active ? '#ffffff' : available ? 'var(--text-secondary)' : 'var(--text-tertiary)',
                  border: 'none',
                  cursor: available ? 'pointer' : 'not-allowed',
                  opacity: available ? 1 : 0.5,
                  borderRight: '1px solid var(--border-light)',
                }}
                title={!available ? `No ${tf} data cached` : undefined}
              >
                {TIMEFRAME_LABELS[tf] || tf}
              </button>
            )
          })}
        </div>

        <div style={{ width: 1, height: 20, backgroundColor: 'var(--border-light)' }} />

        {/* Indicator toggles */}
        <div className="flex gap-1 flex-wrap">
          {allIndicators.map((ind) => {
            const specKey = ind === 'rsi' ? 'rsi_14' : ind === 'bollinger' ? 'bollinger_20' : ind === 'sma' ? 'sma_20' : ind === 'ema' ? 'ema_20' : ind === 'atr' ? 'atr_14' : ind === 'adx' ? 'adx_14' : ind
            const active = enabledIndicators.includes(specKey)
            return (
              <button
                key={ind}
                onClick={() => toggleIndicator(specKey)}
                className="font-mono px-2 py-0.5 rounded"
                style={{
                  fontSize: 10,
                  backgroundColor: active ? 'rgba(88, 166, 255, 0.15)' : 'var(--bg-secondary)',
                  color: active ? 'var(--accent-blue)' : 'var(--text-tertiary)',
                  border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border-light)'}`,
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                }}
              >
                {ind}
              </button>
            )
          })}
        </div>
      </div>

      {/* Main chart */}
      <ChartContainer
        title={ticker ? `${ticker} — ${timeframe.toUpperCase()}` : 'Select a ticker'}
        meta={barsResp?.meta}
        isLoading={barsLoading}
        error={barsError?.message}
        isEmpty={!ticker || bars.length === 0}
        emptyMessage={ticker ? `No ${timeframe} data available for ${ticker}` : 'Select a ticker from the universe'}
      >
        <CandlestickChart
          bars={bars}
          overlays={overlays}
          bollingerOverlays={bollingerOverlays}
          height={500}
        />

        {/* Sub-pane indicator panels */}
        {panelIndicators.map((pi) => (
          <IndicatorPanel key={pi.key} label={pi.key} data={pi.data} height={150} />
        ))}
      </ChartContainer>

      {/* Bar count info */}
      {bars.length > 0 && (
        <div className="font-mono mt-1 px-1" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          {bars.length} bars loaded
          {availableTimeframes.length > 0 && ` | Cached timeframes: ${availableTimeframes.join(', ')}`}
        </div>
      )}
    </div>
  )
}
