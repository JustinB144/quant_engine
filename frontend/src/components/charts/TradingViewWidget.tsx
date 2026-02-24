import React, { useEffect, useRef, memo } from 'react'

interface TradingViewWidgetProps {
  symbol: string
  theme?: 'dark' | 'light'
  interval?: string
  height?: number
}

/**
 * Free TradingView Advanced Chart embed widget.
 * Uses TradingView's own data feed — shown as a "Market Reference" panel
 * alongside the custom lightweight-charts that uses local cached data.
 *
 * Note: This widget loads external scripts from TradingView. It requires
 * network access and uses TV's data, not local cached data.
 */
function TradingViewWidgetInner({
  symbol,
  theme = 'dark',
  interval = 'D',
  height = 500,
}: TradingViewWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    // Clear previous widget
    containerRef.current.innerHTML = ''

    const widgetContainer = document.createElement('div')
    widgetContainer.className = 'tradingview-widget-container'
    widgetContainer.style.height = `${height}px`

    const widgetInner = document.createElement('div')
    widgetInner.className = 'tradingview-widget-container__widget'
    widgetInner.style.height = '100%'
    widgetInner.style.width = '100%'
    widgetContainer.appendChild(widgetInner)

    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: symbol,
      interval: interval,
      timezone: 'America/New_York',
      theme: theme,
      style: '1',
      locale: 'en',
      hide_top_toolbar: false,
      hide_legend: false,
      allow_symbol_change: true,
      save_image: false,
      calendar: false,
      hide_volume: false,
      support_host: 'https://www.tradingview.com',
    })
    widgetContainer.appendChild(script)

    containerRef.current.appendChild(widgetContainer)

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [symbol, theme, interval, height])

  return (
    <div
      ref={containerRef}
      style={{
        height,
        width: '100%',
        borderRadius: 6,
        overflow: 'hidden',
        border: '1px solid var(--border-light)',
      }}
    />
  )
}

export default memo(TradingViewWidgetInner)
