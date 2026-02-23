import React, { useEffect, useState } from 'react'
import { useClock } from '@/hooks/useClock'

interface ModelAge {
  age_days: number | null
  version_id: string | null
  status: string
}

interface DataMode {
  mode: string
  n_fallbacks: number
  status: string
}

interface HealthStatus {
  status: string
}

export default function StatusBar() {
  const time = useClock()
  const [modelAge, setModelAge] = useState<ModelAge | null>(null)
  const [dataMode, setDataMode] = useState<DataMode | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [modelRes, dataRes, healthRes] = await Promise.allSettled([
          fetch('/api/v1/system/model-age').then(r => r.json()),
          fetch('/api/v1/system/data-mode').then(r => r.json()),
          fetch('/api/health').then(r => r.json()),
        ])
        if (modelRes.status === 'fulfilled') setModelAge(modelRes.value?.data)
        if (dataRes.status === 'fulfilled') setDataMode(dataRes.value?.data)
        if (healthRes.status === 'fulfilled') setHealth(healthRes.value?.data)
      } catch {
        // Silently fail — status bar is informational only
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 60_000) // Refresh every 60s
    return () => clearInterval(interval)
  }, [])

  const healthColor =
    health?.status === 'healthy'
      ? 'var(--accent-green)'
      : health?.status === 'degraded'
        ? 'var(--accent-yellow, #f0ad4e)'
        : health?.status === 'unhealthy'
          ? 'var(--accent-red, #d9534f)'
          : 'var(--text-tertiary)'

  const modelAgeLabel = modelAge?.age_days != null
    ? `Model: ${modelAge.age_days}d old`
    : 'Model: --'

  const modelAgeColor = modelAge?.status === 'stale'
    ? 'var(--accent-yellow, #f0ad4e)'
    : 'var(--text-tertiary)'

  const dataModeLabel = dataMode
    ? `Data: ${dataMode.mode.toUpperCase()}${dataMode.n_fallbacks > 0 ? ` (${dataMode.n_fallbacks} fallbacks)` : ''}`
    : 'Data: --'

  const dataModeColor = dataMode?.status === 'degraded'
    ? 'var(--accent-yellow, #f0ad4e)'
    : 'var(--text-tertiary)'

  return (
    <div
      className="flex items-center justify-between px-4 font-mono shrink-0"
      style={{
        height: 'var(--statusbar-height)',
        backgroundColor: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border-light)',
        fontSize: 10,
        color: 'var(--text-tertiary)',
      }}
    >
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block rounded-full"
            style={{
              width: 6,
              height: 6,
              backgroundColor: healthColor,
              animation: health?.status === 'healthy' ? undefined : 'pulse 2s infinite',
            }}
          />
          <span style={{ color: healthColor }}>
            {health?.status?.toUpperCase() ?? 'CONNECTED'}
          </span>
        </div>
        <span>API: localhost:8000</span>
        <span style={{ color: modelAgeColor }}>{modelAgeLabel}</span>
        <span style={{ color: dataModeColor }}>{dataModeLabel}</span>
      </div>
      <div className="flex items-center gap-3">
        <span>Quant Engine v2.0.0</span>
        <span style={{ color: 'var(--text-secondary)' }}>{time}</span>
      </div>
    </div>
  )
}
