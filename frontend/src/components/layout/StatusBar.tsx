import React from 'react'
import { useClock } from '@/hooks/useClock'

export default function StatusBar() {
  const time = useClock()

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
            className="animate-pulse-dot inline-block rounded-full"
            style={{ width: 6, height: 6, backgroundColor: 'var(--accent-green)' }}
          />
          <span style={{ color: 'var(--accent-green)' }}>CONNECTED</span>
        </div>
        <span>API: localhost:8000</span>
      </div>
      <div className="flex items-center gap-3">
        <span>Quant Engine v2.0.0</span>
        <span style={{ color: 'var(--text-secondary)' }}>{time}</span>
      </div>
    </div>
  )
}
