import React from 'react'

const REGIME_COLORS: Record<number, string> = {
  0: 'var(--regime-0)',
  1: 'var(--regime-1)',
  2: 'var(--regime-2)',
  3: 'var(--regime-3)',
}

const REGIME_NAMES: Record<number, string> = {
  0: 'Trending Bull',
  1: 'Trending Bear',
  2: 'Mean Reverting',
  3: 'High Volatility',
}

export default function RegimeBadge({ regime }: { regime: number | string }) {
  const idx = typeof regime === 'string' ? parseInt(regime, 10) : regime
  const color = REGIME_COLORS[idx] ?? 'var(--text-tertiary)'
  const label = REGIME_NAMES[idx] ?? (typeof regime === 'string' ? regime : `Regime ${regime}`)

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block rounded-full"
        style={{ width: 6, height: 6, backgroundColor: color }}
      />
      <span className="font-mono" style={{ fontSize: 11, color }}>
        {label}
      </span>
    </span>
  )
}
