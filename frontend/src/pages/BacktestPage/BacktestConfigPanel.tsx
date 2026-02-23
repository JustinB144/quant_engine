import React from 'react'
import type { BacktestRequest } from '@/types/compute'

interface Props {
  config: BacktestRequest
  onChange: (cfg: BacktestRequest) => void
}

function SliderRow({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void
}) {
  return (
    <div className="mb-3">
      <div className="flex justify-between mb-1">
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{label}</span>
        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
        style={{ accentColor: 'var(--accent-blue)' }}
      />
    </div>
  )
}

export default function BacktestConfigPanel({ config, onChange }: Props) {
  return (
    <div className="card-panel">
      <div className="card-panel-header">Backtest Configuration</div>
      <SliderRow label="Holding Period (days)" value={config.holding_period ?? 10} min={1} max={60} step={1}
        onChange={(v) => onChange({ ...config, holding_period: v })} />
      <SliderRow label="Max Positions" value={config.max_positions ?? 10} min={1} max={50} step={1}
        onChange={(v) => onChange({ ...config, max_positions: v })} />
      <SliderRow label="Entry Threshold" value={config.entry_threshold ?? 0.01} min={0} max={0.1} step={0.005}
        onChange={(v) => onChange({ ...config, entry_threshold: v })} />
      <SliderRow label="Position Size" value={config.position_size ?? 0.1} min={0.01} max={0.5} step={0.01}
        onChange={(v) => onChange({ ...config, position_size: v })} />
    </div>
  )
}
