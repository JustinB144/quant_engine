import React from 'react'

interface SliderDef {
  id: string
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (val: number) => void
}

interface IVControlsProps {
  title: string
  presets: Record<string, Record<string, number>>
  activePreset: string
  onPresetChange: (preset: string) => void
  sliders: SliderDef[]
  actionButton?: React.ReactNode
}

export default function IVControls({
  title,
  presets,
  activePreset,
  onPresetChange,
  sliders,
  actionButton,
}: IVControlsProps) {
  return (
    <div className="card-panel">
      <div className="card-panel-header">{title}</div>

      <label className="font-mono block mb-1" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
        Preset
      </label>
      <select
        value={activePreset}
        onChange={(e) => onPresetChange(e.target.value)}
        className="w-full rounded px-2 py-1.5 mb-4 font-mono"
        style={{
          fontSize: 11,
          backgroundColor: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
        }}
      >
        {Object.keys(presets).map((k) => (
          <option key={k} value={k}>{k}</option>
        ))}
      </select>

      {sliders.map((s) => (
        <div key={s.id} className="mb-4">
          <div className="flex justify-between mb-1">
            <label className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
              {s.label}
            </label>
            <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
              {s.value.toFixed(s.step < 0.01 ? 3 : 2)}
            </span>
          </div>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={s.value}
            onChange={(e) => s.onChange(Number(e.target.value))}
            className="w-full"
            style={{ accentColor: 'var(--accent-blue)' }}
          />
          <div className="flex justify-between font-mono" style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
            <span>{s.min}</span>
            <span>{s.max}</span>
          </div>
        </div>
      ))}

      {actionButton}
    </div>
  )
}
