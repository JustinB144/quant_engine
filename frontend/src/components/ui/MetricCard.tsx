import React from 'react'

interface MetricCardProps {
  label: string
  value: string
  color?: string
  subtitle?: string
}

export default function MetricCard({ label, value, color = 'var(--accent-blue)', subtitle }: MetricCardProps) {
  return (
    <div
      className="rounded-lg transition-all duration-150"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-light)',
        padding: '14px 16px',
      }}
      onMouseEnter={(e) => {
        ;(e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border)'
      }}
      onMouseLeave={(e) => {
        ;(e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border-light)'
      }}
    >
      <div
        className="font-mono uppercase"
        style={{
          fontSize: 10,
          color: 'var(--text-tertiary)',
          letterSpacing: '0.5px',
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div
        className="font-mono"
        style={{
          fontSize: 22,
          fontWeight: 600,
          color,
          lineHeight: 1.2,
        }}
      >
        {value}
      </div>
      {subtitle && (
        <div
          className="font-mono"
          style={{
            fontSize: 10,
            color: 'var(--text-tertiary)',
            marginTop: 4,
          }}
        >
          {subtitle}
        </div>
      )}
    </div>
  )
}
