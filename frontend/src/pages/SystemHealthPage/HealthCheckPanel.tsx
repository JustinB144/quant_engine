import React, { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { HealthCheckItem } from '@/types/health'

const STATUS_COLORS: Record<string, string> = {
  PASS: 'var(--accent-green)',
  pass: 'var(--accent-green)',
  WARN: 'var(--accent-amber)',
  warn: 'var(--accent-amber)',
  WARNING: 'var(--accent-amber)',
  FAIL: 'var(--accent-red)',
  fail: 'var(--accent-red)',
  CRITICAL: 'var(--accent-red)',
  UNAVAILABLE: 'var(--text-tertiary)',
  unavailable: 'var(--text-tertiary)',
}

const SEVERITY_LABELS: Record<string, { label: string; color: string }> = {
  critical: { label: 'CRIT', color: 'var(--accent-red)' },
  standard: { label: 'STD', color: 'var(--text-tertiary)' },
  informational: { label: 'INFO', color: 'var(--accent-blue)' },
}

interface Props {
  title: string
  checks: HealthCheckItem[]
}

function StatusBadge({ status }: { status: string }) {
  const upper = status.toUpperCase()
  const color = STATUS_COLORS[status] ?? 'var(--text-tertiary)'
  const isUnavailable = upper === 'UNAVAILABLE'

  return (
    <span
      className="font-mono rounded px-1.5 py-0.5"
      style={{
        fontSize: 9,
        fontWeight: 600,
        color: isUnavailable ? 'var(--text-tertiary)' : color,
        backgroundColor: isUnavailable ? 'rgba(139, 148, 158, 0.1)' : `${color}15`,
        border: isUnavailable ? '1px dashed var(--border)' : 'none',
      }}
    >
      {upper}
    </span>
  )
}

function ThresholdBar({
  rawValue,
  thresholds,
}: {
  rawValue: number
  thresholds: Record<string, number>
}) {
  // Simple visual bar showing where the raw value falls
  const entries = Object.entries(thresholds).sort(([, a], [, b]) => a - b)
  const max = Math.max(rawValue, ...entries.map(([, v]) => v)) * 1.2

  return (
    <div className="mt-1" style={{ height: 4, width: '100%', backgroundColor: 'var(--bg-tertiary)', borderRadius: 2, position: 'relative' }}>
      {/* Threshold markers */}
      {entries.map(([key, val]) => (
        <div
          key={key}
          title={`${key}: ${val}`}
          style={{
            position: 'absolute',
            left: `${Math.min((val / max) * 100, 100)}%`,
            top: -2,
            width: 1,
            height: 8,
            backgroundColor: 'var(--border)',
          }}
        />
      ))}
      {/* Value marker */}
      <div
        style={{
          position: 'absolute',
          left: `${Math.min((rawValue / max) * 100, 100)}%`,
          top: -3,
          width: 6,
          height: 6,
          borderRadius: '50%',
          backgroundColor: 'var(--accent-blue)',
          transform: 'translateX(-3px)',
        }}
      />
    </div>
  )
}

export default function HealthCheckPanel({ title, checks }: Props) {
  const [open, setOpen] = useState(false)
  const [expandedCheck, setExpandedCheck] = useState<number | null>(null)

  if (!checks || checks.length === 0) return null

  const available = checks.filter((c) => c.status.toUpperCase() !== 'UNAVAILABLE')
  const passCount = available.filter((c) => c.status.toUpperCase() === 'PASS').length
  const unavailableCount = checks.length - available.length
  const totalCount = checks.length
  const allPass = passCount === available.length && available.length > 0

  return (
    <div className="card-panel mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between"
        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown size={14} style={{ color: 'var(--text-tertiary)' }} /> : <ChevronRight size={14} style={{ color: 'var(--text-tertiary)' }} />}
          <span style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>{title}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono" style={{ fontSize: 11, color: allPass ? 'var(--accent-green)' : 'var(--accent-amber)', fontWeight: 600 }}>
            {passCount}/{available.length} PASS
          </span>
          {unavailableCount > 0 && (
            <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
              ({unavailableCount} unavail.)
            </span>
          )}
        </div>
      </button>

      {open && (
        <div className="mt-3 ml-5">
          {checks.map((item, i) => {
            const isUnavailable = item.status.toUpperCase() === 'UNAVAILABLE'
            const isExpanded = expandedCheck === i

            return (
              <div key={i} className="mb-2">
                <div
                  className="flex items-start gap-2 cursor-pointer"
                  onClick={() => setExpandedCheck(isExpanded ? null : i)}
                >
                  <span
                    className="inline-block rounded-full mt-1 flex-shrink-0"
                    style={{
                      width: 6,
                      height: 6,
                      backgroundColor: STATUS_COLORS[item.status] ?? 'var(--text-tertiary)',
                    }}
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono" style={{ fontSize: 11, color: isUnavailable ? 'var(--text-tertiary)' : 'var(--text-secondary)' }}>
                        {item.name}
                      </span>
                      <StatusBadge status={item.status} />
                      {item.severity && SEVERITY_LABELS[item.severity] && (
                        <span
                          className="font-mono"
                          style={{ fontSize: 8, color: SEVERITY_LABELS[item.severity].color }}
                        >
                          {SEVERITY_LABELS[item.severity].label}
                        </span>
                      )}
                      {item.raw_value != null && (
                        <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                          val={typeof item.raw_value === 'number' ? item.raw_value.toFixed(3) : item.raw_value}
                        </span>
                      )}
                    </div>
                    {item.detail && (
                      <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 0 }}>
                        {item.detail}
                      </span>
                    )}
                    {item.raw_value != null && item.thresholds && Object.keys(item.thresholds).length > 0 && (
                      <ThresholdBar rawValue={item.raw_value} thresholds={item.thresholds} />
                    )}
                  </div>
                </div>

                {/* Expandable methodology section */}
                {isExpanded && (
                  <div className="ml-5 mt-1 mb-1 rounded" style={{ padding: '8px 12px', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-light)' }}>
                    {item.methodology && (
                      <div className="mb-1.5">
                        <span className="font-mono" style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Methodology</span>
                        <p className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: 2 }}>
                          {item.methodology}
                        </p>
                      </div>
                    )}
                    {item.thresholds && Object.keys(item.thresholds).length > 0 && (
                      <div className="mb-1.5">
                        <span className="font-mono" style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Thresholds</span>
                        <div className="flex gap-3 mt-1">
                          {Object.entries(item.thresholds).map(([key, val]) => (
                            <span key={key} className="font-mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                              {key}: {typeof val === 'number' ? val.toFixed(2) : val}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {item.recommendation && (
                      <div>
                        <span className="font-mono" style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Recommendation</span>
                        <p className="font-mono" style={{ fontSize: 10, color: 'var(--accent-amber)', lineHeight: 1.5, marginTop: 2 }}>
                          {item.recommendation}
                        </p>
                      </div>
                    )}
                    {!item.methodology && !item.recommendation && (
                      <p className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                        No additional details available for this check.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
