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
}

interface Props {
  title: string
  checks: HealthCheckItem[]
}

export default function HealthCheckPanel({ title, checks }: Props) {
  const [open, setOpen] = useState(false)

  if (!checks || checks.length === 0) return null

  const passCount = checks.filter((c) => c.status.toUpperCase() === 'PASS').length
  const totalCount = checks.length
  const allPass = passCount === totalCount

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
            {passCount}/{totalCount} PASS
          </span>
        </div>
      </button>

      {open && (
        <div className="mt-3 ml-5">
          {checks.map((item, i) => (
            <div key={i} className="flex items-start gap-2 mb-1.5">
              <span
                className="inline-block rounded-full mt-1 flex-shrink-0"
                style={{
                  width: 6,
                  height: 6,
                  backgroundColor: STATUS_COLORS[item.status] ?? 'var(--text-tertiary)',
                }}
              />
              <div>
                <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  {item.name}
                </span>
                {item.detail && (
                  <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 8 }}>
                    — {item.detail}
                  </span>
                )}
                {item.recommendation && (
                  <div className="font-mono" style={{ fontSize: 10, color: 'var(--accent-amber)', marginTop: 2 }}>
                    Recommendation: {item.recommendation}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
