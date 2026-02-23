import React from 'react'
import { CheckCircle, Info, AlertTriangle, XCircle } from 'lucide-react'

type Severity = 'success' | 'info' | 'warning' | 'error'

const CONFIG: Record<Severity, { color: string; Icon: typeof Info }> = {
  success: { color: 'var(--accent-green)', Icon: CheckCircle },
  info: { color: 'var(--accent-blue)', Icon: Info },
  warning: { color: 'var(--accent-amber)', Icon: AlertTriangle },
  error: { color: 'var(--accent-red)', Icon: XCircle },
}

interface AlertBannerProps {
  severity: Severity
  message: string
  detail?: string
}

export default function AlertBanner({ severity, message, detail }: AlertBannerProps) {
  const { color, Icon } = CONFIG[severity]

  return (
    <div
      className="flex items-start gap-3 rounded-md p-3 mb-3"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderLeft: `3px solid ${color}`,
      }}
    >
      <Icon size={16} style={{ color, marginTop: 1, flexShrink: 0 }} />
      <div>
        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{message}</div>
        {detail && (
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>{detail}</div>
        )}
      </div>
    </div>
  )
}
