import React from 'react'
import MetricCard from '@/components/ui/MetricCard'
import type { SystemHealthDetail } from '@/types/health'

function scoreColor(score: number | null | undefined): string {
  if (score == null) return 'var(--text-tertiary)'
  if (score > 80) return 'var(--accent-green)'
  if (score > 50) return 'var(--accent-amber)'
  return 'var(--accent-red)'
}

function formatScore(score: number | null | undefined): string {
  if (score == null) return 'N/A'
  return score.toFixed(0)
}

export default function HealthScoreCards({ health }: { health: SystemHealthDetail }) {
  return (
    <div className="grid grid-cols-6 gap-3 mb-4">
      <MetricCard
        label="Overall"
        value={`${formatScore(health.overall_score)}/100`}
        color={scoreColor(health.overall_score)}
        subtitle={health.overall_status}
      />
      <MetricCard
        label="Data Integrity (25%)"
        value={formatScore(health.data_integrity_score)}
        color={scoreColor(health.data_integrity_score)}
        subtitle={health.data_integrity_checks_available != null
          ? `${health.data_integrity_checks_available}/${health.data_integrity_checks_total} checks`
          : undefined}
      />
      <MetricCard
        label="Promotion Gate (25%)"
        value={formatScore(health.promotion_score)}
        color={scoreColor(health.promotion_score)}
        subtitle={health.promotion_checks_available != null
          ? `${health.promotion_checks_available}/${health.promotion_checks_total} checks`
          : undefined}
      />
      <MetricCard
        label="Walk-Forward (20%)"
        value={formatScore(health.wf_score)}
        color={scoreColor(health.wf_score)}
        subtitle={health.wf_checks_available != null
          ? `${health.wf_checks_available}/${health.wf_checks_total} checks`
          : undefined}
      />
      <MetricCard
        label="Execution (15%)"
        value={formatScore(health.execution_score)}
        color={scoreColor(health.execution_score)}
        subtitle={health.execution_checks_available != null
          ? `${health.execution_checks_available}/${health.execution_checks_total} checks`
          : undefined}
      />
      <MetricCard
        label="Complexity (15%)"
        value={formatScore(health.complexity_score)}
        color={scoreColor(health.complexity_score)}
        subtitle={health.complexity_checks_available != null
          ? `${health.complexity_checks_available}/${health.complexity_checks_total} checks`
          : undefined}
      />
    </div>
  )
}
