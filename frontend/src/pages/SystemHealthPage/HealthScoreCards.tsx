import React from 'react'
import MetricCard from '@/components/ui/MetricCard'
import type { SystemHealthDetail } from '@/types/health'

function scoreColor(score: number): string {
  if (score > 80) return 'var(--accent-green)'
  if (score > 50) return 'var(--accent-amber)'
  return 'var(--accent-red)'
}

export default function HealthScoreCards({ health }: { health: SystemHealthDetail }) {
  return (
    <div className="grid grid-cols-6 gap-3 mb-4">
      <MetricCard label="Overall" value={`${health.overall_score.toFixed(0)}/100`} color={scoreColor(health.overall_score)} subtitle={health.overall_status} />
      <MetricCard label="Data Integrity" value={`${health.data_integrity_score.toFixed(0)}`} color={scoreColor(health.data_integrity_score)} />
      <MetricCard label="Promotion Gate" value={`${health.promotion_score.toFixed(0)}`} color={scoreColor(health.promotion_score)} />
      <MetricCard label="Walk-Forward" value={`${health.wf_score.toFixed(0)}`} color={scoreColor(health.wf_score)} />
      <MetricCard label="Execution" value={`${health.execution_score.toFixed(0)}`} color={scoreColor(health.execution_score)} />
      <MetricCard label="Complexity" value={`${health.complexity_score.toFixed(0)}`} color={scoreColor(health.complexity_score)} />
    </div>
  )
}
