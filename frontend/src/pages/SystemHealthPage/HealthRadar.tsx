import React from 'react'
import ChartContainer from '@/components/charts/ChartContainer'
import RadarChart from '@/components/charts/RadarChart'
import type { SystemHealthDetail } from '@/types/health'

export default function HealthRadar({ health }: { health: SystemHealthDetail }) {
  const indicators = [
    { name: 'Data Integrity', max: 100 },
    { name: 'Promotion', max: 100 },
    { name: 'Walk-Forward', max: 100 },
    { name: 'Execution', max: 100 },
    { name: 'Complexity', max: 100 },
  ]
  const values = [
    health.data_integrity_score,
    health.promotion_score,
    health.wf_score,
    health.execution_score,
    health.complexity_score,
  ]

  return (
    <ChartContainer title="Health Radar">
      <RadarChart indicators={indicators} values={values} height={340} />
    </ChartContainer>
  )
}
