/** Matches actual /api/health response.data */
export interface QuickStatus {
  status: string
  checks: Record<string, string>
  timestamp: string
}

/** Matches actual /api/health/detailed response.data */
export interface SystemHealthDetail {
  overall_score: number
  overall_status: string
  generated_at: string
  data_integrity_score: number
  promotion_score: number
  wf_score: number
  execution_score: number
  complexity_score: number
  survivorship_checks: HealthCheckItem[]
  data_quality_checks: HealthCheckItem[]
  promotion_checks: HealthCheckItem[]
  wf_checks: HealthCheckItem[]
  execution_checks: HealthCheckItem[]
  complexity_checks: HealthCheckItem[]
  strengths: HealthCheckItem[]
  promotion_funnel: Record<string, number>
  feature_inventory: Record<string, number>
  knob_inventory: Array<{ name: string; value: string | number; module: string }>
}

export interface HealthCheckItem {
  name: string
  status: string
  detail: string
  value?: string | number | null
  recommendation?: string | null
}
