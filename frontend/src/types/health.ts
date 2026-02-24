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
  overall_methodology?: string
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
  history?: HealthHistoryEntry[]
  data_integrity_checks_available?: number
  data_integrity_checks_total?: number
  promotion_checks_available?: number
  promotion_checks_total?: number
  wf_checks_available?: number
  wf_checks_total?: number
  execution_checks_available?: number
  execution_checks_total?: number
  complexity_checks_available?: number
  complexity_checks_total?: number
}

export interface HealthCheckItem {
  name: string
  status: string
  detail: string
  value?: string | number | null
  recommendation?: string | null
  methodology?: string
  thresholds?: Record<string, number>
  severity?: string
  raw_value?: number | null
  data_available?: boolean
}

export interface HealthHistoryEntry {
  timestamp: string
  overall_score: number
  domain_scores?: Record<string, number>
}
