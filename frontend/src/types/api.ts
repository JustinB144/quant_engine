/** Mirrors api/schemas/envelope.py exactly */

export interface ResponseMeta {
  data_mode: string
  generated_at: string
  warnings: string[]
  source_summary?: string
  predictor_type?: string
  walk_forward_mode?: string
  regime_suppressed?: boolean
  feature_pipeline_version?: string
  model_version?: string
  sizing_method?: string
  cache_hit: boolean
  elapsed_ms?: number
}

export interface ApiResponse<T> {
  ok: boolean
  data?: T
  error?: string
  meta: ResponseMeta
}
