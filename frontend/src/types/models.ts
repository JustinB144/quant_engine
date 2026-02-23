/** Matches actual /api/models/versions response.data (array) */
export interface ModelVersionInfo {
  version_id: string
  training_date: string
  horizon: number
  universe_size: number
  n_samples: number
  n_features: number
  oos_spearman: number
  cv_gap: number
  holdout_r2: number
  holdout_spearman: number
  survivorship_mode: boolean
  universe_as_of: string | null
  delisted_included: number
  notes: string
  tags: string[]
}

/** Matches actual /api/models/health response.data */
export interface ModelHealth {
  cv_gap: number
  holdout_r2: number
  holdout_ic: number
  ic_drift: number
  retrain_triggered: boolean
  retrain_reasons: string[]
  registry_history: Array<{
    version_id: string
    cv_gap: number
    holdout_r2: number
    holdout_spearman: number
  }>
}

/** Matches actual /api/models/features/importance response.data */
export interface FeatureImportance {
  global_importance: Record<string, number>
  regime_heatmap: Record<string, Record<string, number>>
}
