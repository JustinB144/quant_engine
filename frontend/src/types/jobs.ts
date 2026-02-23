export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface JobRecord {
  job_id: string
  job_type: string
  status: JobStatus
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  message?: string
  result?: unknown
  error?: string
}

export interface JobEvent {
  event: string
  data: {
    job_id: string
    progress: number
    message?: string
    status?: JobStatus
    result?: unknown
    error?: string
  }
}
