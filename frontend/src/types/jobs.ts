export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface JobRecord {
  job_id: string
  job_type: string
  status: JobStatus
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  progress_message?: string
  result?: unknown
  error?: string
}

export interface JobEvent {
  event: string
  data: {
    job_id: string
    progress: number
    progress_message?: string
    status?: JobStatus
    result?: unknown
    error?: string
  }
}
