import { useState, useCallback } from 'react'
import { useJob } from '@/api/queries/useJobs'
import { useSSE } from './useSSE'
import { JOBS_EVENTS } from '@/api/endpoints'
import type { JobRecord, JobStatus } from '@/types/jobs'

interface JobProgress {
  progress: number
  progress_message: string
  status: JobStatus
  result?: unknown
  error?: string
}

export function useJobProgress(jobId: string | null) {
  const [sseProgress, setSSEProgress] = useState<JobProgress | null>(null)
  const jobQuery = useJob(jobId)

  const isActive = jobId != null && (
    sseProgress?.status === 'running' ||
    sseProgress?.status === 'queued' ||
    jobQuery.data?.data?.status === 'running' ||
    jobQuery.data?.data?.status === 'queued'
  )

  useSSE({
    url: jobId ? JOBS_EVENTS(jobId) : '',
    enabled: !!jobId && isActive,
    onMessage: useCallback((event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data)
        setSSEProgress({
          progress: data.progress ?? 0,
          progress_message: data.progress_message ?? data.message ?? '',
          status: data.status ?? 'running',
          result: data.result,
          error: data.error,
        })
      } catch {
        // ignore malformed SSE
      }
    }, []),
  })

  // Merge SSE + polling: SSE takes priority when available
  const polled = jobQuery.data?.data as JobRecord | undefined
  const progress: JobProgress = sseProgress ?? {
    progress: polled?.progress ?? 0,
    progress_message: polled?.progress_message ?? '',
    status: polled?.status ?? 'queued',
    result: polled?.result,
    error: polled?.error,
  }

  return {
    ...progress,
    isActive,
    isLoading: jobQuery.isLoading,
    meta: jobQuery.data?.meta,
  }
}
