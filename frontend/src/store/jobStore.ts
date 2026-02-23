import { create } from 'zustand'
import type { JobStatus } from '@/types/jobs'

interface ActiveJob {
  jobId: string
  jobType: string
  progress: number
  message: string
  status: JobStatus
}

interface JobStoreState {
  activeJobs: Map<string, ActiveJob>
  addJob: (jobId: string, jobType: string) => void
  updateProgress: (jobId: string, progress: number, message: string, status: JobStatus) => void
  removeJob: (jobId: string) => void
}

export const useJobStore = create<JobStoreState>((set) => ({
  activeJobs: new Map(),

  addJob: (jobId, jobType) =>
    set((state) => {
      const next = new Map(state.activeJobs)
      next.set(jobId, { jobId, jobType, progress: 0, message: 'Queued', status: 'pending' })
      return { activeJobs: next }
    }),

  updateProgress: (jobId, progress, message, status) =>
    set((state) => {
      const next = new Map(state.activeJobs)
      const existing = next.get(jobId)
      if (existing) {
        next.set(jobId, { ...existing, progress, message, status })
      }
      return { activeJobs: next }
    }),

  removeJob: (jobId) =>
    set((state) => {
      const next = new Map(state.activeJobs)
      next.delete(jobId)
      return { activeJobs: next }
    }),
}))
